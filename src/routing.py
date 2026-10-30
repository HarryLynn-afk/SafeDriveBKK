"""
Route generation and external routing API clients.

Responsibility:
  - Wrap the OSRM and Google Routes APIs.
  - Generate fast / safe route geometries.
  - Handle the fallback chain: live OSRM → cached preset → dense interpolation.

Google Routes requests run server-side using the configured server API key.
"""

from __future__ import annotations

import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests
from googlemaps.convert import decode_polyline

from . import config
from .config import (
    AVG_SPEED_KMH,
    BKK_LAT_MAX,
    BKK_LAT_MIN,
    BKK_LON_MAX,
    BKK_LON_MIN,
    OSRM_BASE_URL,
    ROUTE_PRESETS_PATH,
)
from .utils.geo import find_grid_for_point, haversine_km

logger = logging.getLogger(__name__)

# Max safe-route detour midpoints evaluated per request. Each one issues two
# OSRM calls, so this caps the synchronous fan-out (and public-OSRM load).
_MAX_DETOUR_CANDIDATES = 16

# ---------------------------------------------------------------------------
# ResolvedRoute — the single return type of resolve_route()
# ---------------------------------------------------------------------------

@dataclass
class ResolvedRoute:
    waypoints: list[list[float]]
    distance_km: float
    duration_min: int
    traffic_source: str  # "google_maps_traffic" | "google_maps" | "osrm" | "cached_osrm" | "estimated"


# ---------------------------------------------------------------------------
# Route-preset cache (loaded once from static/route_presets.json)
# ---------------------------------------------------------------------------
_route_presets_cache: dict | None = None


def _load_route_presets() -> dict:
    global _route_presets_cache
    if _route_presets_cache is not None:
        return _route_presets_cache
    try:
        with open(ROUTE_PRESETS_PATH) as fh:
            _route_presets_cache = json.load(fh)
    except FileNotFoundError:
        logger.warning("Route presets file not found at %s — cached routes disabled", ROUTE_PRESETS_PATH)
        _route_presets_cache = {}
    except Exception:
        logger.exception("Failed to load route presets from %s", ROUTE_PRESETS_PATH)
        _route_presets_cache = {}
    return _route_presets_cache


def reload_route_presets() -> None:
    """Discard the in-process presets cache; next find_cached_route() re-reads the file."""
    global _route_presets_cache
    _route_presets_cache = None


def find_cached_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    threshold_km: float = 2.0,
) -> dict | None:
    """Return cached route-preset waypoints if start/end are close enough."""
    for _name, route in _load_route_presets().items():
        r_start = route.get("start", {})
        r_end = route.get("end", {})
        if not r_start or not r_end:
            continue
        if (
            haversine_km(start_lat, start_lon, r_start["lat"], r_start["lon"]) <= threshold_km
            and haversine_km(end_lat, end_lon, r_end["lat"], r_end["lon"]) <= threshold_km
        ):
            wps = route.get("waypoints", [])
            if len(wps) >= 10:
                return {
                    "waypoints": wps,
                    "distance_km": route.get("distance", 0),
                    "duration_min": route.get("duration", 0),
                }
    return None


# ---------------------------------------------------------------------------
# Interpolation fallback
# ---------------------------------------------------------------------------

def generate_dense_interpolation(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    num_points: int = 80,
) -> list[list[float]]:
    """Dense curved path when no OSRM or cached route is available."""
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        curve = math.sin(t * math.pi) * 0.003
        dlat = end_lat - start_lat
        dlon = end_lon - start_lon
        perp_lat = -dlon
        perp_lon = dlat
        length = math.sqrt(perp_lat ** 2 + perp_lon ** 2) or 1.0
        perp_lat /= length
        perp_lon /= length
        lat = start_lat + dlat * t + perp_lat * curve
        lon = start_lon + dlon * t + perp_lon * curve
        points.append([round(lat, 6), round(lon, 6)])
    return points


# ---------------------------------------------------------------------------
# Google Routes API client wrappers
# ---------------------------------------------------------------------------

def _compute_google_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    alternatives: bool = False,
) -> list[dict]:
    """Fetch Routes API results in the existing internal format, or [] on failure."""
    if not config.GOOGLE_MAPS_SERVER_KEY:
        return []
    fields = "routes.duration,routes.distanceMeters,fallbackInfo"
    if alternatives:
        fields += ",routes.polyline.encodedPolyline,routes.description"
    try:
        response = requests.post(
            "https://routes.googleapis.com/directions/v2:computeRoutes",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": config.GOOGLE_MAPS_SERVER_KEY,
                "X-Goog-FieldMask": fields,
            },
            json={
                "origin": {"location": {"latLng": {"latitude": origin_lat, "longitude": origin_lon}}},
                "destination": {"location": {"latLng": {"latitude": dest_lat, "longitude": dest_lon}}},
                "travelMode": "DRIVE",
                "routingPreference": "TRAFFIC_AWARE",
                "computeAlternativeRoutes": alternatives,
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        has_traffic = data.get("fallbackInfo", {}).get("routingMode") != "FALLBACK_TRAFFIC_UNAWARE"
        routes = []
        for route in data.get("routes", []):
            duration = route["duration"]
            distance = route["distanceMeters"]
            if not re.fullmatch(r"\d+(?:\.\d{1,9})?s", duration):
                raise ValueError("Invalid route duration")
            if type(distance) is not int or distance < 0:
                raise ValueError("Invalid route distance")
            parsed = {
                "duration_minutes": round(float(duration[:-1]) / 60),
                "distance_km": round(distance / 1000, 2),
                "has_traffic": has_traffic,
            }
            if alternatives:
                decoded = decode_polyline(route["polyline"]["encodedPolyline"])
                if len(decoded) < 2:
                    raise ValueError("Route geometry needs at least two points")
                parsed["waypoints"] = [[point["lat"], point["lng"]] for point in decoded]
                parsed["summary"] = route.get("description", "")
            routes.append(parsed)
        return routes
    except Exception as exc:
        logger.warning("Google Routes API request failed (%s)", type(exc).__name__)
        return []


def get_google_maps_travel_time(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> dict | None:
    """Return traffic-aware travel time from Routes API, or None if unavailable."""
    routes = _compute_google_routes(origin_lat, origin_lon, dest_lat, dest_lon)
    return routes[0] if routes else None


def get_google_maps_routes(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
) -> list[dict]:
    """Return alternative driving routes from Google Routes API."""
    return _compute_google_routes(origin_lat, origin_lon, dest_lat, dest_lon, alternatives=True)


# ---------------------------------------------------------------------------
# OSRM client wrappers
# ---------------------------------------------------------------------------

def get_osrm_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> dict | None:
    """Fetch a single route from OSRM, or None on failure."""
    url = (
        f"{OSRM_BASE_URL}/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    )
    try:
        data = requests.get(url, timeout=5).json()
        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            return {
                "waypoints": [[c[1], c[0]] for c in route["geometry"]["coordinates"]],
                "distance_km": route["distance"] / 1000,
                "duration_min": int(route["duration"] / 60),
            }
    except Exception:
        logger.debug("OSRM request failed", exc_info=True)
    return None


def get_osrm_route_from_waypoints(waypoints: list) -> dict | None:
    """Fetch an OSRM route passing through multiple waypoints."""
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        return None
    try:
        coords = []
        for point in waypoints:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                coords.append(f"{float(point[1])},{float(point[0])}")
        if len(coords) < 2:
            return None
        url = f"{OSRM_BASE_URL}/route/v1/driving/{';'.join(coords)}"
        data = requests.get(
            url,
            params={"overview": "full", "geometries": "geojson", "steps": "true"},
            timeout=5,
        ).json()
        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            return {
                "waypoints": [[c[1], c[0]] for c in route["geometry"]["coordinates"]],
                "distance_km": route["distance"] / 1000,
                "duration_min": int(route["duration"] / 60),
            }
    except Exception:
        logger.debug("OSRM request failed", exc_info=True)
    return None


def get_osrm_alternatives(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    num_alternatives: int = 3,
) -> list[dict]:
    """Return up to num_alternatives routes from OSRM."""
    url = (
        f"{OSRM_BASE_URL}/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&alternatives={num_alternatives}"
    )
    try:
        data = requests.get(url, timeout=5).json()
        if data.get("code") == "Ok":
            routes = []
            for route in data.get("routes", []):
                routes.append({
                    "waypoints": [[c[1], c[0]] for c in route["geometry"]["coordinates"]],
                    "distance_km": route["distance"] / 1000,
                    "duration_min": int(route["duration"] / 60),
                })
            return routes
    except Exception:
        logger.debug("OSRM request failed", exc_info=True)

    cached = find_cached_route(start_lat, start_lon, end_lat, end_lon)
    return [cached] if cached else []


# ---------------------------------------------------------------------------
# Unified route resolver (geometry + timing, full fallback chain)
# ---------------------------------------------------------------------------

def resolve_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    client_waypoints: list[list[float]] | None = None,
) -> ResolvedRoute:
    """Resolve a route geometry and travel time using the full fallback chain.

    Priority:
      1. Client-supplied waypoints (geometry only; timing still fetched from GMaps/estimate)
      2. Live OSRM (geometry + timing)
      3. Cached route preset (geometry + estimated timing)
      4. Dense interpolation (geometry + estimated timing)

    Google Maps timing is layered on top whenever the client is available,
    regardless of which geometry source was used.

    Always returns a ResolvedRoute — never raises.
    """
    waypoints: list[list[float]]
    distance_km: float
    duration_min: int
    traffic_source: str

    if client_waypoints and len(client_waypoints) >= 2:
        waypoints = client_waypoints
        distance_km = sum(
            haversine_km(waypoints[i - 1][0], waypoints[i - 1][1], waypoints[i][0], waypoints[i][1])
            for i in range(1, len(waypoints))
        )
        duration_min = int((distance_km / AVG_SPEED_KMH) * 60) if distance_km > 0 else 0
        traffic_source = "estimated"
    else:
        osrm = get_osrm_route(start_lat, start_lon, end_lat, end_lon)
        if osrm:
            waypoints = osrm["waypoints"]
            distance_km = float(osrm["distance_km"])
            duration_min = int(osrm["duration_min"])
            traffic_source = "osrm"
        else:
            cached = find_cached_route(start_lat, start_lon, end_lat, end_lon)
            if cached:
                waypoints = cached["waypoints"]
                distance_km = float(cached.get("distance_km") or 0) or sum(
                    haversine_km(cached["waypoints"][i - 1][0], cached["waypoints"][i - 1][1],
                                 cached["waypoints"][i][0], cached["waypoints"][i][1])
                    for i in range(1, len(cached["waypoints"]))
                )
                duration_min = int(cached.get("duration_min") or 0) or int((distance_km / AVG_SPEED_KMH) * 60)
                traffic_source = "cached_osrm"
            else:
                waypoints = generate_dense_interpolation(start_lat, start_lon, end_lat, end_lon)
                distance_km = sum(
                    haversine_km(waypoints[i - 1][0], waypoints[i - 1][1], waypoints[i][0], waypoints[i][1])
                    for i in range(1, len(waypoints))
                )
                duration_min = int((distance_km / AVG_SPEED_KMH) * 60)
                traffic_source = "estimated"

    # Layer GMaps timing on top whenever available
    gmaps = get_google_maps_travel_time(start_lat, start_lon, end_lat, end_lon)
    if gmaps:
        duration_min = gmaps["duration_minutes"]
        if gmaps["distance_km"] and traffic_source == "estimated":
            distance_km = gmaps["distance_km"]
        traffic_source = "google_maps_traffic" if gmaps["has_traffic"] else "google_maps"

    return ResolvedRoute(
        waypoints=waypoints,
        distance_km=round(distance_km, 2),
        duration_min=duration_min,
        traffic_source=traffic_source,
    )


# ---------------------------------------------------------------------------
# High-level route generators
# ---------------------------------------------------------------------------

def _sample_waypoints(waypoints: list[list[float]], max_points: int = 100) -> list[list[float]]:
    sampled = waypoints[:: max(1, len(waypoints) // max_points)]
    if sampled[-1] != waypoints[-1]:
        sampled.append(waypoints[-1])
    return sampled


def generate_fast_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> list[list[float]]:
    """Return the fastest route geometry.

    Fallback chain: live OSRM → cached preset → dense interpolation.
    """
    osrm = get_osrm_route(start_lat, start_lon, end_lat, end_lon)
    if osrm:
        return _sample_waypoints(osrm["waypoints"])

    cached = find_cached_route(start_lat, start_lon, end_lat, end_lon)
    if cached:
        return _sample_waypoints(cached["waypoints"])

    return generate_dense_interpolation(start_lat, start_lon, end_lat, end_lon)


def _build_detour_path(
    start_lat: float,
    start_lon: float,
    mid_lat: float,
    mid_lon: float,
    end_lat: float,
    end_lon: float,
    points_per_leg: int = 40,
) -> list[list[float]]:
    """Build a dense interpolated detour: start → midpoint → end."""
    path: list[list[float]] = []
    for seg_start, seg_end in [
        ((start_lat, start_lon), (mid_lat, mid_lon)),
        ((mid_lat, mid_lon), (end_lat, end_lon)),
    ]:
        offset = 1 if seg_start == (mid_lat, mid_lon) else 0
        for i in range(offset, points_per_leg + 1):
            t = i / points_per_leg
            lat = seg_start[0] + (seg_end[0] - seg_start[0]) * t
            lon = seg_start[1] + (seg_end[1] - seg_start[1]) * t
            curve = math.sin(t * math.pi) * 0.002
            dlat = seg_end[0] - seg_start[0]
            dlon = seg_end[1] - seg_start[1]
            length = math.sqrt(dlat ** 2 + dlon ** 2) or 1.0
            lat += (-dlon / length) * curve
            lon += (dlat / length) * curve
            path.append([round(lat, 6), round(lon, 6)])
    return path


def get_route_alternatives(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    grid_df,
) -> tuple[list[dict], int]:
    """Return scored route alternatives between two points.

    Tries Google Routes API first (multiple alternatives). Falls back to
    a fast + safe pair generated locally, with GMaps timing layered on top of
    the fast leg when available.

    Returns (scored_routes, alternatives_analyzed). Each route dict is
    ready to feed into `risk.rank_and_annotate_routes()`.
    """
    # Imported here to avoid circular import (risk imports nothing from routing).
    from .risk import evaluate_route, score_route

    gmaps_routes = get_google_maps_routes(start_lat, start_lon, end_lat, end_lon)

    if gmaps_routes:
        scored_routes: list[dict] = []
        for gr in gmaps_routes:
            wps = gr["waypoints"]
            sampled = wps[:: max(1, len(wps) // 80)]
            if sampled[-1] != wps[-1]:
                sampled.append(wps[-1])

            risk = score_route(sampled, grid_df)
            scored_routes.append({
                "waypoints": wps,
                "distance_km": gr["distance_km"],
                "time_min": gr["duration_minutes"],
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "high_risk_zones": risk.get("high_risk_zones", 0),
                "high_risk_km": risk.get("high_risk_km", 0),
                "medium_risk_zones": risk.get("medium_risk_zones", 0),
                "matched_grids": risk.get("matched_grids", 0),
                "evaluated_waypoints": risk.get("evaluated_waypoints", len(sampled)),
                "traffic_source": "google_maps_traffic" if gr["has_traffic"] else "google_maps",
                "summary": gr.get("summary", ""),
            })
        return scored_routes, len(gmaps_routes)

    fast_wps = generate_fast_route(start_lat, start_lon, end_lat, end_lon)
    safe_wps = generate_safe_route(start_lat, start_lon, end_lat, end_lon, grid_df, fast_waypoints=fast_wps)
    route_fast = evaluate_route(fast_wps, grid_df)
    route_safe = evaluate_route(safe_wps, grid_df)

    gmaps_time = get_google_maps_travel_time(start_lat, start_lon, end_lat, end_lon)
    if gmaps_time:
        route_fast["distance_km"] = gmaps_time["distance_km"]
        route_fast["time_min"] = gmaps_time["duration_minutes"]
        route_fast["traffic_source"] = "google_maps_traffic" if gmaps_time["has_traffic"] else "google_maps"
    else:
        route_fast["traffic_source"] = "estimated"
    route_safe["traffic_source"] = "estimated"

    return [route_fast, route_safe], 1


def generate_safe_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    grid_df,
    fast_waypoints: list[list[float]] | None = None,
) -> list[list[float]]:
    """Return the safest route geometry.

    Strategy:
      1. Score OSRM alternatives — pick lowest-risk one.
      2. Try OSRM-detour through low-risk midpoints (parallel requests).
      3. Fall back to interpolated detours when OSRM is unavailable.
      4. Return the fast route if nothing safer is found.
    """
    # Import here to avoid a circular import (risk imports nothing from routing)
    from .risk import calculate_route_risk_density_aware

    if fast_waypoints is None:
        fast_waypoints = generate_fast_route(start_lat, start_lon, end_lat, end_lon)
    fast_risk_score = int(calculate_route_risk_density_aware(fast_waypoints, grid_df)["risk_score"])

    # ---- Strategy 1: OSRM alternatives ----
    alternatives = get_osrm_alternatives(start_lat, start_lon, end_lat, end_lon, num_alternatives=5)
    if alternatives:
        best_route = None
        best_risk = float("inf")
        for alt in alternatives:
            sampled = _sample_waypoints(alt["waypoints"])
            score = int(calculate_route_risk_density_aware(sampled, grid_df)["risk_score"])
            if score < fast_risk_score and score < best_risk:
                best_risk = score
                best_route = sampled
        if best_route:
            return best_route

    # ---- Build candidate midpoints ----
    dlat = end_lat - start_lat
    dlon = end_lon - start_lon
    candidate_midpoints: list[tuple[float, float]] = []
    for angle in range(0, 360, 30):
        for offset in [0.015, 0.03, 0.05]:
            mlat = start_lat + dlat * 0.5 + offset * math.cos(math.radians(angle))
            mlon = start_lon + dlon * 0.5 + offset * math.sin(math.radians(angle))
            if not (BKK_LAT_MIN <= mlat <= BKK_LAT_MAX and BKK_LON_MIN <= mlon <= BKK_LON_MAX):
                continue
            grid = find_grid_for_point(mlat, mlon, grid_df)
            if grid is not None and int(grid["total_accidents"]) > 5:
                continue
            candidate_midpoints.append((mlat, mlon))

    # Bound the OSRM fan-out: each candidate triggers two routing calls, so cap
    # the count (downsampling evenly to keep angular spread) to limit request
    # latency and load on the public OSRM server.
    if len(candidate_midpoints) > _MAX_DETOUR_CANDIDATES:
        stride = len(candidate_midpoints) // _MAX_DETOUR_CANDIDATES
        candidate_midpoints = candidate_midpoints[::stride][:_MAX_DETOUR_CANDIDATES]

    # ---- Strategy 2: OSRM detour legs (parallel) ----
    best_detour: list[list[float]] | None = None
    best_detour_risk = float("inf")
    osrm_found = False

    if candidate_midpoints:
        with ThreadPoolExecutor(max_workers=min(16, len(candidate_midpoints) * 2)) as pool:
            futures = {
                (mlat, mlon): (
                    pool.submit(get_osrm_route, start_lat, start_lon, mlat, mlon),
                    pool.submit(get_osrm_route, mlat, mlon, end_lat, end_lon),
                )
                for mlat, mlon in candidate_midpoints
            }
            for (mlat, mlon), (f1, f2) in futures.items():
                r1, r2 = f1.result(), f2.result()
                if r1 and r2:
                    osrm_found = True
                    combined = r1["waypoints"][:-1] + r2["waypoints"]
                    sampled = _sample_waypoints(combined)
                    score = int(calculate_route_risk_density_aware(sampled, grid_df)["risk_score"])
                    if score < fast_risk_score and score < best_detour_risk:
                        best_detour_risk = score
                        best_detour = sampled

    # ---- Strategy 3: Interpolated detours (OSRM unavailable) ----
    if not osrm_found and candidate_midpoints:
        for mlat, mlon in candidate_midpoints:
            path = _build_detour_path(start_lat, start_lon, mlat, mlon, end_lat, end_lon)
            score = int(calculate_route_risk_density_aware(path, grid_df)["risk_score"])
            if score < fast_risk_score and score < best_detour_risk:
                best_detour_risk = score
                best_detour = path

    return best_detour if best_detour is not None else fast_waypoints
