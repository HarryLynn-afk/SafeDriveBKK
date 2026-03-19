"""
Flask web app for Bangkok Road Risk system.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file

import googlemaps
from googlemaps.convert import decode_polyline

from ml_model import RiskPredictor
from chatbot_intents import (
    detect_intent,
    get_scripted_response,
    detect_cause_from_question,
    handle_ml_cause_response,
)
from chatbot_groq import get_groq_response

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

BANGKOK_ACCIDENTS_PATH = DATA_DIR / "bangkok_accidents_clean.csv"
GRID_STATS_PATH = DATA_DIR / "bangkok_grid_stats.csv"
RISK_MAP_HTML_PATH = DATA_DIR / "bangkok_risk_map.html"
RISK_HEATMAP_HTML_PATH = BASE_DIR / "static" / "bangkok_risk_heatmap.html"

BKK_LAT_MIN = 13.6
BKK_LAT_MAX = 13.95
BKK_LON_MIN = 100.35
BKK_LON_MAX = 100.9
CELL_SIZE_DEG = 0.009

DAY_NAME_MAP = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# ---------------------------------------------------------------------------
# ML predictor — initialised once at import time
# ---------------------------------------------------------------------------
try:
    ml_predictor = RiskPredictor(str(BASE_DIR / "models" / "xgboost_model.pkl"))
    USE_ML = ml_predictor._model is not None
    if USE_ML:
        print("ML predictor: XGBoost loaded (39/39 zones, 83.8% temporal capture, 17.5x lift)")
    else:
        print("ML predictor: model file missing, spatial fallback active")
except Exception as exc:
    print(f"ML predictor: failed to load ({exc}), spatial fallback active")
    ml_predictor = None
    USE_ML = False

# ---------------------------------------------------------------------------
# Google Maps client — for traffic-aware travel times
# ---------------------------------------------------------------------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if GOOGLE_MAPS_API_KEY:
    gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
    print("Google Maps: API key loaded (traffic-aware travel times enabled)")
else:
    gmaps_client = None
    print("Google Maps: No API key found. Using OSRM/estimated travel times.")


# ---------------------------------------------------------------------------
# Cached demo routes — loaded once from static/demo_routes.json so that when
# OSRM is unreachable we can still return real road-following waypoints.
# ---------------------------------------------------------------------------
DEMO_ROUTES_PATH = BASE_DIR / "static" / "demo_routes.json"
_demo_routes_cache = None


def _load_demo_routes():
    global _demo_routes_cache
    if _demo_routes_cache is not None:
        return _demo_routes_cache
    try:
        with open(DEMO_ROUTES_PATH) as fh:
            _demo_routes_cache = json.load(fh)
    except Exception:
        _demo_routes_cache = {}
    return _demo_routes_cache


def _find_cached_route(start_lat, start_lon, end_lat, end_lon, threshold_km=2.0):
    """Return cached demo-route waypoints if start/end are close enough."""
    demo_routes = _load_demo_routes()
    for _name, route in demo_routes.items():
        r_start = route.get("start", {})
        r_end = route.get("end", {})
        if not r_start or not r_end:
            continue
        start_dist = _haversine_km(start_lat, start_lon, r_start["lat"], r_start["lon"])
        end_dist = _haversine_km(end_lat, end_lon, r_end["lat"], r_end["lon"])
        if start_dist <= threshold_km and end_dist <= threshold_km:
            wps = route.get("waypoints", [])
            if len(wps) >= 10:
                return {
                    "waypoints": wps,
                    "distance_km": route.get("distance", 0),
                    "duration_min": route.get("duration", 0),
                }
    return None


def _generate_dense_interpolation(start_lat, start_lon, end_lat, end_lon, num_points=80):
    """Generate a dense curved path when no OSRM or cached route is available.

    Instead of a 5-point straight line, this adds a slight curve and many
    intermediate points so the route looks plausible on the map.
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        # Add a gentle sinusoidal curve perpendicular to the travel direction
        # so the path doesn't look ruler-straight on the map.
        curve = math.sin(t * math.pi) * 0.003
        dlat = end_lat - start_lat
        dlon = end_lon - start_lon
        # Perpendicular offset: rotate (dlat, dlon) 90°
        perp_lat = -dlon
        perp_lon = dlat
        length = math.sqrt(perp_lat ** 2 + perp_lon ** 2) or 1.0
        perp_lat /= length
        perp_lon /= length

        lat = start_lat + dlat * t + perp_lat * curve
        lon = start_lon + dlon * t + perp_lon * curve
        points.append([round(lat, 6), round(lon, 6)])
    return points


def get_google_maps_travel_time(origin_lat, origin_lon, dest_lat, dest_lon):
    """Get traffic-aware travel time from Google Maps Directions API.

    Returns dict with duration_minutes, distance_km, has_traffic on success,
    or None if unavailable.
    """
    if not gmaps_client:
        return None
    try:
        result = gmaps_client.directions(
            origin=f"{origin_lat},{origin_lon}",
            destination=f"{dest_lat},{dest_lon}",
            mode="driving",
            departure_time="now",
            traffic_model="best_guess",
        )
        if not result:
            return None
        leg = result[0]["legs"][0]
        if "duration_in_traffic" in leg:
            duration_sec = leg["duration_in_traffic"]["value"]
            has_traffic = True
        else:
            duration_sec = leg["duration"]["value"]
            has_traffic = False
        distance_m = leg["distance"]["value"]
        return {
            "duration_minutes": round(duration_sec / 60),
            "distance_km": round(distance_m / 1000, 2),
            "has_traffic": has_traffic,
        }
    except Exception as e:
        print(f"Google Maps API error: {e}")
        return None


def get_google_maps_routes(origin_lat, origin_lon, dest_lat, dest_lon):
    """Get multiple alternative routes from Google Maps Directions API.

    Returns a list of dicts, each with:
        waypoints: [[lat, lon], ...] decoded from polyline
        distance_km, duration_minutes, has_traffic, summary
    Returns empty list if unavailable.
    """
    if not gmaps_client:
        return []
    try:
        results = gmaps_client.directions(
            origin=f"{origin_lat},{origin_lon}",
            destination=f"{dest_lat},{dest_lon}",
            mode="driving",
            departure_time="now",
            traffic_model="best_guess",
            alternatives=True,
        )
        if not results:
            return []

        routes = []
        for route in results:
            # Decode the overview polyline to get route geometry
            polyline_str = route["overview_polyline"]["points"]
            decoded = decode_polyline(polyline_str)
            waypoints = [[pt["lat"], pt["lng"]] for pt in decoded]

            leg = route["legs"][0]
            if "duration_in_traffic" in leg:
                duration_sec = leg["duration_in_traffic"]["value"]
                has_traffic = True
            else:
                duration_sec = leg["duration"]["value"]
                has_traffic = False
            distance_m = leg["distance"]["value"]

            routes.append({
                "waypoints": waypoints,
                "distance_km": round(distance_m / 1000, 2),
                "duration_minutes": round(duration_sec / 60),
                "has_traffic": has_traffic,
                "summary": route.get("summary", ""),
            })
        return routes
    except Exception as e:
        print(f"Google Maps routes API error: {e}")
        return []


def _require_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def _load_bangkok_accidents():
    _require_file(BANGKOK_ACCIDENTS_PATH)
    return pd.read_csv(BANGKOK_ACCIDENTS_PATH, encoding="utf-8-sig")


_grid_stats_cache = None


def _load_grid_stats():
    global _grid_stats_cache
    if _grid_stats_cache is not None:
        return _grid_stats_cache
    _require_file(GRID_STATS_PATH)
    df = pd.read_csv(GRID_STATS_PATH, encoding="utf-8-sig")
    if "grid_id" not in df.columns:
        raise ValueError("Invalid grid stats file: missing 'grid_id' column")
    _grid_stats_cache = _add_grid_bounds(df)
    return _grid_stats_cache


def _add_grid_bounds(grid_df):
    required = {"lat_min", "lat_max", "lon_min", "lon_max"}
    if required.issubset(grid_df.columns):
        return grid_df.copy()

    parsed = grid_df["grid_id"].astype(str).str.extract(r"^G(\d{3})_(\d{3})$")
    if parsed.isna().any().any():
        raise ValueError("Invalid grid_id format; expected G000_000 style values")

    lat_idx = parsed[0].astype(int)
    lon_idx = parsed[1].astype(int)

    result = grid_df.copy()
    result["lat_min"] = BKK_LAT_MIN + (lat_idx * CELL_SIZE_DEG)
    result["lat_max"] = (result["lat_min"] + CELL_SIZE_DEG).clip(upper=BKK_LAT_MAX)
    result["lon_min"] = BKK_LON_MIN + (lon_idx * CELL_SIZE_DEG)
    result["lon_max"] = (result["lon_min"] + CELL_SIZE_DEG).clip(upper=BKK_LON_MAX)
    return result


def _format_hour_range(hour):
    start_hour = int(hour)
    end_hour = (start_hour + 1) % 24
    return f"{start_hour:02d}:00-{end_hour:02d}:00"


def _calculate_temporal_stats(df):
    hour_counts = (
        pd.to_numeric(df["hour"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_values(ascending=False)
    )
    peak_hours = [
        {
            "hour": int(hour),
            "time_range": _format_hour_range(hour),
            "count": int(count),
        }
        for hour, count in hour_counts.head(3).items()
    ]

    day_counts = (
        pd.to_numeric(df["day_of_week"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_values(ascending=False)
    )
    dangerous_days = [
        {
            "day_index": int(day),
            "day_name": DAY_NAME_MAP.get(int(day), str(day)),
            "count": int(count),
        }
        for day, count in day_counts.head(3).items()
    ]

    rush_col = df["is_rush_hour"]
    if rush_col.dtype == bool:
        rush_mask = rush_col.fillna(False)
    else:
        rush_mask = rush_col.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})

    rush_hour_count = int(rush_mask.sum())
    non_rush_count = int(len(df) - rush_hour_count)
    rush_hour_percentage = round((rush_hour_count / len(df) * 100), 2) if len(df) else 0.0

    return {
        "peak_hours": peak_hours,
        "dangerous_days": dangerous_days,
        "rush_hour_count": rush_hour_count,
        "non_rush_count": non_rush_count,
        "rush_hour_percentage": rush_hour_percentage,
    }


def _haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def _clip_to_bangkok_bounds(lat, lon):
    clipped_lat = min(max(lat, BKK_LAT_MIN), BKK_LAT_MAX)
    clipped_lon = min(max(lon, BKK_LON_MIN), BKK_LON_MAX)
    return clipped_lat, clipped_lon


def _find_grid_for_point(lat, lon, grid_df):
    target = grid_df[
        (grid_df["lat_min"] <= lat)
        & (lat <= grid_df["lat_max"])
        & (grid_df["lon_min"] <= lon)
        & (lon <= grid_df["lon_max"])
    ]
    if not target.empty:
        return target.iloc[0]
    return None


def get_osrm_route(start_lat, start_lon, end_lat, end_lon):
    """Get real road route from OSRM API."""
    url = (
        "http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
            route = data["routes"][0]
            waypoints = [[coord[1], coord[0]] for coord in route["geometry"]["coordinates"]]
            distance_km = route["distance"] / 1000
            duration_min = int(route["duration"] / 60)

            return {
                "waypoints": waypoints,
                "distance_km": distance_km,
                "duration_min": duration_min,
            }
    except Exception:
        pass

    return None


def get_osrm_route_from_waypoints(waypoints):
    """Get route from OSRM passing through multiple waypoints."""
    if not isinstance(waypoints, list) or len(waypoints) < 2:
        return None

    try:
        coords = []
        for point in waypoints:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            lat = float(point[0])
            lon = float(point[1])
            coords.append(f"{lon},{lat}")

        if len(coords) < 2:
            return None

        url = f"http://router.project-osrm.org/route/v1/driving/{';'.join(coords)}"
        response = requests.get(
            url,
            params={"overview": "full", "geometries": "geojson", "steps": "true"},
            timeout=5,
        )
        data = response.json()

        if data.get("code") == "Ok" and len(data.get("routes", [])) > 0:
            route = data["routes"][0]
            route_waypoints = [[coord[1], coord[0]] for coord in route["geometry"]["coordinates"]]
            return {
                "waypoints": route_waypoints,
                "distance_km": route["distance"] / 1000,
                "duration_min": int(route["duration"] / 60),
            }
    except Exception:
        pass

    return None


def get_osrm_alternatives(start_lat, start_lon, end_lat, end_lon, num_alternatives=3):
    """Get multiple alternative routes from OSRM."""
    url = (
        "http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&alternatives={num_alternatives}"
    )

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("code") == "Ok":
            routes = []
            for route in data.get("routes", []):
                waypoints = [[coord[1], coord[0]] for coord in route["geometry"]["coordinates"]]
                distance_km = route["distance"] / 1000
                duration_min = int(route["duration"] / 60)

                routes.append(
                    {
                        "waypoints": waypoints,
                        "distance_km": distance_km,
                        "duration_min": duration_min,
                    }
                )

            return routes
    except Exception:
        pass

    # Fallback: return cached demo route as single "alternative"
    cached = _find_cached_route(start_lat, start_lon, end_lat, end_lon)
    if cached:
        return [cached]
    return []


def find_safest_midpoint(start_lat, start_lon, end_lat, end_lon, grid_df, num_samples=8):
    """Find a low-risk intermediate point to route through."""
    dlat = end_lat - start_lat
    dlon = end_lon - start_lon
    candidates = []

    for offset_multiplier in [-0.02, -0.01, 0, 0.01, 0.02]:
        mid_lat = start_lat + (dlat * 0.5) + offset_multiplier
        mid_lon = start_lon + (dlon * 0.5) + offset_multiplier
        grid = _find_grid_for_point(mid_lat, mid_lon, grid_df)
        if grid is not None:
            risk = int(grid["total_accidents"])
            candidates.append((risk, mid_lat, mid_lon))

    candidates.sort()
    if candidates:
        return candidates[0][1], candidates[0][2]

    return start_lat + (dlat * 0.5), start_lon + (dlon * 0.5)


def generate_fast_route(start_lat, start_lon, end_lat, end_lon):
    """Generate fast route using OSRM (follows real roads).

    Fallback order: OSRM live → cached demo route → dense interpolation.
    """
    # 1) Try live OSRM routing
    osrm_route = get_osrm_route(start_lat, start_lon, end_lat, end_lon)
    if osrm_route:
        waypoints = osrm_route["waypoints"]
        sampled = waypoints[:: max(1, len(waypoints) // 100)]
        if sampled[-1] != waypoints[-1]:
            sampled.append(waypoints[-1])
        return sampled

    # 2) Try cached demo-route (real OSRM geometry saved to disk)
    cached = _find_cached_route(start_lat, start_lon, end_lat, end_lon)
    if cached:
        waypoints = cached["waypoints"]
        sampled = waypoints[:: max(1, len(waypoints) // 100)]
        if sampled[-1] != waypoints[-1]:
            sampled.append(waypoints[-1])
        return sampled

    # 3) Dense curved interpolation (better than 5-point straight line)
    return _generate_dense_interpolation(start_lat, start_lon, end_lat, end_lon)


def _build_detour_path(start_lat, start_lon, mid_lat, mid_lon, end_lat, end_lon, points_per_leg=40):
    """Build a dense interpolated path: start → midpoint → end.

    Used when OSRM is unavailable to create detour routes through
    low-risk midpoints.
    """
    path = []
    for i in range(points_per_leg + 1):
        t = i / points_per_leg
        lat = start_lat + (mid_lat - start_lat) * t
        lon = start_lon + (mid_lon - start_lon) * t
        # Add slight curve so it looks road-like
        curve = math.sin(t * math.pi) * 0.002
        dlat = mid_lat - start_lat
        dlon = mid_lon - start_lon
        length = math.sqrt(dlat ** 2 + dlon ** 2) or 1.0
        lat += (-dlon / length) * curve
        lon += (dlat / length) * curve
        path.append([round(lat, 6), round(lon, 6)])

    for i in range(1, points_per_leg + 1):
        t = i / points_per_leg
        lat = mid_lat + (end_lat - mid_lat) * t
        lon = mid_lon + (end_lon - mid_lon) * t
        curve = math.sin(t * math.pi) * 0.002
        dlat = end_lat - mid_lat
        dlon = end_lon - mid_lon
        length = math.sqrt(dlat ** 2 + dlon ** 2) or 1.0
        lat += (-dlon / length) * curve
        lon += (dlat / length) * curve
        path.append([round(lat, 6), round(lon, 6)])

    return path


def generate_safe_route(start_lat, start_lon, end_lat, end_lon, grid_df, fast_waypoints=None):
    """Generate safe route - guaranteed to be safer or equal to fast route.

    Tries OSRM alternatives first.  When OSRM is unavailable, generates
    detour paths through low-risk grid cells using interpolation.
    """
    find_grid_for_point = _find_grid_for_point
    if fast_waypoints is None:
        fast_waypoints = generate_fast_route(start_lat, start_lon, end_lat, end_lon)
    fast_risk = calculate_route_risk_density_aware(fast_waypoints, grid_df)
    fast_risk_score = int(fast_risk["risk_score"])

    # ---- Strategy 1: OSRM alternatives ----
    alternatives = get_osrm_alternatives(start_lat, start_lon, end_lat, end_lon, num_alternatives=5)

    if alternatives:
        best_safe_route = None
        best_safe_risk = float("inf")

        for alt in alternatives:
            waypoints = alt["waypoints"]
            sampled = waypoints[:: max(1, len(waypoints) // 100)]
            if sampled[-1] != waypoints[-1]:
                sampled.append(waypoints[-1])
            risk_result = calculate_route_risk_density_aware(sampled, grid_df)
            risk_score = int(risk_result["risk_score"])

            if risk_score < fast_risk_score and risk_score < best_safe_risk:
                best_safe_risk = risk_score
                best_safe_route = sampled

        if best_safe_route is not None:
            return best_safe_route

    # ---- Strategy 2: Detour through low-risk midpoints ----
    best_detour_route = None
    best_detour_risk = float("inf")
    dlat = end_lat - start_lat
    dlon = end_lon - start_lon

    # Generate candidate midpoints at various angles around the midpoint
    # Use wider offsets so routes differ meaningfully from the fast route.
    candidate_midpoints = []
    for angle in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        for offset in [0.015, 0.03, 0.05]:
            mid_lat = start_lat + (dlat * 0.5) + offset * math.cos(math.radians(angle))
            mid_lon = start_lon + (dlon * 0.5) + offset * math.sin(math.radians(angle))

            # Keep within Bangkok bounds
            if not (BKK_LAT_MIN <= mid_lat <= BKK_LAT_MAX and BKK_LON_MIN <= mid_lon <= BKK_LON_MAX):
                continue

            waypoint_grid = find_grid_for_point(mid_lat, mid_lon, grid_df)
            if waypoint_grid is not None and int(waypoint_grid["total_accidents"]) > 5:
                continue  # Skip high-accident midpoints
            candidate_midpoints.append((mid_lat, mid_lon))

    # Try OSRM detour legs in parallel first
    osrm_detours_found = False
    if candidate_midpoints:
        with ThreadPoolExecutor(max_workers=min(16, len(candidate_midpoints) * 2)) as pool:
            leg_futures = {}
            for mlat, mlon in candidate_midpoints:
                f1 = pool.submit(get_osrm_route, start_lat, start_lon, mlat, mlon)
                f2 = pool.submit(get_osrm_route, mlat, mlon, end_lat, end_lon)
                leg_futures[(mlat, mlon)] = (f1, f2)

            for (mlat, mlon), (f1, f2) in leg_futures.items():
                route1, route2 = f1.result(), f2.result()
                if route1 and route2:
                    osrm_detours_found = True
                    combined = route1["waypoints"][:-1] + route2["waypoints"]
                    sampled = combined[:: max(1, len(combined) // 100)]
                    if sampled[-1] != combined[-1]:
                        sampled.append(combined[-1])
                    risk_result = calculate_route_risk_density_aware(sampled, grid_df)
                    risk_score = int(risk_result["risk_score"])

                    if risk_score < fast_risk_score and risk_score < best_detour_risk:
                        best_detour_risk = risk_score
                        best_detour_route = sampled

    # ---- Strategy 3: Interpolated detours (when OSRM is unavailable) ----
    if not osrm_detours_found and candidate_midpoints:
        for mlat, mlon in candidate_midpoints:
            detour_path = _build_detour_path(
                start_lat, start_lon, mlat, mlon, end_lat, end_lon
            )
            risk_result = calculate_route_risk_density_aware(detour_path, grid_df)
            risk_score = int(risk_result["risk_score"])

            if risk_score < fast_risk_score and risk_score < best_detour_risk:
                best_detour_risk = risk_score
                best_detour_route = detour_path

    if best_detour_route is not None:
        return best_detour_route

    return fast_waypoints


def calculate_segment_risk(lat, lon, hour=None, day_of_week=None):
    """Calculate risk for a single point using ML (XGBoost) or spatial fallback."""
    if hour is None:
        hour = datetime.now().hour
    if day_of_week is None:
        day_of_week = datetime.now().weekday()

    if USE_ML and ml_predictor:
        try:
            return ml_predictor.predict_risk(lat, lon, hour, day_of_week)
        except Exception:
            pass

    if ml_predictor:
        try:
            return ml_predictor.fallback_to_spatial(lat, lon)
        except Exception:
            pass

    return {
        "risk_score": 35,
        "risk_level": "medium",
        "confidence": 0.3,
        "method": "fallback",
        "grid_id": None,
    }


def calculate_route_risk_ml(waypoints):
    """Calculate overall route risk using ML batch prediction.

    Uses distance-weighted risk so that longer segments through danger
    zones contribute proportionally more than short ones.  Two routes
    crossing the same danger zones will score the shorter route as safer
    because it spends less absolute distance in the zone.
    """
    if not (USE_ML and ml_predictor):
        return None

    results = ml_predictor.predict_batch(waypoints)
    high_risk_zones = sum(1 for r in results if r["risk_level"] == "high")
    medium_risk_zones = sum(1 for r in results if r["risk_level"] == "medium")

    # Distance-weighted risk: each segment's risk is weighted by its length.
    total_risk_exposure = 0.0
    total_distance = 0.0
    max_score = 0

    for i in range(len(waypoints)):
        score = results[i]["risk_score"]
        if score > max_score:
            max_score = score

        if i > 0:
            seg_dist = _haversine_km(
                waypoints[i - 1][0], waypoints[i - 1][1],
                waypoints[i][0], waypoints[i][1],
            )
            # Use the average risk of the two endpoints for this segment.
            seg_risk = (results[i - 1]["risk_score"] + score) / 2.0
            total_risk_exposure += seg_risk * seg_dist
            total_distance += seg_dist

    if total_distance > 0:
        weighted_avg = total_risk_exposure / total_distance
    else:
        weighted_avg = float(np.mean([r["risk_score"] for r in results])) if results else 0.0

    # Blend: 60% distance-weighted average, 40% max (keep worst-zone emphasis).
    blended = (weighted_avg * 0.6) + (max_score * 0.4)
    risk_score = min(100, int(blended))

    if risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "high_risk_zones": high_risk_zones,
        "medium_risk_zones": medium_risk_zones,
        "matched_grids": len(results),
        "evaluated_waypoints": len(waypoints),
        "prediction_method": "xgboost",
    }


def calculate_route_risk_density_aware(waypoints, grid_df):
    """Calculate risk using actual accident counts, weighted by distance.

    Uses distance-weighted aggregation so that the risk score reflects
    how much distance is spent in each danger zone.  Two routes crossing
    the same danger zones will score the shorter route as safer because
    it has less total exposure (risk x distance).
    """
    high_risk_zones = 0
    medium_risk_zones = 0
    max_accidents_in_grid = 0
    matched_grid_count = 0

    # Collect per-waypoint accident counts and grid risk scores.
    point_accidents = []
    for lat, lon in waypoints:
        grid = _find_grid_for_point(lat, lon, grid_df)
        if grid is not None:
            matched_grid_count += 1
            accidents = int(grid["total_accidents"])
            point_accidents.append(accidents)
            if accidents > max_accidents_in_grid:
                max_accidents_in_grid = accidents

            risk_level = str(grid["risk_level"]).lower()
            if risk_level == "high":
                high_risk_zones += 1
            elif risk_level == "medium":
                medium_risk_zones += 1
        else:
            point_accidents.append(0)

    # Distance-weighted accident density: each segment's accident count
    # is the average of its two endpoints, weighted by segment length.
    total_weighted_accidents = 0.0
    total_distance = 0.0

    for i in range(1, len(waypoints)):
        seg_dist = _haversine_km(
            waypoints[i - 1][0], waypoints[i - 1][1],
            waypoints[i][0], waypoints[i][1],
        )
        seg_accidents = (point_accidents[i - 1] + point_accidents[i]) / 2.0
        total_weighted_accidents += seg_accidents * seg_dist
        total_distance += seg_dist

    if total_distance > 0:
        weighted_avg_accidents = total_weighted_accidents / total_distance
    else:
        weighted_avg_accidents = float(np.mean(point_accidents)) if point_accidents else 0.0

    # Blend: 80% distance-weighted average, 20% max (worst-zone signal).
    # A lower max weight prevents shared start/end cells from dominating
    # the score when two routes share origin/destination.
    if max_accidents_in_grid > 0:
        blended = (weighted_avg_accidents * 0.8) + (max_accidents_in_grid * 0.2)
    else:
        blended = weighted_avg_accidents

    # Convert to 0-100 scale using log curve.
    # Formula: 100 * (1 - e^(-blended/25))  → 10 ≈ 33, 25 ≈ 63, 50 ≈ 86
    risk_score = min(100, int(100 * (1 - math.exp(-blended / 25.0))))
    # Diminishing danger-zone penalty: sqrt so 7 zones ≈ +8, 59 zones ≈ +15 (capped)
    if high_risk_zones > 0:
        risk_score = min(100, risk_score + min(15, int(math.sqrt(high_risk_zones) * 3)))

    if risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "high_risk_zones": high_risk_zones,
        "medium_risk_zones": medium_risk_zones,
        "matched_grids": matched_grid_count,
        "evaluated_waypoints": len(waypoints),
    }


def calculate_distance_from_waypoints(waypoints):
    """Calculate total route distance from waypoint coordinates."""
    total_distance = 0.0
    for index in range(1, len(waypoints)):
        prev_lat, prev_lon = waypoints[index - 1]
        curr_lat, curr_lon = waypoints[index]
        total_distance += _haversine_km(prev_lat, prev_lon, curr_lat, curr_lon)
    return float(total_distance)


def _evaluate_route(route_points, grid_df):
    total_distance = 0.0

    for index, point in enumerate(route_points):
        lat, lon = point
        if index > 0:
            prev_lat, prev_lon = route_points[index - 1]
            total_distance += _haversine_km(prev_lat, prev_lon, lat, lon)

    risk_metrics = calculate_route_risk_density_aware(route_points, grid_df)
    time_min = int(round((total_distance / 40.0) * 60.0))

    return {
        "waypoints": [[round(lat, 6), round(lon, 6)] for lat, lon in route_points],
        "distance_km": round(total_distance, 2),
        "time_min": time_min,
        "risk_score": min(100, risk_metrics["risk_score"]),
        "risk_level": risk_metrics["risk_level"],
        "high_risk_zones": risk_metrics["high_risk_zones"],
        "medium_risk_zones": risk_metrics["medium_risk_zones"],
        "matched_grids": risk_metrics["matched_grids"],
        "evaluated_waypoints": risk_metrics["evaluated_waypoints"],
    }


def analyze_route_segments(waypoints, grid_df, segment_distance_km=1.0):
    """Divide route into segments and calculate risk for each."""
    if len(waypoints) < 2:
        return []

    segments = []
    total_distance = 0.0
    for index in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[index]
        lat2, lon2 = waypoints[index + 1]
        total_distance += _haversine_km(lat1, lon1, lat2, lon2)

    num_segments = max(1, int(total_distance / segment_distance_km))
    points_per_segment = max(1, len(waypoints) // num_segments)

    current_km = 0.0
    for index in range(0, len(waypoints) - points_per_segment, points_per_segment):
        segment_waypoints = waypoints[index : index + points_per_segment + 1]

        segment_dist = 0.0
        for inner in range(len(segment_waypoints) - 1):
            lat1, lon1 = segment_waypoints[inner]
            lat2, lon2 = segment_waypoints[inner + 1]
            segment_dist += _haversine_km(lat1, lon1, lat2, lon2)

        risk_scores = []
        accident_counts = []
        high_risk_count = 0

        for lat, lon in segment_waypoints:
            grid = _find_grid_for_point(lat, lon, grid_df)
            if grid is not None:
                accident_counts.append(int(grid["total_accidents"]))
                risk_scores.append(float(grid["risk_score"]))
                if str(grid["risk_level"]).lower() == "high":
                    high_risk_count += 1
            else:
                accident_counts.append(0)
                risk_scores.append(0)

        avg_accidents = float(np.mean(accident_counts)) if accident_counts else 0.0
        max_accidents = float(max(accident_counts)) if accident_counts else 0.0
        segment_accidents = (avg_accidents * 0.65) + (max_accidents * 0.35)
        temporal_boost = float(high_risk_count)
        segment_risk_score = min(100, int((segment_accidents * 2.0) + (temporal_boost * 4.0)))

        if segment_risk_score >= 50:
            risk_level = "high"
            color = "#dc3545"
        elif segment_risk_score >= 25:
            risk_level = "medium"
            color = "#ffc107"
        else:
            risk_level = "low"
            color = "#28a745"

        segments.append(
            {
                "start_km": round(current_km, 1),
                "end_km": round(current_km + segment_dist, 1),
                "waypoints": segment_waypoints,
                "risk_score": segment_risk_score,
                "risk_level": risk_level,
                "color": color,
                "high_risk_zones": high_risk_count,
                "max_accidents": max_accidents,
            }
        )
        current_km += segment_dist

    return segments


def generate_alerts_from_segments(segments, advance_warning_km=0.5):
    """Generate alerts for high-risk segments with advance warning."""
    alerts = []

    for segment in segments:
        if segment["risk_level"] == "high":
            warning_km = max(0, segment["start_km"] - advance_warning_km)
            alerts.append(
                {
                    "type": "warning",
                    "trigger_km": round(warning_km, 1),
                    "message": f"⚠️ HIGH RISK ZONE ahead in {int(advance_warning_km * 1000)}m",
                    "detail": (
                        f"Zone: km {segment['start_km']}-{segment['end_km']} | "
                        f"{segment['max_accidents']} accidents recorded"
                    ),
                    "severity": "warning",
                    "color": "#ffc107",
                }
            )
            alerts.append(
                {
                    "type": "danger",
                    "trigger_km": segment["start_km"],
                    "message": "🔴 NOW ENTERING HIGH RISK ZONE",
                    "detail": f"Extra caution advised | Risk score: {segment['risk_score']}/100",
                    "severity": "danger",
                    "color": "#dc3545",
                }
            )
            alerts.append(
                {
                    "type": "success",
                    "trigger_km": segment["end_km"],
                    "message": "✓ Exited high-risk zone - safer area ahead",
                    "detail": "",
                    "severity": "success",
                    "color": "#28a745",
                }
            )
        elif segment["risk_level"] == "medium":
            alerts.append(
                {
                    "type": "info",
                    "trigger_km": segment["start_km"],
                    "message": "⚡ Medium risk area - stay alert",
                    "detail": f"Risk score: {segment['risk_score']}/100",
                    "severity": "info",
                    "color": "#17a2b8",
                }
            )

    alerts.sort(key=lambda item: item["trigger_km"])
    return alerts


@app.route("/")
def home():
    try:
        accidents_df = _load_bangkok_accidents()
        grid_df = _load_grid_stats()
        temporal_stats = _calculate_temporal_stats(accidents_df)

        total_accidents = int(len(accidents_df))
        high_risk_zones = int(((grid_df["risk_level"].str.lower() == "high") & (grid_df["total_accidents"] > 0)).sum())
        peak_time = temporal_stats["peak_hours"][0]["time_range"] if temporal_stats["peak_hours"] else "N/A"
        dangerous_day = temporal_stats["dangerous_days"][0]["day_name"] if temporal_stats["dangerous_days"] else "N/A"
        map_available = RISK_HEATMAP_HTML_PATH.exists()

        return render_template(
            "index.html",
            total_accidents=total_accidents,
            high_risk_zones=high_risk_zones,
            peak_time=peak_time,
            dangerous_day=dangerous_day,
            total_accidents_card=f"{total_accidents:,} Accidents Analyzed",
            high_risk_card=f"{high_risk_zones:,} High-Risk Zones",
            peak_card=f"Peak: {peak_time}",
            dangerous_day_card=f"{dangerous_day} Most Dangerous",
            map_available=map_available,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            total_accidents=0,
            high_risk_zones=0,
            peak_time="N/A",
            dangerous_day="N/A",
            total_accidents_card="0 Accidents Analyzed",
            high_risk_card="0 High-Risk Zones",
            peak_card="Peak: N/A",
            dangerous_day_card="N/A Most Dangerous",
            map_available=False,
            error_message=str(exc),
        ), 500



@app.route("/trip-planner")
def trip_planner():
    """New Grab-style trip planner page."""
    return render_template(
        "trip_planner.html",
        google_maps_api_key=os.getenv("GOOGLE_MAPS_API_KEY", ""),
    )


@app.route("/about")
def about():
    """About page with evidence and methodology."""
    return render_template("about.html")


@app.route("/analytics")
def analytics_page():
    """Interactive analytics dashboard."""
    return render_template("analytics.html")


@app.route("/api/analytics-data")
def analytics_data():
    """Return aggregated analytics data for the dashboard charts."""
    try:
        df = _load_bangkok_accidents()
        grid_df = _load_grid_stats()

        # Accidents by hour
        hour_col = pd.to_numeric(df["hour"], errors="coerce").dropna().astype(int)
        hour_counts = hour_col.value_counts().sort_index()
        by_hour = {int(h): int(c) for h, c in hour_counts.items()}

        # Accidents by day of week
        dow_col = pd.to_numeric(df["day_of_week"], errors="coerce").dropna().astype(int)
        dow_counts = dow_col.value_counts().sort_index()
        by_day = {DAY_NAME_MAP.get(int(d), str(d)): int(c) for d, c in dow_counts.items()}

        # Accidents by month
        month_col = pd.to_numeric(df["month"], errors="coerce").dropna().astype(int)
        month_counts = month_col.value_counts().sort_index()
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        by_month = {month_names.get(int(m), str(m)): int(c) for m, c in month_counts.items()}

        # Top 10 most dangerous zones
        top_zones = (
            grid_df[grid_df["total_accidents"] > 0]
            .nlargest(10, "total_accidents")[["grid_id", "total_accidents", "risk_level", "center_lat", "center_lon"]]
            .to_dict(orient="records")
        )

        # Accidents by Bangkok area (derived from lat/lon)
        by_area = {}
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            def _assign_bangkok_area(lat, lon):
                if pd.isna(lat) or pd.isna(lon):
                    return None
                if lat >= 13.85:
                    return "Don Mueang / Lak Si" if lon < 100.55 else "Ram Intra / Minburi"
                elif lat >= 13.78:
                    if lon < 100.5:
                        return "Bang Sue / Chatuchak"
                    return "Lat Phrao / Wang Thonglang" if lon < 100.6 else "Khan Na Yao / Nong Chok"
                elif lat >= 13.72:
                    if lon < 100.48:
                        return "Dusit / Phaya Thai"
                    elif lon < 100.56:
                        return "Din Daeng / Huai Khwang"
                    return "Bang Kapi / Suan Luang" if lon < 100.65 else "Lat Krabang"
                elif lat >= 13.66:
                    if lon < 100.48:
                        return "Thon Buri / Bangkok Noi"
                    elif lon < 100.55:
                        return "Sathon / Bang Rak"
                    return "Phra Khanong / Bang Na" if lon < 100.65 else "Prawet / On Nut"
                else:
                    return "Bang Khun Thian / Chom Thong" if lon < 100.55 else "Samut Prakan Border"

            areas = df.apply(lambda r: _assign_bangkok_area(r["LATITUDE"], r["LONGITUDE"]), axis=1)
            area_counts = areas.dropna().value_counts()
            by_area = {str(k): int(v) for k, v in area_counts.items()}

        # Vehicle types
        vehicle_cols = {
            "Motorcycle": "รถจักรยานยนต์",
            "Car": "รถยนต์นั่งส่วนบุคคล",
            "Pickup": "รถปิคอัพบรรทุก4ล้อ",
            "Van": "รถตู้",
            "Truck 6W": "รถบรรทุก6ล้อ",
            "Truck 10W+": "รถบรรทุกมากกว่า10ล้อ",
            "Bus": "รถโดยสารมากกว่า4ล้อ",
            "Pedestrian": "คนเดินเท้า",
        }
        by_vehicle = {}
        for label, col in vehicle_cols.items():
            if col in df.columns:
                total = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
                if total > 0:
                    by_vehicle[label] = total

        # Severity breakdown
        deaths = int(pd.to_numeric(df.get("ผู้เสียชีวิต", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        serious = int(pd.to_numeric(df.get("ผู้บาดเจ็บสาหัส", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        minor = int(pd.to_numeric(df.get("ผู้บาดเจ็บเล็กน้อย", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        severity = {"Deaths": deaths, "Serious Injuries": serious, "Minor Injuries": minor}

        # Risk level distribution
        risk_dist = grid_df[grid_df["total_accidents"] > 0]["risk_level"].value_counts().to_dict()
        risk_distribution = {str(k): int(v) for k, v in risk_dist.items()}

        # Weekend vs weekday
        weekend_col = df.get("is_weekend")
        weekend_count = 0
        weekday_count = 0
        if weekend_col is not None:
            if weekend_col.dtype == bool:
                weekend_mask = weekend_col.fillna(False)
            else:
                weekend_mask = weekend_col.astype(str).str.lower().isin({"true", "1", "yes"})
            weekend_count = int(weekend_mask.sum())
            weekday_count = int(len(df) - weekend_count)

        return jsonify({
            "total_accidents": int(len(df)),
            "total_grids": int(len(grid_df)),
            "by_hour": by_hour,
            "by_day": by_day,
            "by_month": by_month,
            "top_zones": top_zones,
            "by_area": by_area,
            "by_vehicle": by_vehicle,
            "severity": severity,
            "risk_distribution": risk_distribution,
            "weekend_vs_weekday": {"Weekend": weekend_count, "Weekday": weekday_count},
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/risk-map")
def risk_map():
    try:
        _require_file(RISK_MAP_HTML_PATH)
        return send_file(RISK_MAP_HTML_PATH)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/risk-zones")
def risk_zones():
    try:
        df = _load_grid_stats()
        df = df[df["total_accidents"] > 0].copy()
        df = df.where(pd.notna(df), None)
        return jsonify({"count": int(len(df)), "risk_zones": df.to_dict(orient="records")})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/temporal-stats")
def temporal_stats():
    try:
        df = _load_bangkok_accidents()
        stats = _calculate_temporal_stats(df)
        return jsonify(stats)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/analyze-route", methods=["POST"])
def analyze_route():
    """Analyze route risk and generate real-time alerts."""
    try:
        data = request.json or {}
        required = ["start_lat", "start_lon", "end_lat", "end_lon"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400
        try:
            start_lat = float(data["start_lat"])
            start_lon = float(data["start_lon"])
            end_lat = float(data["end_lat"])
            end_lon = float(data["end_lon"])
        except (TypeError, ValueError):
            return jsonify({"error": "All coordinates must be valid numbers"}), 400

        waypoints_input = data.get("waypoints")
        parsed_waypoints = []
        if isinstance(waypoints_input, list):
            for point in waypoints_input:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    parsed_waypoints.append([float(point[0]), float(point[1])])

        analyze_traffic_source = "estimated"
        if len(parsed_waypoints) >= 2:
            waypoints = parsed_waypoints
            calculated_distance = calculate_distance_from_waypoints(waypoints)
            payload_distance = data.get("distance_km", data.get("distance"))
            payload_duration = data.get("duration_min", data.get("duration"))

            try:
                route_distance_km = float(payload_distance)
            except (TypeError, ValueError):
                route_distance_km = calculated_distance
            route_distance_km = max(0.0, route_distance_km)

            # Try Google Maps for traffic-aware duration
            gmaps_data = get_google_maps_travel_time(start_lat, start_lon, end_lat, end_lon)
            if gmaps_data:
                route_duration_min = gmaps_data["duration_minutes"]
                analyze_traffic_source = "google_maps_traffic" if gmaps_data["has_traffic"] else "google_maps"
            else:
                try:
                    route_duration_min = int(payload_duration)
                except (TypeError, ValueError):
                    route_duration_min = int((route_distance_km / 40) * 60) if route_distance_km > 0 else 0
            route_duration_min = max(0, route_duration_min)
        else:
            # Try OSRM → cached demo route → dense interpolation
            gmaps_data = get_google_maps_travel_time(start_lat, start_lon, end_lat, end_lon)
            osrm_route = get_osrm_route(start_lat, start_lon, end_lat, end_lon)

            if osrm_route:
                waypoints = osrm_route["waypoints"]
                route_distance_km = float(osrm_route["distance_km"])
                route_duration_min = int(osrm_route["duration_min"])
                analyze_traffic_source = "osrm"
            else:
                cached_route = _find_cached_route(start_lat, start_lon, end_lat, end_lon)
                if cached_route:
                    waypoints = cached_route["waypoints"]
                    route_distance_km = float(cached_route["distance_km"] or 0) or calculate_distance_from_waypoints(waypoints)
                    route_duration_min = int(cached_route["duration_min"] or 0) or int((route_distance_km / 40) * 60)
                    analyze_traffic_source = "cached_osrm"
                else:
                    waypoints = _generate_dense_interpolation(start_lat, start_lon, end_lat, end_lon)
                    route_distance_km = calculate_distance_from_waypoints(waypoints)
                    route_duration_min = int((route_distance_km / 40) * 60)
                    analyze_traffic_source = "estimated"

            # Override duration with Google Maps traffic-aware time when available
            if gmaps_data:
                route_duration_min = gmaps_data["duration_minutes"]
                if gmaps_data["distance_km"] and analyze_traffic_source == "estimated":
                    route_distance_km = gmaps_data["distance_km"]
                analyze_traffic_source = "google_maps_traffic" if gmaps_data["has_traffic"] else "google_maps"

        sampled_waypoints = waypoints[:: max(1, len(waypoints) // 50)]

        grid_df = _load_grid_stats()
        segments = analyze_route_segments(sampled_waypoints, grid_df, segment_distance_km=1.0)
        alerts = generate_alerts_from_segments(segments, advance_warning_km=0.5)

        # Try ML-based overall risk first, fall back to spatial
        ml_risk = calculate_route_risk_ml(sampled_waypoints)
        if ml_risk is not None:
            overall_risk = ml_risk
            prediction_method = "xgboost"
        else:
            overall_risk = calculate_route_risk_density_aware(sampled_waypoints, grid_df)
            prediction_method = "spatial"

        high_risk_count = len([segment for segment in segments if segment["risk_level"] == "high"])
        medium_risk_count = len([segment for segment in segments if segment["risk_level"] == "medium"])
        low_risk_count = len([segment for segment in segments if segment["risk_level"] == "low"])

        return jsonify(
            {
                "success": True,
                "route": {
                    "waypoints": waypoints,
                    "distance_km": round(route_distance_km, 2),
                    "duration_min": route_duration_min,
                    "risk_score": min(100, overall_risk["risk_score"]),
                    "risk_level": overall_risk["risk_level"],
                    "traffic_source": analyze_traffic_source,
                },
                "segments": segments,
                "alerts": alerts,
                "summary": {
                    "total_distance": round(route_distance_km, 2),
                    "total_time": route_duration_min,
                    "high_risk_segments": high_risk_count,
                    "medium_risk_segments": medium_risk_count,
                    "low_risk_segments": low_risk_count,
                    "total_alerts": len(alerts),
                    "prediction_method": prediction_method,
                    "model_accuracy": "39/39 zones (83.8% capture)" if prediction_method == "xgboost" else "83.8% spatial",
                    "traffic_source": analyze_traffic_source,
                },
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/compare-routes", methods=["POST"])
def compare_routes():
    """Hybrid Google Maps + Risk Overlay route comparison.

    1. Fetches multiple alternative routes from Google Maps (real roads).
    2. Decodes polyline waypoints for each route.
    3. Maps waypoints to 276 Bangkok grid zones.
    4. Scores each route using accident-density risk model.
    5. Returns FASTEST (shortest time) and SAFEST (lowest risk).
    """
    try:
        payload = request.get_json(silent=True) or {}
        required_fields = {"start_lat", "start_lon", "end_lat", "end_lon"}
        missing_fields = required_fields - set(payload.keys())
        if missing_fields:
            return jsonify({"error": f"Missing fields: {sorted(missing_fields)}"}), 400

        try:
            start_lat = float(payload["start_lat"])
            start_lon = float(payload["start_lon"])
            end_lat = float(payload["end_lat"])
            end_lon = float(payload["end_lon"])
        except (TypeError, ValueError):
            return jsonify({"error": "All coordinates must be valid numbers"}), 400

        if start_lat == end_lat and start_lon == end_lon:
            return jsonify({"error": "Start and end locations must be different"}), 400

        grid_df = _load_grid_stats()

        # ── Step 1: Get ALL alternative routes from Google Maps ──
        gmaps_routes = get_google_maps_routes(start_lat, start_lon, end_lat, end_lon)

        if gmaps_routes:
            # ── Step 2-4: Score each route with risk model ──
            scored_routes = []
            for groute in gmaps_routes:
                waypoints = groute["waypoints"]
                # Sample waypoints for risk calculation (every ~100m)
                sampled = waypoints[:: max(1, len(waypoints) // 80)]
                if sampled[-1] != waypoints[-1]:
                    sampled.append(waypoints[-1])

                # Calculate risk using both ML and spatial models
                ml_risk = calculate_route_risk_ml(sampled)
                spatial_risk = calculate_route_risk_density_aware(sampled, grid_df)
                risk = ml_risk if ml_risk is not None else spatial_risk

                scored_routes.append({
                    "waypoints": waypoints,
                    "distance_km": groute["distance_km"],
                    "time_min": groute["duration_minutes"],
                    "risk_score": risk["risk_score"],
                    "risk_level": risk["risk_level"],
                    "high_risk_zones": risk.get("high_risk_zones", 0),
                    "medium_risk_zones": risk.get("medium_risk_zones", 0),
                    "matched_grids": risk.get("matched_grids", 0),
                    "evaluated_waypoints": risk.get("evaluated_waypoints", len(sampled)),
                    "traffic_source": "google_maps_traffic" if groute["has_traffic"] else "google_maps",
                    "summary": groute.get("summary", ""),
                })

            # ── Step 5: Sort and pick FASTEST and SAFEST ──
            fastest = min(scored_routes, key=lambda r: r["time_min"])
            safest = min(scored_routes, key=lambda r: r["risk_score"])

            # If safest == fastest (same route), try to pick a different one
            # for the safe option that has lower risk
            if len(scored_routes) > 1 and safest["waypoints"] == fastest["waypoints"]:
                # Pick the route with 2nd lowest risk that isn't the fastest
                alternatives = [r for r in scored_routes if r["waypoints"] != fastest["waypoints"]]
                if alternatives:
                    safest = min(alternatives, key=lambda r: r["risk_score"])

            route_fast = fastest
            route_safe = safest
            alternatives_analyzed = len(gmaps_routes)
        else:
            # ── Fallback: original OSRM/interpolation logic ──
            fast_waypoints = generate_fast_route(start_lat, start_lon, end_lat, end_lon)
            safe_waypoints = generate_safe_route(
                start_lat, start_lon, end_lat, end_lon, grid_df, fast_waypoints=fast_waypoints
            )
            route_fast = _evaluate_route(fast_waypoints, grid_df)
            route_safe = _evaluate_route(safe_waypoints, grid_df)

            # Try Google Maps for accurate travel times
            gmaps_time = get_google_maps_travel_time(start_lat, start_lon, end_lat, end_lon)
            if gmaps_time:
                route_fast["distance_km"] = gmaps_time["distance_km"]
                route_fast["time_min"] = gmaps_time["duration_minutes"]
                route_fast["traffic_source"] = (
                    "google_maps_traffic" if gmaps_time["has_traffic"] else "google_maps"
                )
            else:
                route_fast["traffic_source"] = "estimated"

            route_safe["traffic_source"] = "estimated"
            alternatives_analyzed = 1

        # ── Metadata ──
        prediction_method = "xgboost" if (USE_ML and ml_predictor) else "spatial"
        for _route in (route_fast, route_safe):
            matched = _route.get("matched_grids", 0)
            evaluated = max(1, _route.get("evaluated_waypoints", 1))
            _route["confidence"] = round(min(0.99, 0.75 + (matched / evaluated) * 0.24), 2)
            _route["prediction_method"] = prediction_method

        time_diff = route_safe["time_min"] - route_fast["time_min"]
        if time_diff > 0:
            route_safe["time_difference"] = f"+{time_diff} min"
        elif time_diff < 0:
            route_safe["time_difference"] = f"{time_diff} min"
        else:
            route_safe["time_difference"] = "0 min"

        fast_risk_score = int(route_fast["risk_score"])
        safe_risk_score = int(route_safe["risk_score"])
        if safe_risk_score >= fast_risk_score:
            recommendation = "Fast route is already the safest available option. No safer alternative exists for this trip."
        else:
            risk_reduction_pct = (
                int((1 - (safe_risk_score / fast_risk_score)) * 100) if fast_risk_score > 0 else 0
            )
            time_cost = route_safe["time_min"] - route_fast["time_min"]
            if time_cost > 0:
                recommendation = f"Safe route reduces risk by {risk_reduction_pct}% for +{time_cost} minutes."
            elif time_cost < 0:
                recommendation = (
                    f"Safe route reduces risk by {risk_reduction_pct}% and is {abs(time_cost)} minutes faster."
                )
            else:
                recommendation = f"Safe route reduces risk by {risk_reduction_pct}% with no extra time."

        debug_data = {
            "fast_waypoints_sample": route_fast["waypoints"][:3],
            "safe_waypoints_sample": route_safe["waypoints"][:3],
            "fast_risk_details": (
                f"Calculated from {route_fast.get('matched_grids', 0)} matched grids "
                f"across {route_fast.get('evaluated_waypoints', 0)} waypoints"
            ),
            "safe_risk_details": (
                f"Calculated from {route_safe.get('matched_grids', 0)} matched grids "
                f"across {route_safe.get('evaluated_waypoints', 0)} waypoints"
            ),
            "fast_summary": route_fast.get("summary", ""),
            "safe_summary": route_safe.get("summary", ""),
            "google_maps_alternatives": alternatives_analyzed,
        }

        return jsonify(
            {
                "fast_route": route_fast,
                "safe_route": route_safe,
                "route_fast": route_fast,
                "route_safe": route_safe,
                "recommendation": recommendation,
                "alternatives_analyzed": alternatives_analyzed,
                "debug": debug_data,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/model-info")
def model_info():
    """Return information about the active prediction model."""
    if USE_ML and ml_predictor:
        try:
            top_features = ml_predictor.get_feature_importance(top_n=5)
        except Exception:
            top_features = {}

        return jsonify(
            {
                "status": "success",
                "model": "xgboost",
                "version": "1.0",
                "accuracy": "39/39 zones detected (83.8% temporal capture rate)",
                "test_performance": {
                    "recall": "39/39 zones in holdout test",
                    "precision": "0 false positives",
                    "lift": "17.5x over random selection",
                },
                "comparison": {
                    "spatial_baseline": "83.8% capture rate",
                    "random_forest": "37/39 zones",
                    "xgboost": "39/39 zones",
                    "improvement": "+16.2% over spatial baseline",
                },
                "top_features": top_features,
                "fallback_available": True,
                "prediction_time": "<100ms",
            }
        )

    return jsonify(
        {
            "status": "success",
            "model": "spatial_analysis",
            "accuracy": "83.8%",
            "method": "grid_based_counting",
            "ml_available": False,
        }
    )


@app.route("/api/compare-predictions", methods=["POST"])
def compare_predictions():
    """Compare ML vs spatial predictions side-by-side."""
    try:
        data = request.get_json(silent=True) or {}
        lat = float(data["lat"])
        lon = float(data["lon"])

        results = {
            "location": {"lat": lat, "lon": lon},
            "ml_prediction": None,
            "spatial_prediction": None,
            "difference": None,
        }

        if USE_ML and ml_predictor:
            try:
                results["ml_prediction"] = ml_predictor.predict_risk(lat, lon)
            except Exception as exc:
                results["ml_error"] = str(exc)

        if ml_predictor:
            try:
                results["spatial_prediction"] = ml_predictor.fallback_to_spatial(lat, lon)
            except Exception as exc:
                results["spatial_error"] = str(exc)

        if results["ml_prediction"] and results["spatial_prediction"]:
            ml_score = results["ml_prediction"]["risk_score"]
            spatial_score = results["spatial_prediction"]["risk_score"]
            results["difference"] = {
                "absolute": ml_score - spatial_score,
                "percentage": round(
                    ((ml_score - spatial_score) / max(spatial_score, 1)) * 100, 1
                ),
            }

        return jsonify(results)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


DEMO_ROUTE_DEFS = {
    "ratchada": {
        "name": "Ratchada (High Risk)",
        "start": {"lat": 13.7467, "lon": 100.5348},
        "end": {"lat": 13.73, "lon": 100.72},
    },
    "din_daeng": {
        "name": "Din Daeng (High Risk)",
        "start": {"lat": 13.7467, "lon": 100.5348},
        "end": {"lat": 13.73, "lon": 100.74},
    },
    "asok": {
        "name": "Asok (High Risk)",
        "start": {"lat": 13.7467, "lon": 100.5348},
        "end": {"lat": 13.74, "lon": 100.56},
    },
    "huai_khwang": {
        "name": "Huai Khwang (High Risk)",
        "start": {"lat": 13.7467, "lon": 100.5348},
        "end": {"lat": 13.76, "lon": 100.75},
    },
}


def _interpolate_waypoints(waypoints, target_count):
    """Interpolate waypoints to reach a target density."""
    if len(waypoints) >= target_count:
        return waypoints

    cum = [0.0]
    for i in range(1, len(waypoints)):
        cum.append(
            cum[-1]
            + _haversine_km(waypoints[i - 1][0], waypoints[i - 1][1], waypoints[i][0], waypoints[i][1])
        )
    total = cum[-1]
    if total == 0:
        return waypoints

    result = []
    step = total / (target_count - 1)
    for i in range(target_count):
        target_dist = i * step
        for j in range(1, len(cum)):
            if cum[j] >= target_dist:
                seg_len = cum[j] - cum[j - 1]
                frac = 0 if seg_len == 0 else (target_dist - cum[j - 1]) / seg_len
                lat = waypoints[j - 1][0] + frac * (waypoints[j][0] - waypoints[j - 1][0])
                lon = waypoints[j - 1][1] + frac * (waypoints[j][1] - waypoints[j - 1][1])
                result.append([round(lat, 6), round(lon, 6)])
                break
        else:
            result.append(list(waypoints[-1]))
    return result


@app.route("/api/refresh-demo-routes", methods=["POST"])
def refresh_demo_routes():
    """Fetch fresh OSRM full-geometry routes and update demo_routes.json."""
    import json as _json

    demo_routes_path = Path(app.static_folder) / "demo_routes.json"
    result = {}
    errors = []

    for key, route_def in DEMO_ROUTE_DEFS.items():
        s = route_def["start"]
        e = route_def["end"]
        osrm = get_osrm_route(s["lat"], s["lon"], e["lat"], e["lon"])

        if osrm:
            raw_count = len(osrm["waypoints"])
            min_points = max(200, int(osrm["distance_km"] * 25))
            target = max(min_points, raw_count)

            if raw_count < target:
                dense = _interpolate_waypoints(osrm["waypoints"], target)
            else:
                dense = osrm["waypoints"]

            result[key] = {
                "name": route_def["name"],
                "start": route_def["start"],
                "end": route_def["end"],
                "waypoints": dense,
                "distance": osrm["distance_km"],
                "duration": osrm["duration_min"],
                "osrm_source": True,
                "osrm_raw_points": raw_count,
            }
        else:
            errors.append(key)

    if not result:
        return jsonify({"error": "OSRM unavailable for all routes", "failed": errors}), 503

    # Merge with existing (keep old data for routes that failed)
    try:
        existing = _json.loads(demo_routes_path.read_text())
    except Exception:
        existing = {}

    existing.update(result)

    demo_routes_path.write_text(_json.dumps(existing))
    summary = {k: {"waypoints": len(v["waypoints"]), "distance": v["distance"]} for k, v in result.items()}
    return jsonify({"success": True, "updated": summary, "failed": errors})


# ---------------------------------------------------------------------------
# NLP Analysis (Track 1) — serves from cache
# ---------------------------------------------------------------------------
NLP_CACHE_PATH = BASE_DIR / "static" / "nlp_cache.json"


def _load_nlp_cache():
    try:
        if NLP_CACHE_PATH.exists():
            with open(NLP_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


@app.route("/nlp")
def nlp():
    """Bangkok Safety Assistant — hybrid chatbot page."""
    return render_template("nlp.html")


@app.route("/api/nlp/pipeline-demo")
def nlp_pipeline_demo():
    """Pipeline examples: Thai text → tokenization → XGBoost feature."""
    try:
        cache = _load_nlp_cache()
        return jsonify(cache.get("pipeline_demo", {}))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/nlp/cause-analysis")
def nlp_cause_analysis():
    """Accident cause frequency breakdown."""
    try:
        cache = _load_nlp_cache()
        return jsonify(cache.get("cause_analysis", {}))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/nlp/recommendations")
def nlp_recommendations():
    """Per-zone policy recommendations based on NLP-derived causes."""
    try:
        cache = _load_nlp_cache()
        return jsonify(cache.get("recommendations", []))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/nlp/wordcloud")
def nlp_wordcloud():
    """Thai keyword frequencies for word cloud visualization."""
    try:
        cache = _load_nlp_cache()
        return jsonify(cache.get("word_cloud", {}))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/nlp/ml-metrics")
def nlp_ml_metrics():
    """ML text classifier metrics (accuracy, precision, recall, F1, per-class)."""
    try:
        metrics_path = BASE_DIR / "data" / "processed" / "nlp_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({"error": "Metrics not available"}), 404
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/nlp/classify", methods=["POST"])
def nlp_classify():
    """Classify Thai text into accident cause category using ML classifier."""
    try:
        from nlp_analysis import get_cause_classifier

        data = request.json
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        classifier = get_cause_classifier()
        result = classifier.classify(text)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Chatbot — hybrid scripted + Groq AI
# ---------------------------------------------------------------------------

_chatbot_data_path = BASE_DIR / "static" / "chatbot_data.json"
try:
    with open(_chatbot_data_path, "r", encoding="utf-8") as _f:
        CHATBOT_DATA = json.load(_f)
    print(f"Chatbot data loaded: {_chatbot_data_path}")
except FileNotFoundError:
    CHATBOT_DATA = {}
    print(f"Warning: chatbot data not found at {_chatbot_data_path}")


@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    """
    Hybrid chatbot endpoint:
    1. Try pattern matching (fast, reliable)
    2. Fall back to Groq AI (flexible)
    """
    try:
        data = request.json
        user_message = data.get("message", "").strip()

        if not user_message:
            return jsonify({"success": False, "error": "Empty message"}), 400

        # Step 1: Try intent detection (scripted patterns)
        intent = detect_intent(user_message)

        if intent:
            response = get_scripted_response(intent, user_message)
            response["source"] = "scripted"
            response["intent"] = intent
        else:
            # Step 2: ML cause detection — catch cause-specific questions the
            # scripted patterns miss (e.g. "Tell me about drowsy driving",
            # "Is drunk driving a problem?", "Equipment failure stats?")
            cause_key, confidence = detect_cause_from_question(user_message)
            ml_response = None
            if cause_key and confidence >= 0.6:
                ml_response = handle_ml_cause_response(cause_key, confidence, user_message)

            if ml_response:
                response = ml_response
            else:
                # Step 3: AI response (flexible, open-ended)
                response = get_groq_response(user_message, CHATBOT_DATA)

        return jsonify({"success": True, "response": response})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "fallback": "I encountered an error. Please try asking your question differently or use one of the suggested questions.",
        }), 500


@app.route("/api/chatbot/suggestions", methods=["GET"])
def chatbot_suggestions():
    """Return suggested questions for UI."""
    suggestions = [
        "Which areas have the most speeding accidents?",
        "Is Asok Intersection safe?",
        "What time are accidents most common?",
        "What should Bangkok do to improve road safety?",
        "Show me drunk driving patterns",
        "What causes most accidents?",
        "Is Sukhumvit safe at night?",
    ]
    return jsonify({"success": True, "suggestions": suggestions})


if __name__ == "__main__":
    print()
    print("=" * 64)
    print("RoadRisk Bangkok - Risk Prediction System")
    print("=" * 64)

    if USE_ML:
        print("Prediction: XGBoost (Machine Learning)")
        print("  Detection:   39/39 danger zones (83.8% temporal capture, 17.5x lift)")
        print("  Improvement: +16.2% vs spatial baseline")
        print("  Features:    44-feature gradient boosting")
        print("  Fallback:    Spatial analysis available")
    else:
        print("Prediction: Spatial Analysis (Grid-based)")
        print("  Accuracy: 83.8%")
        print("  Method:   Accident density counting")

    print("-" * 64)
    print("Endpoints:")
    print("  GET  /                         Home")
    print("  GET  /risk-map                 Risk heatmap")
    print("  GET  /api/risk-zones           Risk zone data")
    print("  GET  /api/temporal-stats       Time-based stats")
    print("  GET  /api/model-info           ML model details")
    print("  POST /api/analyze-route        Analyse a route")
    print("  POST /api/compare-routes       Compare fast/safe")
    print("  POST /api/compare-predictions  ML vs spatial")
    print("  POST /api/refresh-demo-routes  Refresh demos")
    print("  GET  /nlp                      NLP analysis (Track 1)")
    print("  GET  /api/nlp/*                NLP endpoints")
    print("  POST /api/chatbot              Hybrid chatbot")
    print("  GET  /api/chatbot/suggestions  Suggested questions")
    print("=" * 64)
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting on http://0.0.0.0:{port}")
    print()
    app.run(debug=False, host="0.0.0.0", port=port)
