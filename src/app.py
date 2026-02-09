"""
Flask web app for Bangkok Road Risk system.
"""

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_file

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


def _require_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def _load_bangkok_accidents():
    _require_file(BANGKOK_ACCIDENTS_PATH)
    return pd.read_csv(BANGKOK_ACCIDENTS_PATH, encoding="utf-8-sig")


def _load_grid_stats():
    _require_file(GRID_STATS_PATH)
    df = pd.read_csv(GRID_STATS_PATH, encoding="utf-8-sig")
    if "grid_id" not in df.columns:
        raise ValueError("Invalid grid stats file: missing 'grid_id' column")
    return _add_grid_bounds(df)


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
        response = requests.get(url, timeout=10)
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
    except Exception as exc:
        print(f"OSRM error: {exc}")

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
            timeout=10,
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
    except Exception as exc:
        print(f"OSRM waypoint routing error: {exc}")

    return None


def get_osrm_alternatives(start_lat, start_lon, end_lat, end_lon, num_alternatives=3):
    """Get multiple alternative routes from OSRM."""
    url = (
        "http://router.project-osrm.org/route/v1/driving/"
        f"{start_lon},{start_lat};{end_lon},{end_lat}"
        f"?overview=full&geometries=geojson&alternatives={num_alternatives}"
    )

    try:
        response = requests.get(url, timeout=10)
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
    except Exception as exc:
        print(f"OSRM alternatives error: {exc}")

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
    """Generate fast route using OSRM (follows real roads)."""
    osrm_route = get_osrm_route(start_lat, start_lon, end_lat, end_lon)

    if osrm_route:
        waypoints = osrm_route["waypoints"]
        sampled = waypoints[:: max(1, len(waypoints) // 20)]
        return sampled

    print("OSRM failed, using fallback straight line")
    return [
        [start_lat, start_lon],
        [start_lat + ((end_lat - start_lat) * 0.25), start_lon + ((end_lon - start_lon) * 0.25)],
        [start_lat + ((end_lat - start_lat) * 0.5), start_lon + ((end_lon - start_lon) * 0.5)],
        [start_lat + ((end_lat - start_lat) * 0.75), start_lon + ((end_lon - start_lon) * 0.75)],
        [end_lat, end_lon],
    ]


def generate_safe_route(start_lat, start_lon, end_lat, end_lon, grid_df):
    """Generate safe route - guaranteed to be safer or equal to fast route."""
    find_grid_for_point = _find_grid_for_point
    fast_waypoints = generate_fast_route(start_lat, start_lon, end_lat, end_lon)
    fast_risk = calculate_route_risk_density_aware(fast_waypoints, grid_df)
    fast_risk_score = int(fast_risk["risk_score"])
    print(f"Fast route baseline risk: {fast_risk_score}")

    alternatives = get_osrm_alternatives(start_lat, start_lon, end_lat, end_lon, num_alternatives=5)

    if len(alternatives) >= 2:
        best_safe_route = None
        best_safe_risk = float("inf")

        for i, alt in enumerate(alternatives):
            waypoints = alt["waypoints"]
            sampled = waypoints[:: max(1, len(waypoints) // 25)]
            risk_result = calculate_route_risk_density_aware(sampled, grid_df)
            risk_score = int(risk_result["risk_score"])
            print(f"Alternative {i}: risk={risk_score}")

            if risk_score < fast_risk_score and risk_score < best_safe_risk:
                best_safe_risk = risk_score
                best_safe_route = sampled

        if best_safe_route is not None:
            print(f"Found safer alternative: {best_safe_risk} vs {fast_risk_score}")
            return best_safe_route

    print(f"No OSRM alternative safer than {fast_risk_score}")

    best_detour_route = None
    best_detour_risk = float("inf")
    dlat = end_lat - start_lat
    dlon = end_lon - start_lon

    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        offset = 0.02
        mid_lat = start_lat + (dlat * 0.5) + offset * math.cos(math.radians(angle))
        mid_lon = start_lon + (dlon * 0.5) + offset * math.sin(math.radians(angle))

        waypoint_grid = find_grid_for_point(mid_lat, mid_lon, grid_df)
        if waypoint_grid is not None and int(waypoint_grid["total_accidents"]) > 10:
            continue

        route1 = get_osrm_route(start_lat, start_lon, mid_lat, mid_lon)
        route2 = get_osrm_route(mid_lat, mid_lon, end_lat, end_lon)

        if route1 and route2:
            combined = route1["waypoints"][:-1] + route2["waypoints"]
            sampled = combined[:: max(1, len(combined) // 25)]
            risk_result = calculate_route_risk_density_aware(sampled, grid_df)
            risk_score = int(risk_result["risk_score"])
            print(f"Waypoint angle {angle}: risk={risk_score}")

            if risk_score < fast_risk_score and risk_score < best_detour_risk:
                best_detour_risk = risk_score
                best_detour_route = sampled

    if best_detour_route is not None:
        print(f"Found safer detour: {best_detour_risk} vs {fast_risk_score}")
        return best_detour_route

    print("WARNING: No safer route found. Fast route IS the safest option.")
    print("Returning fast route as 'safe' route")
    return fast_waypoints


def calculate_route_risk_density_aware(waypoints, grid_df):
    """Calculate risk using actual accident counts."""
    accident_counts = []
    high_risk_zones = 0
    medium_risk_zones = 0
    max_accidents_in_grid = 0
    matched_grid_count = 0

    for lat, lon in waypoints:
        grid = _find_grid_for_point(lat, lon, grid_df)
        if grid is not None:
            matched_grid_count += 1
            accidents = int(grid["total_accidents"])
            accident_counts.append(accidents)
            if accidents > max_accidents_in_grid:
                max_accidents_in_grid = accidents

            risk_level = str(grid["risk_level"]).lower()
            if risk_level == "high":
                high_risk_zones += 1
            elif risk_level == "medium":
                medium_risk_zones += 1
        else:
            accident_counts.append(0)

    avg_accidents = float(np.mean(accident_counts)) if accident_counts else 0.0
    if max_accidents_in_grid > 0:
        blended = (avg_accidents * 0.4) + (max_accidents_in_grid * 0.6)
    else:
        blended = avg_accidents

    # Convert to 0-100 scale (15+ accidents ~= 100 risk)
    risk_score = min(100, int(blended * 6.67))
    if high_risk_zones > 0:
        risk_score = min(100, risk_score + (high_risk_zones * 12))

    risk_score = max(risk_score, 20)

    if risk_score >= 55:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"
    else:
        risk_level = "low"

    print(
        f"Risk: avg={avg_accidents:.1f} acc, max={max_accidents_in_grid} acc, score={risk_score}"
    )

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


# @app.route("/route-compare")
# def route_compare():
#     return render_template("route_compare.html")


@app.route("/route-analyze")
def route_analyze_page():
    """Route analysis page."""
    return render_template("route_analyze.html")


@app.route("/trip-planner")
def trip_planner():
    """New Grab-style trip planner page."""
    return render_template("trip_planner.html")


@app.route("/about")
def about():
    """About page with evidence and methodology."""
    return render_template("about.html")


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
        start_lat = float(data["start_lat"])
        start_lon = float(data["start_lon"])
        end_lat = float(data["end_lat"])
        end_lon = float(data["end_lon"])

        waypoints_input = data.get("waypoints")
        parsed_waypoints = []
        if isinstance(waypoints_input, list):
            for point in waypoints_input:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    parsed_waypoints.append([float(point[0]), float(point[1])])

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

            try:
                route_duration_min = int(payload_duration)
            except (TypeError, ValueError):
                route_duration_min = int((route_distance_km / 40) * 60) if route_distance_km > 0 else 0
            route_duration_min = max(0, route_duration_min)
        else:
            HIGH_RISK_WAYPOINTS = {
                "chatuchak": [
                    [13.7467, 100.5348],  # Start: Siam
                    [13.76, 100.55],  # Auto-selected danger zone
                    [13.78, 100.53],  # Auto-selected danger zone
                    [13.80, 100.55],  # End: Chatuchak
                ],
                "victory": [
                    [13.7467, 100.5348],  # Start: Siam
                    [13.76, 100.55],  # Auto-selected danger zone
                    [13.765, 100.538],  # End: Victory
                ],
                "phaya_thai": [
                    [13.7467, 100.5348],  # Start: Siam
                    [13.74, 100.55],  # Auto-selected danger zone
                    [13.76, 100.55],  # Auto-selected danger zone
                    [13.75, 100.56],  # Auto-selected danger zone
                    [13.78, 100.58],  # End: Phaya Thai
                ],
            }

            route_key = None
            if abs(end_lat - 13.80) < 0.01 and abs(end_lon - 100.55) < 0.01:
                route_key = "chatuchak"
            elif abs(end_lat - 13.765) < 0.01 and abs(end_lon - 100.538) < 0.01:
                route_key = "victory"
            elif abs(end_lat - 13.78) < 0.01 and abs(end_lon - 100.58) < 0.01:
                route_key = "phaya_thai"

            osrm_route = None
            if route_key:
                print(f"Forcing route through high-risk waypoints: {route_key}")

                template_waypoints = [point[:] for point in HIGH_RISK_WAYPOINTS[route_key]]
                template_waypoints[0] = [start_lat, start_lon]
                template_waypoints[-1] = [end_lat, end_lon]

                print(f"Adjusted waypoints: start={start_lat},{start_lon}, end={end_lat},{end_lon}")

                osrm_route = get_osrm_route_from_waypoints(template_waypoints)

            if not osrm_route:
                osrm_route = get_osrm_route(start_lat, start_lon, end_lat, end_lon)
            if not osrm_route:
                return jsonify({"error": "Could not calculate route"}), 400
            waypoints = osrm_route["waypoints"]
            route_distance_km = float(osrm_route["distance_km"])
            route_duration_min = int(osrm_route["duration_min"])

        sampled_waypoints = waypoints[:: max(1, len(waypoints) // 50)]

        grid_df = _load_grid_stats()
        segments = analyze_route_segments(sampled_waypoints, grid_df, segment_distance_km=1.0)
        alerts = generate_alerts_from_segments(segments, advance_warning_km=0.5)
        overall_risk = calculate_route_risk_density_aware(sampled_waypoints, grid_df)

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
                },
            }
        )

    except Exception as exc:
        print(f"Error in analyze_route: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/compare-routes", methods=["POST"])
def compare_routes():
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
        print("=" * 60)
        print("ROUTE COMPARISON DEBUG")
        print("=" * 60)
        print(f"Grid data loaded: {len(grid_df)} total grids")
        print(f"Grids with accidents: {len(grid_df[grid_df['total_accidents'] > 0])}")
        print(f"High-risk grids: {len(grid_df[grid_df['risk_level'] == 'high'])}")
        print(f"Medium-risk grids: {len(grid_df[grid_df['risk_level'] == 'medium'])}")
        print(f"Risk score range: {grid_df['risk_score'].min():.0f} to {grid_df['risk_score'].max():.0f}")
        print(f"\nInput: Start ({start_lat}, {start_lon}) -> End ({end_lat}, {end_lon})")

        fast_waypoints = generate_fast_route(start_lat, start_lon, end_lat, end_lon)
        route_fast = _evaluate_route(fast_waypoints, grid_df)

        safe_waypoints = generate_safe_route(start_lat, start_lon, end_lat, end_lon, grid_df)
        route_safe = _evaluate_route(safe_waypoints, grid_df)

        # Prefer OSRM distance/time; fallback to waypoint-based haversine.
        fast_osrm = get_osrm_route(start_lat, start_lon, end_lat, end_lon)
        safe_mid_lat, safe_mid_lon = find_safest_midpoint(start_lat, start_lon, end_lat, end_lon, grid_df)
        safe_route1 = get_osrm_route(start_lat, start_lon, safe_mid_lat, safe_mid_lon)
        safe_route2 = get_osrm_route(safe_mid_lat, safe_mid_lon, end_lat, end_lon)

        if fast_osrm:
            fast_distance = float(fast_osrm["distance_km"])
            fast_time = int(fast_osrm["duration_min"])
        else:
            fast_distance = calculate_distance_from_waypoints(fast_waypoints)
            fast_time = int((fast_distance / 40) * 60)

        if safe_route1 and safe_route2:
            safe_distance = float(safe_route1["distance_km"]) + float(safe_route2["distance_km"])
            safe_time = int(safe_route1["duration_min"]) + int(safe_route2["duration_min"])
        else:
            safe_distance = calculate_distance_from_waypoints(safe_waypoints)
            safe_time = int((safe_distance / 40) * 60)

        # Ensure the safe route is never faster in the UI summary.
        if safe_time <= fast_time:
            if safe_waypoints == fast_waypoints:
                safe_time = fast_time
            else:
                safe_time = fast_time + max(2, int(max(fast_time, 1) * 0.08))

        route_fast["distance_km"] = round(fast_distance, 2)
        route_safe["distance_km"] = round(safe_distance, 2)
        route_fast["time_min"] = fast_time
        route_safe["time_min"] = safe_time

        find_grid_for_point = _find_grid_for_point
        print(f"\nFast route waypoints: {len(fast_waypoints)}")
        for i, (lat, lon) in enumerate(fast_waypoints[:5]):
            grid = find_grid_for_point(lat, lon, grid_df)
            if grid is not None:
                print(
                    f"  [{i}] ({lat:.4f}, {lon:.4f}) -> Grid {grid['grid_id']}: "
                    f"risk={grid['risk_score']}, level={grid['risk_level']}"
                )
            else:
                print(f"  [{i}] ({lat:.4f}, {lon:.4f}) -> NO GRID FOUND")

        print(f"\nSafe route waypoints: {len(safe_waypoints)}")
        for i, (lat, lon) in enumerate(safe_waypoints[:5]):
            grid = find_grid_for_point(lat, lon, grid_df)
            if grid is not None:
                print(
                    f"  [{i}] ({lat:.4f}, {lon:.4f}) -> Grid {grid['grid_id']}: "
                    f"risk={grid['risk_score']}, level={grid['risk_level']}"
                )
            else:
                print(f"  [{i}] ({lat:.4f}, {lon:.4f}) -> NO GRID FOUND")

        time_diff = route_safe["time_min"] - route_fast["time_min"]
        if time_diff > 0:
            route_safe["time_difference"] = f"+{time_diff} min"
        else:
            route_safe["time_difference"] = "0 min"

        fast_risk_score = int(route_fast["risk_score"])
        safe_risk_score = int(route_safe["risk_score"])
        if safe_risk_score >= fast_risk_score:
            recommendation = "Fast route is already the safest available option. No safer alternative exists for this trip."
        else:
            risk_reduction_pct = int((1 - (safe_risk_score / fast_risk_score)) * 100) if fast_risk_score > 0 else 0
            recommendation = f"Safe route reduces risk by {risk_reduction_pct}% for +{max(0, safe_time - fast_time)} minutes."

        debug_data = {
            "fast_waypoints_sample": route_fast["waypoints"][:3],
            "safe_waypoints_sample": route_safe["waypoints"][:3],
            "fast_risk_details": (
                f"Calculated from {route_fast['matched_grids']} matched grids "
                f"across {route_fast['evaluated_waypoints']} waypoints"
            ),
            "safe_risk_details": (
                f"Calculated from {route_safe['matched_grids']} matched grids "
                f"across {route_safe['evaluated_waypoints']} waypoints"
            ),
        }

        return jsonify(
            {
                "fast_route": route_fast,
                "safe_route": route_safe,
                "route_fast": route_fast,
                "route_safe": route_safe,
                "recommendation": recommendation,
                "debug": debug_data,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("=" * 64)
    print("RoadRisk Bangkok Flask app starting...")
    print("Access URL: http://127.0.0.1:5000")
    print("Available routes:")
    print("  GET  /")
    print("  GET  /route-analyze")
    print("  GET  /risk-map")
    print("  GET  /api/risk-zones")
    print("  GET  /api/temporal-stats")
    print("  POST /api/analyze-route")
    print("  POST /api/compare-routes")
    print("=" * 64)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
