"""
Fetch full-geometry OSRM routes for each demo route and write dense
waypoints (500+ per long route) into static/demo_routes.json.

Usage:
    python scripts/generate_dense_routes.py

If OSRM is unavailable, generates dense interpolated waypoints as fallback.
Re-run when OSRM is accessible to get real road-following geometry.
"""

import json
import math
import time
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "static" / "demo_routes.json"

OSRM_BASE = "http://router.project-osrm.org/route/v1/driving"

DEMO_ROUTES = {
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


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(d_lon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def interpolate_to_density(waypoints, target_count):
    """Interpolate waypoints to reach a target count, spacing evenly."""
    if len(waypoints) >= target_count:
        return waypoints

    cum = [0.0]
    for i in range(1, len(waypoints)):
        cum.append(
            cum[-1]
            + haversine_km(
                waypoints[i - 1][0],
                waypoints[i - 1][1],
                waypoints[i][0],
                waypoints[i][1],
            )
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
                seg_start = cum[j - 1]
                seg_len = cum[j] - cum[j - 1]
                if seg_len == 0:
                    frac = 0
                else:
                    frac = (target_dist - seg_start) / seg_len
                lat = waypoints[j - 1][0] + frac * (waypoints[j][0] - waypoints[j - 1][0])
                lon = waypoints[j - 1][1] + frac * (waypoints[j][1] - waypoints[j - 1][1])
                result.append([round(lat, 6), round(lon, 6)])
                break
        else:
            result.append(list(waypoints[-1]))

    return result


def generate_fallback_dense(start_lat, start_lon, end_lat, end_lon, target_count):
    """Generate dense linearly-interpolated waypoints as OSRM fallback."""
    waypoints = [[start_lat, start_lon]]
    for i in range(1, target_count - 1):
        frac = i / (target_count - 1)
        lat = round(start_lat + frac * (end_lat - start_lat), 6)
        lon = round(start_lon + frac * (end_lon - start_lon), 6)
        waypoints.append([lat, lon])
    waypoints.append([end_lat, end_lon])
    return waypoints


def fetch_osrm_full(start_lat, start_lon, end_lat, end_lon):
    """Fetch OSRM route with FULL geometry (all coordinates)."""
    bases = [
        "https://router.project-osrm.org/route/v1/driving",
        "http://router.project-osrm.org/route/v1/driving",
    ]
    for base in bases:
        url = (
            f"{base}/{start_lon},{start_lat};{end_lon},{end_lat}"
            "?overview=full&geometries=geojson&steps=true"
        )
        print(f"  GET {url}")
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}, trying next...")
                continue
            data = resp.json()
        except Exception as exc:
            print(f"  Request failed: {exc}, trying next...")
            continue

        if data.get("code") != "Ok" or not data.get("routes"):
            print(f"  OSRM error: {data.get('code', 'unknown')}")
            continue

        route = data["routes"][0]
        coords = route["geometry"]["coordinates"]
        waypoints = [[round(c[1], 6), round(c[0], 6)] for c in coords]
        distance_km = route["distance"] / 1000
        duration_min = int(route["duration"] / 60)

        return {
            "waypoints": waypoints,
            "distance_km": round(distance_km, 2),
            "duration_min": duration_min,
        }

    return None


def main():
    result = {}

    for key, route_def in DEMO_ROUTES.items():
        s = route_def["start"]
        e = route_def["end"]
        print(f"\n{'='*50}")
        print(f"Route: {key} ({route_def['name']})")

        osrm = fetch_osrm_full(s["lat"], s["lon"], e["lat"], e["lon"])

        if osrm:
            raw_count = len(osrm["waypoints"])
            print(f"  OSRM: {raw_count} pts, {osrm['distance_km']} km, {osrm['duration_min']} min")

            min_points = max(200, int(osrm["distance_km"] * 25))
            target = max(min_points, raw_count)

            if raw_count < target:
                dense = interpolate_to_density(osrm["waypoints"], target)
                print(f"  Interpolated to {len(dense)} waypoints")
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
            print("  OSRM unavailable - generating dense interpolated fallback")
            straight_km = haversine_km(s["lat"], s["lon"], e["lat"], e["lon"])
            est_road_km = round(straight_km * 1.3, 2)
            est_duration = max(1, int((est_road_km / 40) * 60))
            target = max(200, int(est_road_km * 25))

            dense = generate_fallback_dense(s["lat"], s["lon"], e["lat"], e["lon"], target)
            print(f"  Generated {len(dense)} interpolated waypoints (~{est_road_km} km)")

            result[key] = {
                "name": route_def["name"],
                "start": route_def["start"],
                "end": route_def["end"],
                "waypoints": dense,
                "distance": est_road_km,
                "duration": est_duration,
                "osrm_source": False,
            }

        time.sleep(0.5)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f)

    print(f"\n{'='*50}")
    print(f"Written to {OUTPUT_PATH}")
    for key, data in result.items():
        src = "OSRM" if data.get("osrm_source") else "interpolated"
        print(f"  {key}: {len(data['waypoints'])} waypoints, {data['distance']} km ({src})")
    print("Done!")


if __name__ == "__main__":
    main()
