"""Test natural risk levels for each quick destination."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import math
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
GRID_STATS_PATH = DATA_DIR / "bangkok_grid_stats.csv"

BKK_LAT_MIN = 13.6
BKK_LAT_MAX = 13.95
BKK_LON_MIN = 100.35
BKK_LON_MAX = 100.9
CELL_SIZE_DEG = 0.009


def _haversine_km(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def _load_grid_stats():
    df = pd.read_csv(GRID_STATS_PATH)
    return df


def _find_grid_for_point(lat, lon, grid_df):
    if lat < BKK_LAT_MIN or lat > BKK_LAT_MAX or lon < BKK_LON_MIN or lon > BKK_LON_MAX:
        return None
    grid_lat = round(round(lat / CELL_SIZE_DEG) * CELL_SIZE_DEG, 6)
    grid_lon = round(round(lon / CELL_SIZE_DEG) * CELL_SIZE_DEG, 6)
    matches = grid_df[
        (abs(grid_df["center_lat"] - grid_lat) < CELL_SIZE_DEG / 2)
        & (abs(grid_df["center_lon"] - grid_lon) < CELL_SIZE_DEG / 2)
    ]
    if not matches.empty:
        return matches.iloc[0]
    return None


def generate_dense_waypoints(start_lat, start_lon, end_lat, end_lon, num_points=30):
    """Generate dense waypoints along a straight line (OSRM fallback equivalent)."""
    waypoints = []
    for i in range(num_points + 1):
        t = i / num_points
        lat = start_lat + (end_lat - start_lat) * t
        lon = start_lon + (end_lon - start_lon) * t
        waypoints.append([lat, lon])
    return waypoints


def calculate_route_risk(waypoints, grid_df):
    accident_counts = []
    high_risk_zones = 0
    medium_risk_zones = 0
    low_risk_zones = 0
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
                low_risk_zones += 1
        else:
            accident_counts.append(0)
            low_risk_zones += 1

    avg_accidents = float(np.mean(accident_counts)) if accident_counts else 0.0
    if max_accidents_in_grid > 0:
        blended = (avg_accidents * 0.4) + (max_accidents_in_grid * 0.6)
    else:
        blended = avg_accidents

    risk_score = min(100, int(blended * 6.67))
    if high_risk_zones > 0:
        risk_score = min(100, risk_score + (high_risk_zones * 12))

    if risk_score >= 55:
        risk_level = "HIGH"
    elif risk_score >= 30:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "high_risk_zones": high_risk_zones,
        "medium_risk_zones": medium_risk_zones,
        "low_risk_zones": low_risk_zones,
        "matched_grids": matched_grid_count,
    }


def calculate_distance(waypoints):
    total = 0.0
    for i in range(1, len(waypoints)):
        total += _haversine_km(
            waypoints[i-1][0], waypoints[i-1][1],
            waypoints[i][0], waypoints[i][1]
        )
    return total


def analyze_segments(waypoints, grid_df, segment_count=10):
    """Break route into segments and classify each."""
    pts_per_seg = max(1, len(waypoints) // segment_count)
    segments = {"high": 0, "medium": 0, "low": 0}
    for i in range(0, len(waypoints), pts_per_seg):
        seg = waypoints[i:i + pts_per_seg + 1]
        if len(seg) < 2:
            continue
        seg_accidents = []
        for lat, lon in seg:
            grid = _find_grid_for_point(lat, lon, grid_df)
            if grid is not None:
                seg_accidents.append(int(grid["total_accidents"]))
            else:
                seg_accidents.append(0)
        avg = np.mean(seg_accidents)
        if avg >= 8:
            segments["high"] += 1
        elif avg >= 3:
            segments["medium"] += 1
        else:
            segments["low"] += 1
    return segments


# --- Destinations ---
START = (13.7467, 100.5348)  # Siam Paragon

DESTINATIONS = [
    ("Chatuchak", 13.80, 100.55),
    ("Victory Monument", 13.765, 100.538),
    ("Huai Khwang", 13.76, 100.75),
    ("Phaya Thai", 13.78, 100.58),
]

grid_df = _load_grid_stats()

print("=" * 70)
print("  NATURAL RISK ANALYSIS — Siam Paragon → Quick Destinations")
print("  (No artificial shortcuts — real grid-based scoring)")
print("=" * 70)

results = []
for name, end_lat, end_lon in DESTINATIONS:
    waypoints = generate_dense_waypoints(START[0], START[1], end_lat, end_lon, num_points=50)
    distance = calculate_distance(waypoints)
    duration = int((distance / 40) * 60)
    risk = calculate_route_risk(waypoints, grid_df)
    segments = analyze_segments(waypoints, grid_df)

    results.append({
        "name": name,
        "distance": distance,
        "duration": duration,
        "risk": risk,
        "segments": segments,
    })

# Print results
for r in results:
    risk = r["risk"]
    seg = r["segments"]
    print()
    print(f"  📍 {r['name']}")
    print(f"  ├─ Distance:      {r['distance']:.2f} km")
    print(f"  ├─ Duration:      {r['duration']} min")
    print(f"  ├─ Risk Score:    {risk['risk_score']}/100")
    print(f"  ├─ Risk Level:    {risk['risk_level']}")
    print(f"  ├─ Matched Grids: {risk['matched_grids']}")
    print(f"  └─ Segments:      {seg['high']} high | {seg['medium']} medium | {seg['low']} low")

print()
print("=" * 70)

# Summary table
print()
print(f"  {'Destination':<20} {'Dist(km)':>9} {'Time(min)':>10} {'Risk':>5} {'Level':>7}  {'H':>2} {'M':>2} {'L':>2}")
print(f"  {'─'*20} {'─'*9} {'─'*10} {'─'*5} {'─'*7}  {'─'*2} {'─'*2} {'─'*2}")
for r in results:
    risk = r["risk"]
    seg = r["segments"]
    print(f"  {r['name']:<20} {r['distance']:>8.2f}  {r['duration']:>9}  {risk['risk_score']:>4}  {risk['risk_level']:>7}  {seg['high']:>2} {seg['medium']:>2} {seg['low']:>2}")

print()
