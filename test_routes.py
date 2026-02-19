"""Test natural risk levels for 20 destinations — find best demo routes."""
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
    return pd.read_csv(GRID_STATS_PATH)


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


def generate_dense_waypoints(start_lat, start_lon, end_lat, end_lon, num_points=50):
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
        "max_accidents": max_accidents_in_grid,
        "avg_accidents": round(avg_accidents, 2),
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


# ─── Config ───
START = (13.7467, 100.5348)  # Siam Paragon

DESTINATIONS = [
    ("Ratchada",          13.73, 100.72),
    ("Din Daeng",         13.73, 100.74),
    ("Bang Sue",          13.82, 100.55),
    ("Lat Phrao",         13.79, 100.60),
    ("Rama IX",           13.76, 100.57),
    ("Asok",              13.74, 100.56),
    ("Ekkamai",           13.72, 100.58),
    ("Thong Lo",          13.73, 100.59),
    ("Phra Khanong",      13.71, 100.60),
    ("On Nut",            13.70, 100.61),
    ("Bearing",           13.66, 100.62),
    ("Samut Prakan",      13.60, 100.60),
    ("Rangsit",           14.03, 100.60),
    ("Don Mueang",        13.92, 100.61),
    ("Saphan Mai",        13.88, 100.57),
    ("Chatuchak",         13.80, 100.55),
    ("Huai Khwang",       13.76, 100.75),
    ("Ramkhamhaeng",      13.75, 100.62),
    ("Min Buri",          13.82, 100.72),
    ("Lak Si",            13.88, 100.54),
]

grid_df = _load_grid_stats()

# ─── Analyze all routes ───
results = []
for name, end_lat, end_lon in DESTINATIONS:
    waypoints = generate_dense_waypoints(START[0], START[1], end_lat, end_lon, num_points=50)
    distance = calculate_distance(waypoints)
    duration = int((distance / 40) * 60)
    risk = calculate_route_risk(waypoints, grid_df)
    segments = analyze_segments(waypoints, grid_df)

    # Determine demo-worthiness
    variety = segments["high"] + segments["medium"]
    if risk["risk_level"] == "HIGH" and variety >= 2:
        demo = "GREAT"
    elif risk["risk_level"] == "HIGH":
        demo = "YES"
    elif risk["risk_level"] == "MEDIUM" and variety >= 3:
        demo = "GOOD"
    elif risk["risk_level"] == "MEDIUM":
        demo = "OK"
    else:
        demo = "-"

    results.append({
        "name": name,
        "lat": end_lat,
        "lon": end_lon,
        "distance": distance,
        "duration": duration,
        "risk": risk,
        "segments": segments,
        "variety": variety,
        "demo": demo,
    })

# Sort by risk score descending
results.sort(key=lambda r: (-r["risk"]["risk_score"], -r["variety"]))

# ─── Full ranking table ───
print("=" * 90)
print("  ROUTE RISK SCAN — 20 Destinations from Siam Paragon")
print("  Sorted by risk score (highest first)")
print("=" * 90)
print()
print(f"  {'#':>2}  {'Destination':<18} {'km':>6} {'min':>4}  {'Risk':>4} {'Level':>6}  "
      f"{'H':>2} {'M':>2} {'L':>2}  {'MaxAcc':>6} {'AvgAcc':>6}  {'Demo?':>5}")
print(f"  {'─'*2}  {'─'*18} {'─'*6} {'─'*4}  {'─'*4} {'─'*6}  "
      f"{'─'*2} {'─'*2} {'─'*2}  {'─'*6} {'─'*6}  {'─'*5}")

for i, r in enumerate(results, 1):
    rk = r["risk"]
    sg = r["segments"]
    level_tag = rk["risk_level"]
    print(f"  {i:>2}  {r['name']:<18} {r['distance']:>5.1f} {r['duration']:>4}  "
          f"{rk['risk_score']:>4} {level_tag:>6}  "
          f"{sg['high']:>2} {sg['medium']:>2} {sg['low']:>2}  "
          f"{rk['max_accidents']:>6} {rk['avg_accidents']:>6}  "
          f"{r['demo']:>5}")

# ─── Top 5 demo picks ───
top = [r for r in results if r["demo"] in ("GREAT", "YES", "GOOD")][:5]
if len(top) < 5:
    top += [r for r in results if r not in top and r["demo"] == "OK"][:5 - len(top)]
if len(top) < 5:
    top += [r for r in results if r not in top][:5 - len(top)]

print()
print("=" * 90)
print("  TOP 5 DEMO ROUTES")
print("=" * 90)

for i, r in enumerate(top, 1):
    rk = r["risk"]
    sg = r["segments"]
    print()
    print(f"  #{i}  {r['name']}")
    print(f"      Coords:     ({r['lat']}, {r['lon']})")
    print(f"      Distance:   {r['distance']:.2f} km  |  Duration: {r['duration']} min")
    print(f"      Risk Score: {rk['risk_score']}/100  ({rk['risk_level']})")
    print(f"      Segments:   {sg['high']} high (red) | {sg['medium']} medium (yellow) | {sg['low']} low (green)")
    print(f"      Max accidents in a grid: {rk['max_accidents']}  |  Avg: {rk['avg_accidents']}")
    print(f"      Demo grade: {r['demo']}")

print()
print("=" * 90)
