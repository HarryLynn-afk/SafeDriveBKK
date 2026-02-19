"""Verify all demo routes load correctly and show good risk variety."""
import sys, os, json
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


def calculate_distance(waypoints):
    total = 0.0
    for i in range(1, len(waypoints)):
        total += _haversine_km(waypoints[i-1][0], waypoints[i-1][1], waypoints[i][0], waypoints[i][1])
    return total


def calculate_route_risk(waypoints, grid_df):
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
        "matched_grids": matched_grid_count,
        "max_accidents": max_accidents_in_grid,
        "avg_accidents": round(avg_accidents, 2),
    }


def analyze_route_segments(waypoints, grid_df, segment_distance_km=1.0):
    """Replicate app.py segment analysis exactly."""
    if len(waypoints) < 2:
        return []
    segments = []
    total_distance = 0.0
    for i in range(len(waypoints) - 1):
        total_distance += _haversine_km(waypoints[i][0], waypoints[i][1],
                                         waypoints[i+1][0], waypoints[i+1][1])
    num_segments = max(1, int(total_distance / segment_distance_km))
    points_per_segment = max(1, len(waypoints) // num_segments)
    current_km = 0.0
    for idx in range(0, len(waypoints) - points_per_segment, points_per_segment):
        seg_wps = waypoints[idx:idx + points_per_segment + 1]
        seg_dist = 0.0
        for j in range(len(seg_wps) - 1):
            seg_dist += _haversine_km(seg_wps[j][0], seg_wps[j][1], seg_wps[j+1][0], seg_wps[j+1][1])
        risk_scores = []
        accident_counts = []
        high_risk_count = 0
        for lat, lon in seg_wps:
            grid = _find_grid_for_point(lat, lon, grid_df)
            if grid is not None:
                accident_counts.append(int(grid["total_accidents"]))
                risk_scores.append(float(grid["risk_score"]))
                if str(grid["risk_level"]).lower() == "high":
                    high_risk_count += 1
            else:
                accident_counts.append(0)
                risk_scores.append(0)
        avg_acc = float(np.mean(accident_counts)) if accident_counts else 0.0
        max_acc = float(max(accident_counts)) if accident_counts else 0.0
        segment_accidents = (avg_acc * 0.65) + (max_acc * 0.35)
        temporal_boost = float(high_risk_count)
        seg_risk_score = min(100, int((segment_accidents * 2.0) + (temporal_boost * 4.0)))
        if seg_risk_score >= 50:
            risk_level = "high"
            color = "#dc3545"
        elif seg_risk_score >= 25:
            risk_level = "medium"
            color = "#ffc107"
        else:
            risk_level = "low"
            color = "#28a745"
        segments.append({
            "start_km": round(current_km, 1),
            "end_km": round(current_km + seg_dist, 1),
            "risk_score": seg_risk_score,
            "risk_level": risk_level,
            "color": color,
            "high_risk_zones": high_risk_count,
            "max_accidents": max_acc,
        })
        current_km += seg_dist
    return segments


def generate_alerts(segments, advance_warning_km=0.5):
    """Replicate app.py alert generation exactly."""
    alerts = []
    for seg in segments:
        if seg["risk_level"] == "high":
            warning_km = max(0, seg["start_km"] - advance_warning_km)
            alerts.append({
                "trigger_km": round(warning_km, 1),
                "message": f"WARNING: HIGH RISK ZONE ahead in {int(advance_warning_km * 1000)}m",
                "detail": f"Zone: km {seg['start_km']}-{seg['end_km']} | {int(seg['max_accidents'])} accidents recorded",
                "severity": "warning",
            })
            alerts.append({
                "trigger_km": seg["start_km"],
                "message": "NOW ENTERING HIGH RISK ZONE",
                "detail": f"Extra caution advised | Risk score: {seg['risk_score']}/100",
                "severity": "danger",
            })
            alerts.append({
                "trigger_km": seg["end_km"],
                "message": "Exited high-risk zone - safer area ahead",
                "severity": "success",
            })
        elif seg["risk_level"] == "medium":
            alerts.append({
                "trigger_km": seg["start_km"],
                "message": "Medium risk area - stay alert",
                "detail": f"Risk score: {seg['risk_score']}/100",
                "severity": "info",
            })
    alerts.sort(key=lambda a: a["trigger_km"])
    return alerts


# ─── Load demo routes ───
with open(BASE_DIR / "static" / "demo_routes.json") as f:
    demo_routes = json.load(f)

grid_df = _load_grid_stats()

print("=" * 90)
print("  DEMO ROUTE VERIFICATION")
print("  Testing all routes from static/demo_routes.json")
print("=" * 90)

all_pass = True
check_results = []

for key, route in demo_routes.items():
    print()
    print(f"  Loading route: {key}")
    print(f"  ├─ Name: {route['name']}")

    # Verify structure
    errors = []
    if not route.get("start") or not route.get("end"):
        errors.append("Missing start/end coordinates")
    try:
        s_lat = float(route["start"]["lat"])
        s_lon = float(route["start"]["lon"])
        e_lat = float(route["end"]["lat"])
        e_lon = float(route["end"]["lon"])
    except (KeyError, TypeError, ValueError) as e:
        errors.append(f"Invalid coordinates: {e}")
        print(f"  └─ LOAD FAILED: {errors}")
        all_pass = False
        check_results.append({"key": key, "pass": False, "reason": str(errors)})
        continue

    print(f"  ├─ Start: ({s_lat}, {s_lon})")
    print(f"  ├─ End:   ({e_lat}, {e_lon})")
    print(f"  ├─ LOAD: OK")

    # Generate waypoints & analyze
    waypoints = generate_dense_waypoints(s_lat, s_lon, e_lat, e_lon, num_points=50)
    distance = calculate_distance(waypoints)
    duration = int((distance / 40) * 60)
    risk = calculate_route_risk(waypoints, grid_df)
    segments = analyze_route_segments(waypoints, grid_df)
    alerts = generate_alerts(segments)

    seg_high = sum(1 for s in segments if s["risk_level"] == "high")
    seg_med = sum(1 for s in segments if s["risk_level"] == "medium")
    seg_low = sum(1 for s in segments if s["risk_level"] == "low")
    variety = len(set(s["risk_level"] for s in segments))

    print(f"  │")
    print(f"  ├─ Distance:   {distance:.2f} km")
    print(f"  ├─ Duration:   {duration} min")
    print(f"  ├─ Risk Score: {risk['risk_score']}/100  ({risk['risk_level']})")
    print(f"  ├─ Segments:   {len(segments)} total")
    print(f"  │  ├─ {seg_high} HIGH  (red)     #dc3545")
    print(f"  │  ├─ {seg_med} MEDIUM (yellow)  #ffc107")
    print(f"  │  └─ {seg_low} LOW    (green)   #28a745")
    print(f"  │")

    # Show sample alerts (up to 4)
    print(f"  ├─ Alerts: {len(alerts)} total")
    shown = 0
    for a in alerts:
        if shown >= 4:
            break
        sev_icon = {"danger": "🔴", "warning": "⚠️ ", "info": "⚡", "success": "✅"}.get(a["severity"], "  ")
        connector = "├" if shown < min(3, len(alerts) - 1) else "└"
        detail_str = f'  ({a.get("detail", "")})' if a.get("detail") else ""
        print(f"  │  {connector}─ [{a['trigger_km']:>5.1f} km] {sev_icon} {a['message']}{detail_str}")
        shown += 1
    if len(alerts) > 4:
        print(f"  │  └─ ... and {len(alerts) - 4} more alerts")

    # Verification checks
    print(f"  │")
    checks = {
        "loads_ok": True,
        "risk_not_low": risk["risk_level"] != "LOW",
        "has_variety": variety >= 2,
        "alerts_appropriate": len(alerts) > 0,
    }

    for label, passed in checks.items():
        icon = "✅" if passed else "❌"
        desc = {
            "loads_ok": "Route loads without errors",
            "risk_not_low": f"Risk is MEDIUM or HIGH (got {risk['risk_level']})",
            "has_variety": f"Has color variety ({variety} distinct levels)",
            "alerts_appropriate": f"Has alerts ({len(alerts)} generated)",
        }[label]
        print(f"  ├─ {icon} {desc}")
        if not passed:
            all_pass = False

    route_pass = all(checks.values())
    print(f"  └─ {'✅ PASS' if route_pass else '❌ FAIL'}")

    check_results.append({
        "key": key,
        "name": route["name"],
        "distance": round(distance, 2),
        "duration": duration,
        "risk_score": risk["risk_score"],
        "risk_level": risk["risk_level"],
        "seg_high": seg_high,
        "seg_med": seg_med,
        "seg_low": seg_low,
        "alerts": len(alerts),
        "pass": route_pass,
    })

# ─── Summary ───
print()
print("=" * 90)
print("  SUMMARY")
print("=" * 90)
print()
print(f"  {'Route':<28} {'km':>6} {'min':>4}  {'Risk':>4} {'Level':>6}  "
      f"{'🔴':>2} {'🟡':>2} {'🟢':>2}  {'Alerts':>6}  {'Status':>6}")
print(f"  {'─'*28} {'─'*6} {'─'*4}  {'─'*4} {'─'*6}  "
      f"{'─'*2} {'─'*2} {'─'*2}  {'─'*6}  {'─'*6}")

for r in check_results:
    if not r["pass"] or "name" not in r:
        print(f"  {r['key']:<28} {'FAILED':>50}")
        continue
    status = "PASS" if r["pass"] else "FAIL"
    print(f"  {r['name']:<28} {r['distance']:>5.1f} {r['duration']:>4}  "
          f"{r['risk_score']:>4} {r['risk_level']:>6}  "
          f"{r['seg_high']:>2} {r['seg_med']:>2} {r['seg_low']:>2}  "
          f"{r['alerts']:>6}  {status:>6}")

print()
passed = sum(1 for r in check_results if r["pass"])
total = len(check_results)
final = "ALL TESTS PASSED" if all_pass else f"{passed}/{total} PASSED"
print(f"  {'✅' if all_pass else '❌'} {final}")
print()
print("=" * 90)
