import math
from collections import Counter

import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    radius_km = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2) + (math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return radius_km * c


def _resolve_coordinate_columns(df):
    lat_candidates = ["latitude", "LATITUDE", "lat", "LAT"]
    lon_candidates = ["longitude", "LONGITUDE", "lon", "LON"]
    lat_col = next((column for column in lat_candidates if column in df.columns), None)
    lon_col = next((column for column in lon_candidates if column in df.columns), None)
    if lat_col is None or lon_col is None:
        raise ValueError("Could not find latitude/longitude columns in bangkok_accidents_clean.csv")
    return lat_col, lon_col


def find_waypoints_between(start_lat, start_lon, end_lat, end_lon, danger_zones, n_waypoints=3):
    """Find best intermediate waypoints between start and end."""
    direct_dist = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    if direct_dist <= 0:
        return []

    path_lat = end_lat - start_lat
    path_lon = end_lon - start_lon
    path_norm_sq = (path_lat * path_lat) + (path_lon * path_lon)
    max_detour = max(3.0, direct_dist * 0.35)

    candidates = []
    for zone in danger_zones:
        z_lat = float(zone["lat"])
        z_lon = float(zone["lon"])
        accidents = int(zone["accidents"])

        if accidents < 5:
            continue

        if path_norm_sq > 0:
            zone_lat = z_lat - start_lat
            zone_lon = z_lon - start_lon
            projection = ((zone_lat * path_lat) + (zone_lon * path_lon)) / path_norm_sq
            if projection < 0.05 or projection > 0.98:
                continue
            closest_lat = start_lat + (projection * path_lat)
            closest_lon = start_lon + (projection * path_lon)
            path_offset = haversine_distance(z_lat, z_lon, closest_lat, closest_lon)
            if path_offset > max(1.6, direct_dist * 0.08):
                continue

        dist_to_start = haversine_distance(start_lat, start_lon, z_lat, z_lon)
        dist_to_end = haversine_distance(z_lat, z_lon, end_lat, end_lon)
        detour = (dist_to_start + dist_to_end) - direct_dist

        if detour <= max_detour:
            proximity_score = 1 / (detour + 0.5)
            score = accidents * proximity_score
            candidates.append(
                {
                    "lat": z_lat,
                    "lon": z_lon,
                    "accidents": accidents,
                    "detour": detour,
                    "score": score,
                    "dist_from_start": dist_to_start,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[:n_waypoints]
    selected.sort(key=lambda item: item["dist_from_start"])
    return selected


def find_high_risk_coordinates():
    """Find actual high-risk coordinates from accident data."""
    accidents = pd.read_csv("data/processed/bangkok_accidents_clean.csv")
    lat_col, lon_col = _resolve_coordinate_columns(accidents)
    accidents = accidents.copy()
    accidents[lat_col] = pd.to_numeric(accidents[lat_col], errors="coerce")
    accidents[lon_col] = pd.to_numeric(accidents[lon_col], errors="coerce")
    accidents = accidents.dropna(subset=[lat_col, lon_col])

    print(f"Total accidents: {len(accidents)}")
    print("=" * 70)

    accidents["grid_lat"] = accidents[lat_col].round(2)
    accidents["grid_lon"] = accidents[lon_col].round(2)

    grid_counts = Counter()
    for _, row in accidents.iterrows():
        grid_counts[(row["grid_lat"], row["grid_lon"])] += 1

    print("\nTOP 30 DANGER ZONES:")
    print("-" * 70)
    print(f"{'Rank':<6}{'Latitude':<12}{'Longitude':<12}{'Accidents':<12}")
    print("-" * 70)

    danger_zones_all = []
    for index, ((lat, lon), count) in enumerate(grid_counts.most_common(), 1):
        zone = {"lat": float(lat), "lon": float(lon), "accidents": int(count)}
        danger_zones_all.append(zone)
        if index <= 30:
            print(f"{index:<6}{lat:<12.2f}{lon:<12.2f}{count:<12}")

    print("\n" + "=" * 70)
    print("AUTO-SELECTED WAYPOINTS FOR EACH ROUTE:")
    print("=" * 70)

    siam_lat, siam_lon = 13.7467, 100.5348

    print("\nCHATUCHAK MARKET (13.80, 100.55):")
    print("-" * 70)
    chatuchak_wps = find_waypoints_between(siam_lat, siam_lon, 13.80, 100.55, danger_zones_all, n_waypoints=3)
    print("'chatuchak': [")
    print(f"    [{siam_lat}, {siam_lon}],  # Start: Siam")
    for waypoint in chatuchak_wps:
        print(
            f"    [{waypoint['lat']:.2f}, {waypoint['lon']:.2f}],"
            f"  # {waypoint['accidents']} accidents, {waypoint['detour']:.1f}km detour"
        )
    print("    [13.80, 100.55]  # End: Chatuchak")
    print("],")

    print("\nVICTORY MONUMENT (13.765, 100.538):")
    print("-" * 70)
    victory_wps = find_waypoints_between(siam_lat, siam_lon, 13.765, 100.538, danger_zones_all, n_waypoints=2)
    print("'victory': [")
    print(f"    [{siam_lat}, {siam_lon}],  # Start: Siam")
    for waypoint in victory_wps:
        print(
            f"    [{waypoint['lat']:.2f}, {waypoint['lon']:.2f}],"
            f"  # {waypoint['accidents']} accidents, {waypoint['detour']:.1f}km detour"
        )
    print("    [13.765, 100.538]  # End: Victory")
    print("],")

    print("\nPHAYA THAI (13.78, 100.58):")
    print("-" * 70)
    phaya_wps = find_waypoints_between(siam_lat, siam_lon, 13.78, 100.58, danger_zones_all, n_waypoints=3)
    print("'phaya_thai': [")
    print(f"    [{siam_lat}, {siam_lon}],  # Start: Siam")
    for waypoint in phaya_wps:
        print(
            f"    [{waypoint['lat']:.2f}, {waypoint['lon']:.2f}],"
            f"  # {waypoint['accidents']} accidents, {waypoint['detour']:.1f}km detour"
        )
    print("    [13.78, 100.58]  # End: Phaya Thai")
    print("],")

    print("\n" + "=" * 70)
    print("COPY THE ABOVE WAYPOINTS TO app.py HIGH_RISK_WAYPOINTS!")
    print("=" * 70)


if __name__ == "__main__":
    find_high_risk_coordinates()
