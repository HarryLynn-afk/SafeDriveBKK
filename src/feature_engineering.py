"""
Feature Engineering for XGBoost Risk Prediction

Transforms raw accident data + grid stats into ML-ready features.
Binary target: dangerous (1) if grid has 5+ accidents, safe (0) otherwise.
Aligns with the existing 83.8% holdout validation threshold.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Paths
ACCIDENTS_PATH = Path("data/processed/bangkok_accidents_clean.csv")
GRID_STATS_PATH = Path("data/processed/bangkok_grid_stats.csv")
OUTPUT_FEATURES_PATH = Path("data/processed/ml_features.csv")
OUTPUT_INFO_PATH = Path("data/processed/feature_info.json")

# Grid constants (must match grid_aggregation.py)
LAT_MIN, LAT_MAX = 13.6, 13.95
LON_MIN, LON_MAX = 100.35, 100.9
CELL_SIZE_DEG = 0.009

# Bangkok center (approximate)
BANGKOK_CENTER_LAT = 13.7563
BANGKOK_CENTER_LON = 100.5018

# Danger threshold (matches evidence_metrics.py holdout validation)
DANGER_THRESHOLD = 5

# Thai column mappings
COL_DEATHS = "ผู้เสียชีวิต"
COL_SERIOUS_INJURY = "ผู้บาดเจ็บสาหัส"
COL_MINOR_INJURY = "ผู้บาดเจ็บเล็กน้อย"
COL_WEATHER = "สภาพอากาศ"
COL_CAUSE = "มูลเหตุสันนิษฐาน"
COL_ACCIDENT_TYPE = "ลักษณะการเกิดเหตุ"
COL_MOTORCYCLES = "รถจักรยานยนต์"
COL_PASSENGER_CARS = "รถยนต์นั่งส่วนบุคคล"
COL_PICKUPS = "รถปิคอัพบรรทุก4ล้อ"
COL_TRUCKS_6W = "รถบรรทุก6ล้อ"
COL_TRUCKS_10W = "รถบรรทุกไม่เกิน10ล้อ"
COL_TRUCKS_10PLUS = "รถบรรทุกมากกว่า10ล้อ"
COL_VEHICLES = "รถที่เกิดเหตุ"


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two points."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _parse_grid_indices(grid_id):
    """Extract (lat_idx, lon_idx) from grid_id like 'G020_002'."""
    parts = grid_id.replace("G", "").split("_")
    return int(parts[0]), int(parts[1])


def load_data():
    """Load accident records and grid statistics."""
    accidents = pd.read_csv(ACCIDENTS_PATH, encoding="utf-8-sig")
    accidents["datetime"] = pd.to_datetime(accidents["datetime"], errors="coerce")
    LOGGER.info("Loaded %d accident records", len(accidents))

    grid_stats = pd.read_csv(GRID_STATS_PATH, encoding="utf-8-sig")
    LOGGER.info("Loaded %d grid cells", len(grid_stats))

    return accidents, grid_stats


def build_grid_accident_lookup(grid_stats):
    """Build a dict mapping (lat_idx, lon_idx) -> total_accidents for fast neighbor lookup."""
    lookup = {}
    for _, row in grid_stats.iterrows():
        lat_idx, lon_idx = _parse_grid_indices(row["grid_id"])
        lookup[(lat_idx, lon_idx)] = int(row["total_accidents"])
    return lookup


def calculate_neighbor_features(grid_stats, accident_lookup):
    """Calculate accident counts in 8 surrounding grid cells for each cell."""
    directions = {
        "n": (-1, 0), "s": (1, 0), "e": (0, 1), "w": (0, -1),
        "ne": (-1, 1), "nw": (-1, -1), "se": (1, 1), "sw": (1, -1),
    }

    neighbor_data = []
    for _, row in grid_stats.iterrows():
        lat_idx, lon_idx = _parse_grid_indices(row["grid_id"])
        record = {"grid_id": row["grid_id"]}

        total_nearby = 0
        for direction, (dlat, dlon) in directions.items():
            count = accident_lookup.get((lat_idx + dlat, lon_idx + dlon), 0)
            record[f"accidents_{direction}"] = count
            total_nearby += count

        record["accidents_neighbors_total"] = total_nearby
        record["accidents_neighbors_mean"] = total_nearby / 8.0
        record["accidents_neighbors_max"] = max(
            accident_lookup.get((lat_idx + d[0], lon_idx + d[1]), 0)
            for d in directions.values()
        )
        neighbor_data.append(record)

    return pd.DataFrame(neighbor_data)


def calculate_spatial_features(grid_stats):
    """Distance from Bangkok center and grid density."""
    features = grid_stats[["grid_id", "center_lat", "center_lon"]].copy()

    features["distance_from_center_km"] = features.apply(
        lambda r: _haversine_km(
            r["center_lat"], r["center_lon"],
            BANGKOK_CENTER_LAT, BANGKOK_CENTER_LON,
        ),
        axis=1,
    ).round(3)

    cell_area_km2 = (CELL_SIZE_DEG * 111) ** 2  # ~1 km^2
    features["grid_density"] = (
        grid_stats["total_accidents"] / cell_area_km2
    ).round(3)

    return features


def aggregate_accident_features(accidents, grid_stats):
    """Aggregate accident-level data to grid-level features."""
    # Assign accidents to grids
    accidents = accidents.copy()
    accidents["grid_lat_idx"] = np.floor(
        (accidents["LATITUDE"] - LAT_MIN) / CELL_SIZE_DEG
    ).clip(0).astype(int)
    accidents["grid_lon_idx"] = np.floor(
        (accidents["LONGITUDE"] - LON_MIN) / CELL_SIZE_DEG
    ).clip(0).astype(int)
    accidents["grid_id"] = [
        f"G{lat:03d}_{lon:03d}"
        for lat, lon in zip(accidents["grid_lat_idx"], accidents["grid_lon_idx"])
    ]

    # Keep only accidents in known grid cells
    valid_grids = set(grid_stats["grid_id"])
    accidents = accidents[accidents["grid_id"].isin(valid_grids)].copy()

    # --- Severity features ---
    def safe_numeric(series):
        return pd.to_numeric(series, errors="coerce").fillna(0)

    accidents["severity_score"] = (
        safe_numeric(accidents.get(COL_DEATHS, 0)) * 10
        + safe_numeric(accidents.get(COL_SERIOUS_INJURY, 0)) * 5
        + safe_numeric(accidents.get(COL_MINOR_INJURY, 0)) * 1
    )

    severity_agg = accidents.groupby("grid_id").agg(
        severity_total=("severity_score", "sum"),
        severity_mean=("severity_score", "mean"),
        has_fatality=(COL_DEATHS, lambda x: int((safe_numeric(x) > 0).any())),
        max_vehicles=(COL_VEHICLES, "max"),
    ).reset_index()
    severity_agg["severity_mean"] = severity_agg["severity_mean"].round(3)

    # --- Temporal features per grid ---
    temporal_agg = accidents.groupby("grid_id").agg(
        rush_hour_pct=("is_rush_hour", lambda x: round(x.astype(bool).mean() * 100, 2)),
        weekend_pct=("is_weekend", lambda x: round(x.astype(bool).mean() * 100, 2)),
        peak_hour=("hour", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12),
        peak_day=("day_of_week", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
        month_spread=("month", "nunique"),
    ).reset_index()

    # --- Vehicle mix features per grid ---
    vehicle_agg = accidents.groupby("grid_id").agg(
        motorcycle_count=(COL_MOTORCYCLES, lambda x: safe_numeric(x).sum()),
        car_count=(COL_PASSENGER_CARS, lambda x: safe_numeric(x).sum()),
        pickup_count=(COL_PICKUPS, lambda x: safe_numeric(x).sum()),
        heavy_truck_count=(COL_TRUCKS_6W, lambda x: (
            safe_numeric(x).sum()
            + safe_numeric(accidents.loc[x.index, COL_TRUCKS_10W]).sum()
            + safe_numeric(accidents.loc[x.index, COL_TRUCKS_10PLUS]).sum()
        )),
        total_vehicles=(COL_VEHICLES, lambda x: safe_numeric(x).sum()),
    ).reset_index()

    # Convert to percentages (avoid division by zero)
    for col in ["motorcycle_count", "car_count", "pickup_count", "heavy_truck_count"]:
        pct_col = col.replace("_count", "_pct")
        vehicle_agg[pct_col] = np.where(
            vehicle_agg["total_vehicles"] > 0,
            (vehicle_agg[col] / vehicle_agg["total_vehicles"] * 100).round(2),
            0.0,
        )

    vehicle_agg = vehicle_agg.drop(columns=["total_vehicles"])

    # --- Weather features per grid ---
    # Map Thai weather to categories
    weather_map = {}
    if COL_WEATHER in accidents.columns:
        weather_values = accidents[COL_WEATHER].dropna().unique()
        for w in weather_values:
            w_lower = str(w).lower()
            if "ฝน" in w_lower or "rain" in w_lower:
                weather_map[w] = "rain"
            elif "แจ่มใส" in w_lower or "clear" in w_lower:
                weather_map[w] = "clear"
            else:
                weather_map[w] = "other"

        accidents["weather_cat"] = accidents[COL_WEATHER].map(weather_map).fillna("other")

        weather_agg = accidents.groupby("grid_id").agg(
            clear_weather_pct=("weather_cat", lambda x: round((x == "clear").mean() * 100, 2)),
            rain_weather_pct=("weather_cat", lambda x: round((x == "rain").mean() * 100, 2)),
        ).reset_index()
    else:
        weather_agg = pd.DataFrame({"grid_id": grid_stats["grid_id"]})
        weather_agg["clear_weather_pct"] = 0.0
        weather_agg["rain_weather_pct"] = 0.0

    # --- Accident cause features per grid ---
    if COL_CAUSE in accidents.columns:
        top_causes = accidents[COL_CAUSE].value_counts().head(5).index.tolist()
        cause_dummies = pd.get_dummies(
            accidents[COL_CAUSE].where(accidents[COL_CAUSE].isin(top_causes), "other"),
            prefix="cause",
        )
        cause_dummies["grid_id"] = accidents["grid_id"].values
        cause_agg = cause_dummies.groupby("grid_id").mean().round(4).reset_index()
        # Rename to proportion columns
        for col in cause_agg.columns:
            if col.startswith("cause_"):
                cause_agg[col] = (cause_agg[col] * 100).round(2)
    else:
        cause_agg = pd.DataFrame({"grid_id": grid_stats["grid_id"]})

    # --- Accident type features per grid ---
    if COL_ACCIDENT_TYPE in accidents.columns:
        top_types = accidents[COL_ACCIDENT_TYPE].value_counts().head(5).index.tolist()
        type_dummies = pd.get_dummies(
            accidents[COL_ACCIDENT_TYPE].where(
                accidents[COL_ACCIDENT_TYPE].isin(top_types), "other"
            ),
            prefix="type",
        )
        type_dummies["grid_id"] = accidents["grid_id"].values
        type_agg = type_dummies.groupby("grid_id").mean().round(4).reset_index()
        for col in type_agg.columns:
            if col.startswith("type_"):
                type_agg[col] = (type_agg[col] * 100).round(2)
    else:
        type_agg = pd.DataFrame({"grid_id": grid_stats["grid_id"]})

    # Merge all aggregations
    result = severity_agg
    for df in [temporal_agg, vehicle_agg, weather_agg, cause_agg, type_agg]:
        result = result.merge(df, on="grid_id", how="outer")

    return result


def create_ml_dataset():
    """
    Build the full ML feature matrix.

    Returns (features_df, feature_names, target_col) where:
    - features_df has all grid cells with features + binary target
    - feature_names lists the feature column names
    - target_col is 'dangerous'
    """
    accidents, grid_stats = load_data()

    # 1. Base grid data + target
    ml_df = grid_stats[["grid_id", "center_lat", "center_lon", "total_accidents"]].copy()
    ml_df["dangerous"] = (ml_df["total_accidents"] >= DANGER_THRESHOLD).astype(int)

    # 2. Neighbor features
    accident_lookup = build_grid_accident_lookup(grid_stats)
    neighbor_features = calculate_neighbor_features(grid_stats, accident_lookup)
    ml_df = ml_df.merge(neighbor_features, on="grid_id", how="left")

    # 3. Spatial features
    spatial_features = calculate_spatial_features(grid_stats)
    ml_df = ml_df.merge(spatial_features, on="grid_id", how="left", suffixes=("", "_dup"))
    dup_cols = [c for c in ml_df.columns if c.endswith("_dup")]
    ml_df = ml_df.drop(columns=dup_cols)

    # 4. Accident-level aggregated features
    agg_features = aggregate_accident_features(accidents, grid_stats)
    ml_df = ml_df.merge(agg_features, on="grid_id", how="left")

    # Fill NaN for grids with 0 accidents (no accident-level data to aggregate)
    numeric_cols = ml_df.select_dtypes(include=[np.number]).columns
    ml_df[numeric_cols] = ml_df[numeric_cols].fillna(0)

    # Identify feature columns (everything except identifiers and target)
    exclude_cols = {"grid_id", "center_lat", "center_lon", "total_accidents", "dangerous"}
    feature_names = [c for c in ml_df.columns if c not in exclude_cols]

    return ml_df, feature_names, "dangerous"


def save_outputs(ml_df, feature_names, target_col):
    """Save ML features CSV and info JSON."""
    OUTPUT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(OUTPUT_FEATURES_PATH, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved %d rows x %d columns to %s", len(ml_df), len(ml_df.columns), OUTPUT_FEATURES_PATH)

    n_dangerous = int(ml_df[target_col].sum())
    n_safe = int(len(ml_df) - n_dangerous)

    info = {
        "total_grids": len(ml_df),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_column": target_col,
        "danger_threshold": DANGER_THRESHOLD,
        "class_distribution": {
            "dangerous_1": n_dangerous,
            "safe_0": n_safe,
            "imbalance_ratio": round(n_safe / max(n_dangerous, 1), 2),
        },
    }

    with open(OUTPUT_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved feature info to %s", OUTPUT_INFO_PATH)

    return info


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("FEATURE ENGINEERING FOR XGBOOST RISK PREDICTION")
    print("=" * 70)

    ml_df, feature_names, target_col = create_ml_dataset()
    info = save_outputs(ml_df, feature_names, target_col)

    # Summary
    print(f"\nDataset shape: {ml_df.shape[0]} grids x {len(feature_names)} features")
    print(f"Target: '{target_col}' (1 = dangerous, 0 = safe)")
    print(f"Class distribution:")
    print(f"  Safe (0):      {info['class_distribution']['safe_0']:,}")
    print(f"  Dangerous (1): {info['class_distribution']['dangerous_1']:,}")
    print(f"  Imbalance ratio: {info['class_distribution']['imbalance_ratio']}:1")

    print(f"\nFeatures ({len(feature_names)}):")
    for i, name in enumerate(feature_names, 1):
        sample = ml_df[name].describe()
        print(f"  {i:2d}. {name:<35s}  min={sample['min']:.2f}  max={sample['max']:.2f}  mean={sample['mean']:.2f}")

    print(f"\nSaved: {OUTPUT_FEATURES_PATH}")
    print(f"Saved: {OUTPUT_INFO_PATH}")


if __name__ == "__main__":
    main()
