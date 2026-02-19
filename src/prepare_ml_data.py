"""
Temporal Train/Test Split for ML

Splits accidents by month (Jan-Sep train, Oct-Dec test), then
re-aggregates grid-level features independently for each period.
This prevents data leakage: test features only use Oct-Dec accidents.

Matches the holdout validation in evidence_metrics.py.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import (
    ACCIDENTS_PATH,
    BANGKOK_CENTER_LAT,
    BANGKOK_CENTER_LON,
    CELL_SIZE_DEG,
    COL_ACCIDENT_TYPE,
    COL_CAUSE,
    COL_DEATHS,
    COL_MINOR_INJURY,
    COL_MOTORCYCLES,
    COL_PASSENGER_CARS,
    COL_PICKUPS,
    COL_SERIOUS_INJURY,
    COL_TRUCKS_6W,
    COL_TRUCKS_10W,
    COL_TRUCKS_10PLUS,
    COL_VEHICLES,
    COL_WEATHER,
    DANGER_THRESHOLD,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    _haversine_km,
    _parse_grid_indices,
)

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")

TRAIN_MONTHS = list(range(1, 10))  # Jan-Sep
TEST_MONTHS = list(range(10, 13))  # Oct-Dec


def _safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _assign_grid_ids(accidents):
    """Assign grid_id to each accident row."""
    df = accidents.copy()
    df["grid_lat_idx"] = np.floor(
        (df["LATITUDE"] - LAT_MIN) / CELL_SIZE_DEG
    ).clip(0).astype(int)
    df["grid_lon_idx"] = np.floor(
        (df["LONGITUDE"] - LON_MIN) / CELL_SIZE_DEG
    ).clip(0).astype(int)
    df["grid_id"] = [
        f"G{lat:03d}_{lon:03d}"
        for lat, lon in zip(df["grid_lat_idx"], df["grid_lon_idx"])
    ]
    return df


def _build_period_grid_stats(accidents, all_grid_ids):
    """
    Count accidents per grid for a specific time period.
    Returns a DataFrame with all grid IDs (including zero-count grids).
    """
    if len(accidents) == 0:
        counts = pd.DataFrame({"grid_id": list(all_grid_ids), "total_accidents": 0})
    else:
        counts = (
            accidents.groupby("grid_id")
            .size()
            .reset_index(name="total_accidents")
        )

    # Ensure all grids present
    all_grids_df = pd.DataFrame({"grid_id": list(all_grid_ids)})
    counts = all_grids_df.merge(counts, on="grid_id", how="left")
    counts["total_accidents"] = counts["total_accidents"].fillna(0).astype(int)
    return counts


def _build_neighbor_features(grid_counts):
    """Neighbor accident counts from period-specific grid counts."""
    lookup = {}
    for _, row in grid_counts.iterrows():
        lat_idx, lon_idx = _parse_grid_indices(row["grid_id"])
        lookup[(lat_idx, lon_idx)] = int(row["total_accidents"])

    directions = {
        "n": (-1, 0), "s": (1, 0), "e": (0, 1), "w": (0, -1),
        "ne": (-1, 1), "nw": (-1, -1), "se": (1, 1), "sw": (1, -1),
    }

    rows = []
    for _, row in grid_counts.iterrows():
        lat_idx, lon_idx = _parse_grid_indices(row["grid_id"])
        record = {"grid_id": row["grid_id"]}
        total_nearby = 0
        for direction, (dlat, dlon) in directions.items():
            count = lookup.get((lat_idx + dlat, lon_idx + dlon), 0)
            record[f"accidents_{direction}"] = count
            total_nearby += count
        record["accidents_neighbors_total"] = total_nearby
        record["accidents_neighbors_mean"] = total_nearby / 8.0
        record["accidents_neighbors_max"] = max(
            lookup.get((lat_idx + d[0], lon_idx + d[1]), 0)
            for d in directions.values()
        )
        rows.append(record)

    return pd.DataFrame(rows)


def _build_spatial_features(grid_counts):
    """Distance from center + density (uses period-specific counts)."""
    features = grid_counts[["grid_id"]].copy()

    # Parse center coords from grid_id
    cell_area_km2 = (CELL_SIZE_DEG * 111) ** 2

    centers = []
    for gid in grid_counts["grid_id"]:
        lat_idx, lon_idx = _parse_grid_indices(gid)
        clat = LAT_MIN + (lat_idx + 0.5) * CELL_SIZE_DEG
        clon = LON_MIN + (lon_idx + 0.5) * CELL_SIZE_DEG
        centers.append((clat, clon))

    features["center_lat"] = [c[0] for c in centers]
    features["center_lon"] = [c[1] for c in centers]
    features["distance_from_center_km"] = [
        round(_haversine_km(c[0], c[1], BANGKOK_CENTER_LAT, BANGKOK_CENTER_LON), 3)
        for c in centers
    ]
    features["grid_density"] = (
        grid_counts["total_accidents"].values / cell_area_km2
    ).round(3)

    return features


def _build_accident_agg_features(accidents, all_grid_ids, top_causes, top_types):
    """
    Aggregate accident-level features to grid level.
    Uses pre-determined top_causes and top_types for consistent columns.
    """
    if len(accidents) == 0:
        return pd.DataFrame({"grid_id": list(all_grid_ids)})

    acc = accidents.copy()

    # --- Severity ---
    acc["severity_score"] = (
        _safe_numeric(acc.get(COL_DEATHS, 0)) * 10
        + _safe_numeric(acc.get(COL_SERIOUS_INJURY, 0)) * 5
        + _safe_numeric(acc.get(COL_MINOR_INJURY, 0)) * 1
    )

    severity_agg = acc.groupby("grid_id").agg(
        severity_total=("severity_score", "sum"),
        severity_mean=("severity_score", "mean"),
        has_fatality=(COL_DEATHS, lambda x: int((_safe_numeric(x) > 0).any())),
        max_vehicles=(COL_VEHICLES, "max"),
    ).reset_index()
    severity_agg["severity_mean"] = severity_agg["severity_mean"].round(3)

    # --- Temporal ---
    temporal_agg = acc.groupby("grid_id").agg(
        rush_hour_pct=("is_rush_hour", lambda x: round(x.astype(bool).mean() * 100, 2)),
        weekend_pct=("is_weekend", lambda x: round(x.astype(bool).mean() * 100, 2)),
        peak_hour=("hour", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12),
        peak_day=("day_of_week", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0),
        month_spread=("month", "nunique"),
    ).reset_index()

    # --- Vehicle mix ---
    vehicle_agg = acc.groupby("grid_id").agg(
        motorcycle_count=(COL_MOTORCYCLES, lambda x: _safe_numeric(x).sum()),
        car_count=(COL_PASSENGER_CARS, lambda x: _safe_numeric(x).sum()),
        pickup_count=(COL_PICKUPS, lambda x: _safe_numeric(x).sum()),
        heavy_truck_count=(COL_TRUCKS_6W, lambda x: (
            _safe_numeric(x).sum()
            + _safe_numeric(acc.loc[x.index, COL_TRUCKS_10W]).sum()
            + _safe_numeric(acc.loc[x.index, COL_TRUCKS_10PLUS]).sum()
        )),
        total_vehicles=(COL_VEHICLES, lambda x: _safe_numeric(x).sum()),
    ).reset_index()

    for col in ["motorcycle_count", "car_count", "pickup_count", "heavy_truck_count"]:
        pct_col = col.replace("_count", "_pct")
        vehicle_agg[pct_col] = np.where(
            vehicle_agg["total_vehicles"] > 0,
            (vehicle_agg[col] / vehicle_agg["total_vehicles"] * 100).round(2),
            0.0,
        )
    vehicle_agg = vehicle_agg.drop(columns=["total_vehicles"])

    # --- Weather ---
    weather_map = {}
    if COL_WEATHER in acc.columns:
        for w in acc[COL_WEATHER].dropna().unique():
            w_lower = str(w).lower()
            if "ฝน" in w_lower or "rain" in w_lower:
                weather_map[w] = "rain"
            elif "แจ่มใส" in w_lower or "clear" in w_lower:
                weather_map[w] = "clear"
            else:
                weather_map[w] = "other"
        acc["weather_cat"] = acc[COL_WEATHER].map(weather_map).fillna("other")
        weather_agg = acc.groupby("grid_id").agg(
            clear_weather_pct=("weather_cat", lambda x: round((x == "clear").mean() * 100, 2)),
            rain_weather_pct=("weather_cat", lambda x: round((x == "rain").mean() * 100, 2)),
        ).reset_index()
    else:
        weather_agg = pd.DataFrame({
            "grid_id": list(all_grid_ids),
            "clear_weather_pct": 0.0,
            "rain_weather_pct": 0.0,
        })

    # --- Cause (fixed columns from training top_causes) ---
    if COL_CAUSE in acc.columns and top_causes:
        acc["cause_mapped"] = acc[COL_CAUSE].where(
            acc[COL_CAUSE].isin(top_causes), "other"
        )
        cause_dummies = pd.get_dummies(acc["cause_mapped"], prefix="cause")
        cause_dummies["grid_id"] = acc["grid_id"].values
        cause_agg = cause_dummies.groupby("grid_id").mean().round(4).reset_index()
        for col in cause_agg.columns:
            if col.startswith("cause_"):
                cause_agg[col] = (cause_agg[col] * 100).round(2)
        # Ensure all expected cause columns exist
        for cause in top_causes:
            col_name = f"cause_{cause}"
            if col_name not in cause_agg.columns:
                cause_agg[col_name] = 0.0
        if "cause_other" not in cause_agg.columns:
            cause_agg["cause_other"] = 0.0
    else:
        cause_agg = pd.DataFrame({"grid_id": list(all_grid_ids)})

    # --- Accident type (fixed columns from training top_types) ---
    if COL_ACCIDENT_TYPE in acc.columns and top_types:
        acc["type_mapped"] = acc[COL_ACCIDENT_TYPE].where(
            acc[COL_ACCIDENT_TYPE].isin(top_types), "other"
        )
        type_dummies = pd.get_dummies(acc["type_mapped"], prefix="type")
        type_dummies["grid_id"] = acc["grid_id"].values
        type_agg = type_dummies.groupby("grid_id").mean().round(4).reset_index()
        for col in type_agg.columns:
            if col.startswith("type_"):
                type_agg[col] = (type_agg[col] * 100).round(2)
        for t in top_types:
            col_name = f"type_{t}"
            if col_name not in type_agg.columns:
                type_agg[col_name] = 0.0
        if "type_other" not in type_agg.columns:
            type_agg["type_other"] = 0.0
    else:
        type_agg = pd.DataFrame({"grid_id": list(all_grid_ids)})

    # Merge all
    result = severity_agg
    for df in [temporal_agg, vehicle_agg, weather_agg, cause_agg, type_agg]:
        result = result.merge(df, on="grid_id", how="outer")

    return result


def build_features_for_period(accidents, all_grid_ids, top_causes, top_types):
    """
    Build the full feature matrix for a set of accidents (one time period).
    Returns (DataFrame with grid_id + features + target, feature_names).
    """
    # 1. Grid counts for this period
    grid_counts = _build_period_grid_stats(accidents, all_grid_ids)

    # 2. Target
    grid_counts["dangerous"] = (
        grid_counts["total_accidents"] >= DANGER_THRESHOLD
    ).astype(int)

    # 3. Neighbor features
    neighbor_feats = _build_neighbor_features(grid_counts)
    ml_df = grid_counts.merge(neighbor_feats, on="grid_id", how="left")

    # 4. Spatial features
    spatial_feats = _build_spatial_features(grid_counts)
    ml_df = ml_df.merge(spatial_feats, on="grid_id", how="left")

    # 5. Accident-level aggregated features
    agg_feats = _build_accident_agg_features(accidents, all_grid_ids, top_causes, top_types)
    ml_df = ml_df.merge(agg_feats, on="grid_id", how="left")

    # Fill NaN (grids with 0 accidents have no aggregated features)
    numeric_cols = ml_df.select_dtypes(include=[np.number]).columns
    ml_df[numeric_cols] = ml_df[numeric_cols].fillna(0)

    # Feature columns = everything except identifiers and target
    exclude = {"grid_id", "center_lat", "center_lon", "total_accidents", "dangerous"}
    feature_names = sorted(c for c in ml_df.columns if c not in exclude)

    return ml_df, feature_names


def prepare_split():
    """
    Main function: load data, split by month, build features, save outputs.
    """
    # Load raw accidents
    accidents = pd.read_csv(ACCIDENTS_PATH, encoding="utf-8-sig")
    accidents["datetime"] = pd.to_datetime(accidents["datetime"], errors="coerce")
    accidents["month"] = accidents["datetime"].dt.month
    LOGGER.info("Loaded %d accident records", len(accidents))

    # Assign grid IDs
    accidents = _assign_grid_ids(accidents)

    # Build full grid ID list from grid_stats
    grid_stats = pd.read_csv(
        OUTPUT_DIR / "bangkok_grid_stats.csv", encoding="utf-8-sig"
    )
    all_grid_ids = sorted(grid_stats["grid_id"].unique())
    LOGGER.info("Grid universe: %d cells", len(all_grid_ids))

    # Keep only accidents in valid grids
    valid_grids = set(all_grid_ids)
    accidents = accidents[accidents["grid_id"].isin(valid_grids)].copy()

    # Temporal split
    train_acc = accidents[accidents["month"].isin(TRAIN_MONTHS)].copy()
    test_acc = accidents[accidents["month"].isin(TEST_MONTHS)].copy()
    LOGGER.info(
        "Split: %d train accidents (months %s), %d test accidents (months %s)",
        len(train_acc), TRAIN_MONTHS, len(test_acc), TEST_MONTHS,
    )

    # Derive top causes/types from TRAINING data only (no leakage)
    top_causes = []
    if COL_CAUSE in train_acc.columns:
        top_causes = train_acc[COL_CAUSE].value_counts().head(5).index.tolist()

    top_types = []
    if COL_ACCIDENT_TYPE in train_acc.columns:
        top_types = train_acc[COL_ACCIDENT_TYPE].value_counts().head(5).index.tolist()

    LOGGER.info("Top causes (from train): %s", top_causes)
    LOGGER.info("Top types (from train): %s", top_types)

    # Build features for each period
    train_df, train_features = build_features_for_period(
        train_acc, all_grid_ids, top_causes, top_types
    )
    test_df, test_features = build_features_for_period(
        test_acc, all_grid_ids, top_causes, top_types
    )

    # Align columns: use training feature set as canonical
    # Add any missing columns to test (as 0), drop any extra test columns
    for col in train_features:
        if col not in test_df.columns:
            test_df[col] = 0.0
    test_features = train_features  # Use same feature order

    # Extract X and y
    X_train = train_df[train_features]
    y_train = train_df[["dangerous"]]
    X_test = test_df[train_features]
    y_test = test_df[["dangerous"]]

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    # Save grid_id mapping for later analysis
    train_df[["grid_id"]].to_csv(OUTPUT_DIR / "train_grid_ids.csv", index=False)
    test_df[["grid_id"]].to_csv(OUTPUT_DIR / "test_grid_ids.csv", index=False)

    # Metadata
    info = {
        "train_months": TRAIN_MONTHS,
        "test_months": TEST_MONTHS,
        "train_accidents": int(len(train_acc)),
        "test_accidents": int(len(test_acc)),
        "train_grids": int(len(X_train)),
        "test_grids": int(len(X_test)),
        "feature_count": len(train_features),
        "feature_names": train_features,
        "danger_threshold": DANGER_THRESHOLD,
        "train_class_distribution": {
            "dangerous": int(y_train["dangerous"].sum()),
            "safe": int(len(y_train) - y_train["dangerous"].sum()),
            "imbalance_ratio": round(
                (len(y_train) - y_train["dangerous"].sum()) / max(y_train["dangerous"].sum(), 1), 2
            ),
        },
        "test_class_distribution": {
            "dangerous": int(y_test["dangerous"].sum()),
            "safe": int(len(y_test) - y_test["dangerous"].sum()),
            "imbalance_ratio": round(
                (len(y_test) - y_test["dangerous"].sum()) / max(y_test["dangerous"].sum(), 1), 2
            ),
        },
        "top_causes_from_train": top_causes,
        "top_types_from_train": top_types,
    }

    with open(OUTPUT_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    return X_train, y_train, X_test, y_test, info


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("TEMPORAL TRAIN/TEST SPLIT FOR ML")
    print("=" * 70)

    X_train, y_train, X_test, y_test, info = prepare_split()

    print(f"\n{'SPLIT SUMMARY':=^70}")
    print(f"Train period: months {info['train_months']} ({info['train_accidents']} accidents)")
    print(f"Test period:  months {info['test_months']} ({info['test_accidents']} accidents)")
    print(f"Grid cells:   {info['train_grids']} (same grid universe for both)")
    print(f"Features:     {info['feature_count']}")

    td = info["train_class_distribution"]
    print(f"\nTrain target distribution:")
    print(f"  Safe (0):      {td['safe']:,}")
    print(f"  Dangerous (1): {td['dangerous']:,}")
    print(f"  Ratio:         {td['imbalance_ratio']}:1")

    sd = info["test_class_distribution"]
    print(f"\nTest target distribution:")
    print(f"  Safe (0):      {sd['safe']:,}")
    print(f"  Dangerous (1): {sd['dangerous']:,}")
    print(f"  Ratio:         {sd['imbalance_ratio']}:1")

    # Leakage check
    print(f"\n{'DATA LEAKAGE CHECK':=^70}")
    train_acc_months = set(info["train_months"])
    test_acc_months = set(info["test_months"])
    overlap = train_acc_months & test_acc_months
    print(f"Train months: {sorted(train_acc_months)}")
    print(f"Test months:  {sorted(test_acc_months)}")
    print(f"Month overlap: {overlap if overlap else 'NONE'}")
    print(f"Leakage check: {'FAIL - OVERLAP DETECTED' if overlap else 'PASS - No leakage'}")

    # Feature alignment check
    train_cols = set(X_train.columns)
    test_cols = set(X_test.columns)
    print(f"\nFeature alignment: {'PASS' if train_cols == test_cols else 'FAIL'}")
    print(f"  Train features: {len(train_cols)}")
    print(f"  Test features:  {len(test_cols)}")
    if train_cols != test_cols:
        print(f"  Only in train: {train_cols - test_cols}")
        print(f"  Only in test:  {test_cols - train_cols}")

    # NaN check
    train_nan = X_train.isna().sum().sum()
    test_nan = X_test.isna().sum().sum()
    print(f"\nNaN check: Train={train_nan}, Test={test_nan}")

    print(f"\n{'FILES SAVED':=^70}")
    for f in ["X_train.csv", "y_train.csv", "X_test.csv", "y_test.csv",
              "split_info.json", "train_grid_ids.csv", "test_grid_ids.csv"]:
        path = OUTPUT_DIR / f
        print(f"  {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
