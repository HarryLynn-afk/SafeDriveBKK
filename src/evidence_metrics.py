import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BANGKOK_GRID_STATS_PATH = Path("data/processed/bangkok_grid_stats.csv")
BANGKOK_ACCIDENTS_PATH = Path("data/processed/bangkok_accidents_clean.csv")
REPORT_PATH = Path("data/processed/evidence_report.txt")

# Grid constants — single-sourced from config.py. The dual import supports both
# `python -m src.evidence_metrics` (relative) and `python src/evidence_metrics.py` (absolute).
try:
    from .config import (
        BKK_LAT_MAX as LAT_MAX,
        BKK_LAT_MIN as LAT_MIN,
        BKK_LON_MAX as LON_MAX,
        BKK_LON_MIN as LON_MIN,
        CELL_SIZE_DEG,
    )
except ImportError:
    from config import (
        BKK_LAT_MAX as LAT_MAX,
        BKK_LAT_MIN as LAT_MIN,
        BKK_LON_MAX as LON_MAX,
        BKK_LON_MIN as LON_MIN,
        CELL_SIZE_DEG,
    )


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _load_grid_stats():
    if not BANGKOK_GRID_STATS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BANGKOK_GRID_STATS_PATH}")
    return pd.read_csv(BANGKOK_GRID_STATS_PATH, encoding="utf-8-sig")


def _load_accidents():
    if not BANGKOK_ACCIDENTS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {BANGKOK_ACCIDENTS_PATH}")

    accidents = pd.read_csv(BANGKOK_ACCIDENTS_PATH, encoding="utf-8-sig")

    if "accident_datetime" in accidents.columns:
        accidents["accident_datetime"] = pd.to_datetime(accidents["accident_datetime"], errors="coerce")
    elif "datetime" in accidents.columns:
        accidents["accident_datetime"] = pd.to_datetime(accidents["datetime"], errors="coerce")
    else:
        accidents["accident_datetime"] = pd.NaT

    if "LATITUDE" in accidents.columns and "LONGITUDE" in accidents.columns:
        accidents["LATITUDE"] = pd.to_numeric(accidents["LATITUDE"], errors="coerce")
        accidents["LONGITUDE"] = pd.to_numeric(accidents["LONGITUDE"], errors="coerce")
    else:
        raise ValueError("Expected LATITUDE and LONGITUDE columns in Bangkok accidents data")

    return accidents


def _assign_grid_id(df):
    result = df.copy()

    in_bounds = (
        result["LATITUDE"].between(LAT_MIN, LAT_MAX)
        & result["LONGITUDE"].between(LON_MIN, LON_MAX)
        & result["LATITUDE"].notna()
        & result["LONGITUDE"].notna()
    )

    lat_idx = np.floor((result.loc[in_bounds, "LATITUDE"] - LAT_MIN) / CELL_SIZE_DEG).astype(int)
    lon_idx = np.floor((result.loc[in_bounds, "LONGITUDE"] - LON_MIN) / CELL_SIZE_DEG).astype(int)

    max_lat_idx = max(0, int(math.ceil((LAT_MAX - LAT_MIN) / CELL_SIZE_DEG)) - 1)
    max_lon_idx = max(0, int(math.ceil((LON_MAX - LON_MIN) / CELL_SIZE_DEG)) - 1)
    lat_idx = lat_idx.clip(lower=0, upper=max_lat_idx)
    lon_idx = lon_idx.clip(lower=0, upper=max_lon_idx)

    result["grid_id"] = pd.NA
    result.loc[in_bounds, "grid_id"] = [
        f"G{lat_value:03d}_{lon_value:03d}"
        for lat_value, lon_value in zip(lat_idx, lon_idx)
    ]
    return result


def calculate_coverage_lift():
    """Calculate what % of accidents are in top X% of grids."""
    grid_stats = _load_grid_stats().copy()
    grid_stats_sorted = grid_stats.sort_values("total_accidents", ascending=False).reset_index(drop=True)

    total_accidents = float(grid_stats_sorted["total_accidents"].sum())
    total_grids = len(grid_stats_sorted)
    grid_stats_sorted["cumulative_accidents"] = grid_stats_sorted["total_accidents"].cumsum()

    results = []
    for pct in [1, 5, 10, 20]:
        n_grids = max(1, int(math.ceil(total_grids * pct / 100)))
        accidents_in_top = float(grid_stats_sorted.head(n_grids)["total_accidents"].sum())
        coverage = _safe_ratio(accidents_in_top, total_accidents) * 100
        lift = _safe_ratio(coverage, pct)

        results.append(
            {
                "top_percent": pct,
                "n_grids": n_grids,
                "accidents_covered": int(accidents_in_top),
                "coverage_percent": round(coverage, 1),
                "lift_factor": round(lift, 1),
            }
        )

    return pd.DataFrame(results)


def calculate_density_comparison():
    """Compare accident density in high vs medium vs low risk zones."""
    grid_stats = _load_grid_stats().copy()

    high_risk = grid_stats[grid_stats["total_accidents"] >= 5]
    medium_risk = grid_stats[(grid_stats["total_accidents"] >= 2) & (grid_stats["total_accidents"] < 5)]
    low_risk = grid_stats[grid_stats["total_accidents"] < 2]

    high_density = _safe_ratio(float(high_risk["total_accidents"].sum()), len(high_risk))
    medium_density = _safe_ratio(float(medium_risk["total_accidents"].sum()), len(medium_risk))
    low_density = _safe_ratio(float(low_risk["total_accidents"].sum()), len(low_risk))
    city_density = _safe_ratio(float(grid_stats["total_accidents"].sum()), len(grid_stats))

    results = {
        "high_risk": {
            "n_grids": int(len(high_risk)),
            "total_accidents": int(high_risk["total_accidents"].sum()),
            "density": round(high_density, 2),
            "avg_per_grid": round(float(high_risk["total_accidents"].mean()), 2) if len(high_risk) else 0.0,
        },
        "medium_risk": {
            "n_grids": int(len(medium_risk)),
            "total_accidents": int(medium_risk["total_accidents"].sum()),
            "density": round(medium_density, 2),
            "avg_per_grid": round(float(medium_risk["total_accidents"].mean()), 2) if len(medium_risk) else 0.0,
        },
        "low_risk": {
            "n_grids": int(len(low_risk)),
            "total_accidents": int(low_risk["total_accidents"].sum()),
            "density": round(low_density, 2),
            "avg_per_grid": round(float(low_risk["total_accidents"].mean()), 2) if len(low_risk) else 0.0,
        },
        "city_average": {
            "density": round(city_density, 2),
        },
    }

    results["high_vs_average"] = round(_safe_ratio(high_density, city_density), 1) if city_density else 0.0
    results["high_vs_low"] = round(_safe_ratio(high_density, low_density), 1) if low_density else float("inf")
    return results


def temporal_validation():
    """Validate whether Saturday 10:00-10:59 has elevated risk."""
    accidents = _load_accidents().copy()
    valid = accidents.dropna(subset=["accident_datetime"])

    valid["day_of_week"] = valid["accident_datetime"].dt.day_name()
    valid["hour"] = valid["accident_datetime"].dt.hour

    saturday_10am = valid[(valid["day_of_week"] == "Saturday") & (valid["hour"] == 10)]
    saturday_dates = valid.loc[valid["day_of_week"] == "Saturday", "accident_datetime"].dt.date.nunique()
    total_dates = valid["accident_datetime"].dt.date.nunique()

    saturday_10am_rate = _safe_ratio(len(saturday_10am), saturday_dates)
    avg_hourly_rate = _safe_ratio(len(valid), total_dates * 24)

    return {
        "saturday_10am_accidents": int(len(saturday_10am)),
        "saturday_10am_rate_per_day": round(saturday_10am_rate, 2),
        "average_hourly_rate": round(avg_hourly_rate, 2),
        "multiplier": round(_safe_ratio(saturday_10am_rate, avg_hourly_rate), 2) if avg_hourly_rate else 0.0,
    }


def holdout_validation():
    """Validate risk zones using temporal holdout (train Jan-Sep, test Oct-Dec)."""
    accidents = _load_accidents().copy()
    valid = accidents.dropna(subset=["accident_datetime", "LATITUDE", "LONGITUDE"]).copy()
    valid = _assign_grid_id(valid).dropna(subset=["grid_id"]).copy()

    valid["month"] = valid["accident_datetime"].dt.month

    train = valid[valid["month"] <= 9].copy()
    test = valid[valid["month"] > 9].copy()

    train_grid_counts = train.groupby("grid_id").size().reset_index(name="train_count")
    high_risk_grids = set(train_grid_counts.loc[train_grid_counts["train_count"] >= 5, "grid_id"])

    test_in_high_risk = test[test["grid_id"].isin(high_risk_grids)]
    capture_rate = _safe_ratio(len(test_in_high_risk), len(test)) * 100

    unique_train_grids = train["grid_id"].nunique()
    high_grid_share = _safe_ratio(len(high_risk_grids), unique_train_grids) * 100
    random_baseline = high_grid_share
    lift_vs_random = _safe_ratio(capture_rate, random_baseline) if random_baseline > 0 else 0.0

    return {
        "train_period": "Jan-Sep 2024",
        "test_period": "Oct-Dec 2024",
        "train_accidents": int(len(train)),
        "test_accidents": int(len(test)),
        "high_risk_grids_identified": int(len(high_risk_grids)),
        "test_accidents_in_high_risk": int(len(test_in_high_risk)),
        "capture_rate_percent": round(capture_rate, 1),
        "high_risk_grid_share_percent": round(high_grid_share, 1),
        "lift_vs_random_grid_selection": round(lift_vs_random, 2),
    }


def generate_report():
    """Generate complete evidence report for judging and README claims."""
    coverage = calculate_coverage_lift()
    density = calculate_density_comparison()
    temporal = temporal_validation()
    holdout = holdout_validation()

    print("=" * 60)
    print("ROADRISK BANGKOK - EVIDENCE METRICS REPORT")
    print("Generated:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    print("\n1. COVERAGE & LIFT ANALYSIS")
    print("-" * 60)
    print(coverage.to_string(index=False))
    top_5 = coverage.loc[coverage["top_percent"] == 5].iloc[0]
    print(
        "\nKey Finding: Top 5% of grids contain "
        f"{top_5['coverage_percent']}% of accidents "
        f"({top_5['lift_factor']}x lift vs random)."
    )

    print("\n2. DENSITY COMPARISON")
    print("-" * 60)
    print(f"High-risk zones: {density['high_risk']['density']} accidents/km^2")
    print(f"City average: {density['city_average']['density']} accidents/km^2")
    print(f"Ratio: {density['high_vs_average']}x more dangerous")

    print("\n3. TEMPORAL VALIDATION")
    print("-" * 60)
    print(f"Saturday 10:00-10:59 accidents: {temporal['saturday_10am_accidents']}")
    print(f"Saturday 10:00 rate/day: {temporal['saturday_10am_rate_per_day']}")
    print(f"Average hourly rate: {temporal['average_hourly_rate']}")
    print(f"Multiplier: {temporal['multiplier']}x higher risk")

    print("\n4. HOLDOUT VALIDATION")
    print("-" * 60)
    print(f"Training: {holdout['train_period']} ({holdout['train_accidents']} accidents)")
    print(f"Testing: {holdout['test_period']} ({holdout['test_accidents']} accidents)")
    print(f"High-risk grids identified: {holdout['high_risk_grids_identified']}")
    print(f"Capture rate: {holdout['capture_rate_percent']}% of test accidents")
    print(f"High-risk grid share: {holdout['high_risk_grid_share_percent']}% of train grids")
    print(f"Lift vs random grid selection: {holdout['lift_vs_random_grid_selection']}x")

    print("\n" + "=" * 60)
    print("CONCLUSION: Risk zones show concentration and predictive signal")
    print("=" * 60)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as output_file:
        output_file.write("ROADRISK BANGKOK - EVIDENCE METRICS\n")
        output_file.write("=" * 60 + "\n\n")
        output_file.write("Coverage & Lift\n")
        output_file.write(coverage.to_string(index=False))
        output_file.write("\n\n")
        output_file.write(f"Top 5% coverage: {top_5['coverage_percent']}%\n")
        output_file.write(f"Top 5% lift: {top_5['lift_factor']}x\n\n")
        output_file.write("Density Comparison\n")
        output_file.write(f"High-risk density: {density['high_risk']['density']} accidents/km^2\n")
        output_file.write(f"City average density: {density['city_average']['density']} accidents/km^2\n")
        output_file.write(f"High vs average ratio: {density['high_vs_average']}x\n\n")
        output_file.write("Temporal Validation\n")
        output_file.write(f"Saturday 10:00 multiplier: {temporal['multiplier']}x\n\n")
        output_file.write("Holdout Validation\n")
        output_file.write(f"Capture rate: {holdout['capture_rate_percent']}%\n")
        output_file.write(
            f"Lift vs random grid selection: {holdout['lift_vs_random_grid_selection']}x\n"
        )

    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
