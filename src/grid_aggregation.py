"""
Spatial grid aggregation for Bangkok accident risk analysis.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Single-sourced from config.py; dual import supports `-m src.grid_aggregation` and `src/grid_aggregation.py`.
try:
    from .config import (
        BKK_LAT_MAX,
        BKK_LAT_MIN,
        BKK_LON_MAX,
        BKK_LON_MIN,
        DEGREES_PER_KM,
    )
except ImportError:
    from config import (
        BKK_LAT_MAX,
        BKK_LAT_MIN,
        BKK_LON_MAX,
        BKK_LON_MIN,
        DEGREES_PER_KM,
    )

LOGGER = logging.getLogger(__name__)

LOW_MAX = 1
MEDIUM_MIN = 2
MEDIUM_MAX = 4
HIGH_MIN = 5


def _classify_risk_levels(total_accidents):
    """Classify risk levels using configured accident-count thresholds."""
    return np.select(
        [
            total_accidents <= LOW_MAX,
            total_accidents.between(MEDIUM_MIN, MEDIUM_MAX),
            total_accidents >= HIGH_MIN,
        ],
        ["low", "medium", "high"],
        default="low",
    )


def create_grid_cells(lat_min, lat_max, lon_min, lon_max, cell_size_km=1.0):
    """
    Create a rectangular grid of ~1km x 1km cells.

    Returns a DataFrame with columns:
    grid_id, center_lat, center_lon, lat_min, lat_max, lon_min, lon_max
    """
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Invalid bounds: min values must be less than max values")
    if cell_size_km <= 0:
        raise ValueError("cell_size_km must be positive")

    step_deg = cell_size_km * DEGREES_PER_KM

    lat_edges = [float(lat_min)]
    while lat_edges[-1] < lat_max:
        lat_edges.append(min(lat_edges[-1] + step_deg, float(lat_max)))

    lon_edges = [float(lon_min)]
    while lon_edges[-1] < lon_max:
        lon_edges.append(min(lon_edges[-1] + step_deg, float(lon_max)))

    cells = []
    for lat_idx in range(len(lat_edges) - 1):
        for lon_idx in range(len(lon_edges) - 1):
            cell_lat_min = lat_edges[lat_idx]
            cell_lat_max = lat_edges[lat_idx + 1]
            cell_lon_min = lon_edges[lon_idx]
            cell_lon_max = lon_edges[lon_idx + 1]
            cells.append(
                {
                    "grid_id": f"G{lat_idx:03d}_{lon_idx:03d}",
                    "center_lat": (cell_lat_min + cell_lat_max) / 2.0,
                    "center_lon": (cell_lon_min + cell_lon_max) / 2.0,
                    "lat_min": cell_lat_min,
                    "lat_max": cell_lat_max,
                    "lon_min": cell_lon_min,
                    "lon_max": cell_lon_max,
                }
            )

    grid_df = pd.DataFrame(cells)
    LOGGER.info(
        "Created %d grid cells (%d latitude bins x %d longitude bins)",
        len(grid_df),
        len(lat_edges) - 1,
        len(lon_edges) - 1,
    )
    return grid_df


def assign_to_grid(accidents_df, grid_df):
    """
    Assign each accident to a grid cell based on LATITUDE/LONGITUDE bounds.

    Returns the original accidents DataFrame with added:
    grid_id, center_lat, center_lon
    """
    required_accident_cols = {"LATITUDE", "LONGITUDE"}
    missing_accident_cols = required_accident_cols - set(accidents_df.columns)
    if missing_accident_cols:
        raise ValueError(f"Missing accident columns: {sorted(missing_accident_cols)}")

    required_grid_cols = {"grid_id", "lat_min", "lat_max", "lon_min", "lon_max", "center_lat", "center_lon"}
    missing_grid_cols = required_grid_cols - set(grid_df.columns)
    if missing_grid_cols:
        raise ValueError(f"Missing grid columns: {sorted(missing_grid_cols)}")

    result = accidents_df.copy()

    lat_starts = np.sort(grid_df["lat_min"].unique())
    lon_starts = np.sort(grid_df["lon_min"].unique())
    lat_edges = np.append(lat_starts, grid_df["lat_max"].max())
    lon_edges = np.append(lon_starts, grid_df["lon_max"].max())

    lat_values = pd.to_numeric(result["LATITUDE"], errors="coerce").to_numpy()
    lon_values = pd.to_numeric(result["LONGITUDE"], errors="coerce").to_numpy()

    lat_idx = np.searchsorted(lat_edges, lat_values, side="right") - 1
    lon_idx = np.searchsorted(lon_edges, lon_values, side="right") - 1

    in_bounds = (
        np.isfinite(lat_values)
        & np.isfinite(lon_values)
        & (lat_values >= lat_edges[0])
        & (lat_values <= lat_edges[-1])
        & (lon_values >= lon_edges[0])
        & (lon_values <= lon_edges[-1])
    )

    lat_idx = np.where(in_bounds, np.clip(lat_idx, 0, len(lat_edges) - 2), -1)
    lon_idx = np.where(in_bounds, np.clip(lon_idx, 0, len(lon_edges) - 2), -1)

    grid_lookup = grid_df.copy()
    lat_index_map = {value: idx for idx, value in enumerate(lat_starts)}
    lon_index_map = {value: idx for idx, value in enumerate(lon_starts)}
    grid_lookup["_lat_idx"] = grid_lookup["lat_min"].map(lat_index_map)
    grid_lookup["_lon_idx"] = grid_lookup["lon_min"].map(lon_index_map)
    grid_lookup = grid_lookup[["_lat_idx", "_lon_idx", "grid_id", "center_lat", "center_lon"]]

    result["_lat_idx"] = lat_idx
    result["_lon_idx"] = lon_idx
    result = result.merge(grid_lookup, on=["_lat_idx", "_lon_idx"], how="left")
    result = result.drop(columns=["_lat_idx", "_lon_idx"])

    unmatched = result["grid_id"].isna().sum()
    if unmatched > 0:
        LOGGER.warning("%d accidents could not be assigned to any grid cell", unmatched)
    else:
        LOGGER.info("All accidents assigned to grid cells")

    return result


def calculate_grid_statistics(grid_accidents_df):
    """
    Calculate accident totals and risk level per grid with assigned accidents.

    Returns columns:
    grid_id, center_lat, center_lon, total_accidents, risk_score, risk_level
    """
    required_cols = {"grid_id", "center_lat", "center_lon"}
    missing_cols = required_cols - set(grid_accidents_df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in grid_accidents_df: {sorted(missing_cols)}")

    assigned = grid_accidents_df.dropna(subset=["grid_id"]).copy()
    if assigned.empty:
        LOGGER.warning("No assigned accidents found; returning empty statistics DataFrame")
        return pd.DataFrame(
            columns=["grid_id", "center_lat", "center_lon", "total_accidents", "risk_score", "risk_level"]
        )

    stats = (
        assigned.groupby(["grid_id", "center_lat", "center_lon"], as_index=False)
        .size()
        .rename(columns={"size": "total_accidents"})
    )

    max_count = stats["total_accidents"].max()
    if max_count > 0:
        stats["risk_score"] = (stats["total_accidents"] / max_count * 100).round(2)
    else:
        stats["risk_score"] = 0.0

    stats["risk_level"] = _classify_risk_levels(stats["total_accidents"])

    stats = stats.sort_values(["total_accidents", "risk_score"], ascending=[False, False]).reset_index(drop=True)
    LOGGER.info("Calculated statistics for %d non-empty grid cells", len(stats))
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    input_path = Path("data/processed/bangkok_accidents_clean.csv")
    output_path = Path("data/processed/bangkok_grid_stats.csv")

    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        accidents = pd.read_csv(input_path, encoding="utf-8-sig")
        LOGGER.info("Loaded %d Bangkok accidents from %s", len(accidents), input_path)

        accidents = accidents.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
        LOGGER.info("Retained %d accidents with valid coordinates", len(accidents))

        grid = create_grid_cells(
            lat_min=BKK_LAT_MIN,
            lat_max=BKK_LAT_MAX,
            lon_min=BKK_LON_MIN,
            lon_max=BKK_LON_MAX,
            cell_size_km=1.0,
        )

        assigned = assign_to_grid(accidents, grid)
        stats_non_empty = calculate_grid_statistics(assigned)

        full_stats = grid[["grid_id", "center_lat", "center_lon"]].merge(
            stats_non_empty,
            on=["grid_id", "center_lat", "center_lon"],
            how="left",
        )
        full_stats["total_accidents"] = full_stats["total_accidents"].fillna(0).astype(int)

        max_count = full_stats["total_accidents"].max()
        if max_count > 0:
            full_stats["risk_score"] = (full_stats["total_accidents"] / max_count * 100).round(2)
        else:
            full_stats["risk_score"] = 0.0

        full_stats["risk_level"] = _classify_risk_levels(full_stats["total_accidents"])

        full_stats = full_stats[
            ["grid_id", "center_lat", "center_lon", "total_accidents", "risk_score", "risk_level"]
        ].sort_values(["total_accidents", "grid_id"], ascending=[False, True])

        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_stats.to_csv(output_path, index=False, encoding="utf-8-sig")

        assigned_count = assigned["grid_id"].notna().sum()
        LOGGER.info("Assigned %d/%d accidents to grid cells", assigned_count, len(assigned))
        LOGGER.info("Saved grid statistics to %s", output_path)
        LOGGER.info("Total grid cells: %d", len(full_stats))
        LOGGER.info("High-risk cells (5+ accidents): %d", (full_stats["risk_level"] == "high").sum())
        LOGGER.info("Medium-risk cells (2-4 accidents): %d", (full_stats["risk_level"] == "medium").sum())
        LOGGER.info("Low-risk cells (0-1 accidents): %d", (full_stats["risk_level"] == "low").sum())

    except Exception as exc:
        LOGGER.exception("Grid aggregation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
