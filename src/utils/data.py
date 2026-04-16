"""
Data-loading utilities.

Centralises CSV reads and provides a module-level cache for the grid-stats
DataFrame so it is read only once per process.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import BANGKOK_ACCIDENTS_PATH, DAY_NAME_MAP, GRID_STATS_PATH
from .geo import add_grid_bounds

# ---------------------------------------------------------------------------
# Module-level cache (set once, reused across requests)
# ---------------------------------------------------------------------------
_grid_stats_cache: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_file(file_path: Path) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_bangkok_accidents() -> pd.DataFrame:
    """Load and return the cleaned Bangkok accident records."""
    _require_file(BANGKOK_ACCIDENTS_PATH)
    return pd.read_csv(BANGKOK_ACCIDENTS_PATH, encoding="utf-8-sig")


def load_grid_stats() -> pd.DataFrame:
    """Load the Bangkok grid-stats CSV, with result cached in memory."""
    global _grid_stats_cache
    if _grid_stats_cache is not None:
        return _grid_stats_cache
    _require_file(GRID_STATS_PATH)
    df = pd.read_csv(GRID_STATS_PATH, encoding="utf-8-sig")
    if "grid_id" not in df.columns:
        raise ValueError("Invalid grid stats file: missing 'grid_id' column")
    _grid_stats_cache = add_grid_bounds(df)
    return _grid_stats_cache


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def format_hour_range(hour: int) -> str:
    start = int(hour)
    end = (start + 1) % 24
    return f"{start:02d}:00-{end:02d}:00"


def calculate_temporal_stats(df: pd.DataFrame) -> dict:
    """Derive peak-hour, dangerous-day, and rush-hour stats from the accidents DataFrame."""
    hour_counts = (
        pd.to_numeric(df["hour"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_values(ascending=False)
    )
    peak_hours = [
        {"hour": int(h), "time_range": format_hour_range(h), "count": int(c)}
        for h, c in hour_counts.head(3).items()
    ]

    dow_counts = (
        pd.to_numeric(df["day_of_week"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_values(ascending=False)
    )
    dangerous_days = [
        {"day_index": int(d), "day_name": DAY_NAME_MAP.get(int(d), str(d)), "count": int(c)}
        for d, c in dow_counts.head(3).items()
    ]

    rush_col = df["is_rush_hour"]
    if rush_col.dtype == bool:
        rush_mask = rush_col.fillna(False)
    else:
        rush_mask = rush_col.astype(str).str.lower().isin({"true", "1", "yes", "y", "t"})

    rush_count = int(rush_mask.sum())
    non_rush_count = int(len(df) - rush_count)
    rush_pct = round((rush_count / len(df) * 100), 2) if len(df) else 0.0

    return {
        "peak_hours": peak_hours,
        "dangerous_days": dangerous_days,
        "rush_hour_count": rush_count,
        "non_rush_count": non_rush_count,
        "rush_hour_percentage": rush_pct,
    }
