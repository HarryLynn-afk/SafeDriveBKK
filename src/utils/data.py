"""
Data-loading utilities.

Centralises CSV reads and provides a module-level cache for the grid-stats
DataFrame so it is read only once per process.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import BANGKOK_ACCIDENTS_PATH, DAY_NAME_MAP, GRID_STATS_PATH
from .display import build_zone_display_name
from .geo import add_grid_bounds, assign_bangkok_area, latlon_to_grid_id, nearest_district

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

_VEHICLE_COLUMNS = {
    "Motorcycle": "รถจักรยานยนต์",
    "Car": "รถยนต์นั่งส่วนบุคคล",
    "Pickup": "รถปิคอัพบรรทุก4ล้อ",
    "Van": "รถตู้",
    "Truck 6W": "รถบรรทุก6ล้อ",
    "Truck 10W+": "รถบรรทุกมากกว่า10ล้อ",
    "Bus": "รถโดยสารมากกว่า4ล้อ",
    "Pedestrian": "คนเดินเท้า",
}

_SEVERITY_COLUMNS = {
    "Deaths": "ผู้เสียชีวิต",
    "Serious Injuries": "ผู้บาดเจ็บสาหัส",
    "Minor Injuries": "ผู้บาดเจ็บเล็กน้อย",
}

# ---------------------------------------------------------------------------
# Module-level cache (set once, reused across requests)
# ---------------------------------------------------------------------------
_grid_stats_cache: pd.DataFrame | None = None
_accidents_cache: pd.DataFrame | None = None


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
    global _accidents_cache
    if _accidents_cache is not None:
        return _accidents_cache
    _require_file(BANGKOK_ACCIDENTS_PATH)
    _accidents_cache = pd.read_csv(BANGKOK_ACCIDENTS_PATH, encoding="utf-8-sig")
    return _accidents_cache


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


# ---------------------------------------------------------------------------
# Analytics payload (composition of the helpers below)
# ---------------------------------------------------------------------------

def _aggregate_temporal_breakdowns(df: pd.DataFrame) -> dict:
    hour_counts = pd.to_numeric(df["hour"], errors="coerce").dropna().astype(int).value_counts().sort_index()
    by_hour = {int(h): int(c) for h, c in hour_counts.items()}

    dow_counts = pd.to_numeric(df["day_of_week"], errors="coerce").dropna().astype(int).value_counts().sort_index()
    by_day = {DAY_NAME_MAP.get(int(d), str(d)): int(c) for d, c in dow_counts.items()}

    month_counts = pd.to_numeric(df["month"], errors="coerce").dropna().astype(int).value_counts().sort_index()
    by_month = {_MONTH_NAMES.get(int(m), str(m)): int(c) for m, c in month_counts.items()}

    return {"by_hour": by_hour, "by_day": by_day, "by_month": by_month}


def _build_zone_context_lookup(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if not {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        return {}
    ctx = df[["LATITUDE", "LONGITUDE", "รหัสสายทาง", "สายทาง", "KM"]].copy()
    ctx["_lat"] = pd.to_numeric(ctx["LATITUDE"], errors="coerce")
    ctx["_lon"] = pd.to_numeric(ctx["LONGITUDE"], errors="coerce")
    ctx = ctx.dropna(subset=["_lat", "_lon"])
    ctx["grid_id"] = ctx.apply(lambda r: latlon_to_grid_id(r["_lat"], r["_lon"]), axis=1)
    ctx = ctx.dropna(subset=["grid_id"])
    return {gid: grp.copy() for gid, grp in ctx.groupby("grid_id", sort=False)}


def _aggregate_top_zones(df: pd.DataFrame, grid_df: pd.DataFrame, n: int = 10) -> list[dict]:
    zone_context = _build_zone_context_lookup(df)
    top_zones = (
        grid_df[grid_df["total_accidents"] > 0]
        .nlargest(n, "total_accidents")[["grid_id", "total_accidents", "risk_level", "center_lat", "center_lon"]]
        .to_dict(orient="records")
    )
    for zone in top_zones:
        zone["district"] = nearest_district(zone["center_lat"], zone["center_lon"])
        zone["zone_name"] = build_zone_display_name(zone["district"], zone_context.get(zone["grid_id"]))
    return top_zones


def _aggregate_by_area(df: pd.DataFrame) -> dict[str, int]:
    if "LATITUDE" not in df.columns or "LONGITUDE" not in df.columns:
        return {}
    areas = df.apply(lambda r: assign_bangkok_area(r["LATITUDE"], r["LONGITUDE"]), axis=1)
    return {str(k): int(v) for k, v in areas.dropna().value_counts().items()}


def _aggregate_vehicle_types(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, col in _VEHICLE_COLUMNS.items():
        if col not in df.columns:
            continue
        total = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
        if total > 0:
            counts[label] = total
    return counts


def _aggregate_severity(df: pd.DataFrame) -> dict[str, int]:
    return {
        label: int(pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        for label, col in _SEVERITY_COLUMNS.items()
    }


def _aggregate_risk_distribution(grid_df: pd.DataFrame) -> dict:
    risk_level_series = grid_df["risk_level"].astype(str).str.lower()
    active_mask = grid_df["total_accidents"] > 0

    distribution = {str(k): int(v) for k, v in risk_level_series[active_mask].value_counts().items()}
    active_zones = int(active_mask.sum())
    high_risk_zones = int((risk_level_series == "high").sum())
    total_grids = int(len(grid_df))
    total_mapped = int(pd.to_numeric(grid_df["total_accidents"], errors="coerce").fillna(0).sum())
    high_risk_accidents = int(
        pd.to_numeric(grid_df.loc[risk_level_series == "high", "total_accidents"], errors="coerce").fillna(0).sum()
    )

    summary = {
        "active_zones": active_zones,
        "high_risk_zones": high_risk_zones,
        "mapped_accidents": total_mapped,
        "high_risk_accidents": high_risk_accidents,
        "high_risk_grid_share_percent": round((high_risk_zones / total_grids) * 100, 1) if total_grids else 0.0,
        "high_risk_accident_share": round((high_risk_accidents / total_mapped) * 100, 1) if total_mapped else 0.0,
        "high_risk_accident_share_exact": ((high_risk_accidents / total_mapped) * 100) if total_mapped else 0.0,
    }

    return {"distribution": distribution, "total_grids": total_grids, "summary": summary}


def _aggregate_weekend_split(df: pd.DataFrame) -> dict[str, int]:
    weekend_col = df.get("is_weekend")
    if weekend_col is None:
        return {"Weekend": 0, "Weekday": 0}
    if weekend_col.dtype == bool:
        mask = weekend_col.fillna(False)
    else:
        mask = weekend_col.astype(str).str.lower().isin({"true", "1", "yes"})
    weekend_count = int(mask.sum())
    return {"Weekend": weekend_count, "Weekday": int(len(df) - weekend_count)}


def build_analytics_payload(accidents_df: pd.DataFrame, grid_df: pd.DataFrame) -> dict:
    """Build the full analytics response dict from the accidents + grid frames.

    Returns the same shape the /api/analytics-data endpoint serializes to JSON.
    Composed of small private aggregators above; the blueprint sees one entry
    point and no longer knows about Thai column names or grid math.
    """
    temporal = _aggregate_temporal_breakdowns(accidents_df)
    risk_dist = _aggregate_risk_distribution(grid_df)

    return {
        "total_accidents": int(len(accidents_df)),
        "total_grids": risk_dist["total_grids"],
        "by_hour": temporal["by_hour"],
        "by_day": temporal["by_day"],
        "by_month": temporal["by_month"],
        "top_zones": _aggregate_top_zones(accidents_df, grid_df),
        "by_area": _aggregate_by_area(accidents_df),
        "by_vehicle": _aggregate_vehicle_types(accidents_df),
        "severity": _aggregate_severity(accidents_df),
        "risk_distribution": risk_dist["distribution"],
        "weekend_vs_weekday": _aggregate_weekend_split(accidents_df),
        "summary": risk_dist["summary"],
    }
