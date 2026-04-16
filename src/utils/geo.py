"""
Spatial / geographic utilities.

All pure functions — no Flask, no I/O, no side-effects.
"""

import math

import pandas as pd

from ..config import (
    BKK_DISTRICTS,
    BKK_LAT_MAX,
    BKK_LAT_MIN,
    BKK_LON_MAX,
    BKK_LON_MIN,
    CELL_SIZE_DEG,
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two points."""
    radius_km = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    )
    return 2 * radius_km * math.asin(math.sqrt(a))


def clip_to_bangkok_bounds(lat: float, lon: float) -> tuple[float, float]:
    """Clamp a coordinate to the Bangkok bounding box."""
    return (
        min(max(lat, BKK_LAT_MIN), BKK_LAT_MAX),
        min(max(lon, BKK_LON_MIN), BKK_LON_MAX),
    )


def latlon_to_grid_id(lat: float, lon: float) -> str | None:
    """Convert lat/lon to the matching grid_id string, or None if outside Bangkok."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    if not (BKK_LAT_MIN <= lat <= BKK_LAT_MAX and BKK_LON_MIN <= lon <= BKK_LON_MAX):
        return None

    lat_idx = int((lat - BKK_LAT_MIN) / CELL_SIZE_DEG)
    lon_idx = int((lon - BKK_LON_MIN) / CELL_SIZE_DEG)

    max_lat_idx = int((BKK_LAT_MAX - BKK_LAT_MIN) / CELL_SIZE_DEG) - 1
    max_lon_idx = int((BKK_LON_MAX - BKK_LON_MIN) / CELL_SIZE_DEG) - 1
    lat_idx = min(lat_idx, max_lat_idx)
    lon_idx = min(lon_idx, max_lon_idx)
    return f"G{lat_idx:03d}_{lon_idx:03d}"


def nearest_district(lat: float, lon: float) -> str:
    """Return the Bangkok district name whose centroid is closest to (lat, lon)."""
    best_name = "Bangkok"
    best_dist = float("inf")
    for name, dlat, dlon in BKK_DISTRICTS:
        d = (lat - dlat) ** 2 + (lon - dlon) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def find_grid_for_point(lat: float, lon: float, grid_df: pd.DataFrame) -> pd.Series | None:
    """Return the grid row whose bounds contain (lat, lon), or None."""
    target = grid_df[
        (grid_df["lat_min"] <= lat)
        & (lat <= grid_df["lat_max"])
        & (grid_df["lon_min"] <= lon)
        & (lon <= grid_df["lon_max"])
    ]
    return target.iloc[0] if not target.empty else None


def add_grid_bounds(grid_df: pd.DataFrame) -> pd.DataFrame:
    """Derive lat/lon bounding columns from the grid_id if they are missing."""
    required = {"lat_min", "lat_max", "lon_min", "lon_max"}
    if required.issubset(grid_df.columns):
        return grid_df.copy()

    parsed = grid_df["grid_id"].astype(str).str.extract(r"^G(\d{3})_(\d{3})$")
    if parsed.isna().any().any():
        raise ValueError("Invalid grid_id format; expected G000_000 style values")

    lat_idx = parsed[0].astype(int)
    lon_idx = parsed[1].astype(int)

    result = grid_df.copy()
    result["lat_min"] = BKK_LAT_MIN + (lat_idx * CELL_SIZE_DEG)
    result["lat_max"] = (result["lat_min"] + CELL_SIZE_DEG).clip(upper=BKK_LAT_MAX)
    result["lon_min"] = BKK_LON_MIN + (lon_idx * CELL_SIZE_DEG)
    result["lon_max"] = (result["lon_min"] + CELL_SIZE_DEG).clip(upper=BKK_LON_MAX)
    return result


def assign_bangkok_area(lat: float, lon: float) -> str | None:
    """Map a coordinate to a coarse Bangkok area label for analytics grouping."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    if lat >= 13.85:
        return "Don Mueang / Lak Si" if lon < 100.55 else "Ram Intra / Minburi"
    if lat >= 13.78:
        if lon < 100.5:
            return "Bang Sue / Chatuchak"
        return "Lat Phrao / Wang Thonglang" if lon < 100.6 else "Khan Na Yao / Nong Chok"
    if lat >= 13.72:
        if lon < 100.48:
            return "Dusit / Phaya Thai"
        if lon < 100.56:
            return "Din Daeng / Huai Khwang"
        return "Bang Kapi / Suan Luang" if lon < 100.65 else "Lat Krabang"
    if lat >= 13.66:
        if lon < 100.48:
            return "Thon Buri / Bangkok Noi"
        if lon < 100.55:
            return "Sathon / Bang Rak"
        return "Phra Khanong / Bang Na" if lon < 100.65 else "Prawet / On Nut"
    return "Bang Khun Thian / Chom Thong" if lon < 100.55 else "Samut Prakan Border"
