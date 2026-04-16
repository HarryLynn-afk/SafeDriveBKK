"""
Display-name helpers for zones and roads.

Pure functions — no Flask, no I/O.
"""

import pandas as pd

from ..config import ROUTE_DISPLAY_NAMES
from .geo import nearest_district  # re-exported for callers that only need display helpers


def route_display_name(route_code) -> str | None:
    """Return a human-readable road label from the route code, or None."""
    if pd.isna(route_code):
        return None
    key = str(route_code).strip()
    if not key or key.lower() == "nan":
        return None
    return ROUTE_DISPLAY_NAMES.get(key, f"Route {key}")


def build_zone_display_name(district: str, zone_accidents: pd.DataFrame | None) -> str:
    """Create a user-facing location label for a grid from its dominant corridor."""
    if zone_accidents is None or zone_accidents.empty:
        return f"{district} corridor"

    route_codes = (
        zone_accidents.get("รหัสสายทาง", pd.Series(dtype=object))
        .dropna()
        .astype(str)
        .str.strip()
    )
    route_code = route_codes.mode().iloc[0] if len(route_codes) else None
    road_label = route_display_name(route_code)

    if road_label is None:
        road_names = (
            zone_accidents.get("สายทาง", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .str.strip()
        )
        if len(road_names):
            road_label = road_names.mode().iloc[0]

    km_values = pd.to_numeric(
        zone_accidents.get("KM", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    km_label = f"Km {km_values.median():.1f}" if len(km_values) else None

    if road_label and km_label:
        return f"{road_label} ({km_label})"
    if road_label:
        return road_label
    return f"{district} corridor"
