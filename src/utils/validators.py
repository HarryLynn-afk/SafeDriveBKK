"""
Input validation utilities.

Call these at every API boundary — before passing user-supplied data
into business logic or external services.
"""

from __future__ import annotations

# Global coordinate bounds
_LAT_MIN = -90.0
_LAT_MAX = 90.0
_LON_MIN = -180.0
_LON_MAX = 180.0


def validate_coordinates(lat: float, lon: float) -> None:
    """Raise ValueError if lat/lon are outside valid global ranges.

    This is the minimum check: it rejects physically impossible values
    (e.g. lat=99999) before they reach routing or ML logic.
    """
    if not (_LAT_MIN <= lat <= _LAT_MAX):
        raise ValueError(f"Latitude must be between {_LAT_MIN} and {_LAT_MAX}, got {lat}")
    if not (_LON_MIN <= lon <= _LON_MAX):
        raise ValueError(f"Longitude must be between {_LON_MIN} and {_LON_MAX}, got {lon}")


def validate_coordinate_pair_dict(payload: dict, lat_key: str, lon_key: str) -> tuple[float, float]:
    """Parse and validate a lat/lon pair from a request payload dict.

    Raises ValueError with a descriptive message on any failure.
    Returns (lat, lon) as floats.
    """
    missing = [k for k in (lat_key, lon_key) if k not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    try:
        lat = float(payload[lat_key])
        lon = float(payload[lon_key])
    except (TypeError, ValueError):
        raise ValueError(f"'{lat_key}' and '{lon_key}' must be valid numbers")
    validate_coordinates(lat, lon)
    return lat, lon
