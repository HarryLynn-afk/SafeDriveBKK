"""Re-fetch OSRM routes for all presets in route_presets.json.

Usage:
    python -m scripts.refresh_route_presets
"""

import json
import logging

from src.config import ROUTE_PRESETS_PATH
from src.routing import get_osrm_route, reload_route_presets

logger = logging.getLogger(__name__)


def refresh_presets() -> dict[str, bool]:
    """Re-fetch OSRM routes for all stored preset pairs; overwrite route_presets.json.

    Returns {preset_name: refreshed_successfully} for each entry.
    Partial failures are recorded as False and do not abort the remaining presets.
    """
    try:
        with open(ROUTE_PRESETS_PATH, encoding="utf-8") as f:
            presets = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load route presets from %s: %s", ROUTE_PRESETS_PATH, exc)
        return {}

    results: dict[str, bool] = {}
    for name, preset in presets.items():
        start = preset.get("start", {})
        end = preset.get("end", {})
        if not start or not end:
            results[name] = False
            continue
        try:
            route = get_osrm_route(start["lat"], start["lon"], end["lat"], end["lon"])
            if route:
                preset["waypoints"] = route["waypoints"]
                preset["distance"] = route["distance_km"]
                preset["duration"] = route["duration_min"]
                results[name] = True
            else:
                results[name] = False
        except Exception:
            logger.exception("Failed to refresh preset %r", name)
            results[name] = False

    with open(ROUTE_PRESETS_PATH, "w", encoding="utf-8") as f:
        json.dump(presets, f, indent=2, ensure_ascii=False)

    reload_route_presets()
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = refresh_presets()
    succeeded = sum(v for v in results.values())
    print(f"Refreshed {succeeded}/{len(results)} presets")
    for name, ok in results.items():
        print(f"  {'OK    ' if ok else 'FAILED'}  {name}")
