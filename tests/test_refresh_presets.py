"""
Tests for scripts/refresh_route_presets.refresh_presets

Interface boundary: refresh_presets() -> dict[str, bool]

Covers: successful OSRM refresh, OSRM failure per preset, partial failures,
empty presets, missing/corrupt file.
"""

import json
from unittest.mock import patch

import pytest

import src.routing as routing_mod
from scripts.refresh_route_presets import refresh_presets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_presets_cache():
    routing_mod._route_presets_cache = None
    yield
    routing_mod._route_presets_cache = None


_PRESET_A = {
    "start": {"lat": 13.75, "lon": 100.50},
    "end":   {"lat": 13.80, "lon": 100.55},
    "waypoints": [[13.75, 100.50], [13.80, 100.55]],
    "distance": 5.0,
    "duration": 10,
}

_PRESET_B = {
    "start": {"lat": 13.70, "lon": 100.45},
    "end":   {"lat": 13.75, "lon": 100.50},
    "waypoints": [[13.70, 100.45], [13.75, 100.50]],
    "distance": 4.0,
    "duration": 8,
}

_FRESH_ROUTE = {
    "waypoints": [[13.76, 100.51], [13.78, 100.53], [13.80, 100.55]],
    "distance_km": 6.1,
    "duration_min": 13,
}


def _write_presets(path, content):
    path.write_text(json.dumps(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# TestRefreshPresets
# ---------------------------------------------------------------------------

class TestRefreshPresets:

    # ---- successful refresh ----

    def test_returns_true_for_successfully_refreshed_preset(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {"route_a": _PRESET_A})
        with (
            patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file),
            patch("scripts.refresh_route_presets.get_osrm_route", return_value=_FRESH_ROUTE),
        ):
            result = refresh_presets()
        assert result == {"route_a": True}

    def test_writes_fresh_waypoints_to_file(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {"route_a": _PRESET_A})
        with (
            patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file),
            patch("scripts.refresh_route_presets.get_osrm_route", return_value=_FRESH_ROUTE),
        ):
            refresh_presets()
        updated = json.loads(presets_file.read_text())
        assert updated["route_a"]["waypoints"] == _FRESH_ROUTE["waypoints"]

    def test_writes_fresh_distance_and_duration_to_file(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {"route_a": _PRESET_A})
        with (
            patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file),
            patch("scripts.refresh_route_presets.get_osrm_route", return_value=_FRESH_ROUTE),
        ):
            refresh_presets()
        updated = json.loads(presets_file.read_text())
        assert updated["route_a"]["distance"] == _FRESH_ROUTE["distance_km"]
        assert updated["route_a"]["duration"] == _FRESH_ROUTE["duration_min"]

    # ---- OSRM failure ----

    def test_returns_false_when_osrm_returns_none(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {"route_a": _PRESET_A})
        with (
            patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file),
            patch("scripts.refresh_route_presets.get_osrm_route", return_value=None),
        ):
            result = refresh_presets()
        assert result == {"route_a": False}

    def test_failed_preset_keeps_original_waypoints_in_file(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {"route_a": _PRESET_A})
        original_wps = _PRESET_A["waypoints"]
        with (
            patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file),
            patch("scripts.refresh_route_presets.get_osrm_route", return_value=None),
        ):
            refresh_presets()
        updated = json.loads(presets_file.read_text())
        assert updated["route_a"]["waypoints"] == original_wps

    # ---- partial failure ----

    def test_partial_failure_returns_correct_per_preset_status(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {"route_a": _PRESET_A, "route_b": _PRESET_B})

        def osrm_side_effect(slat, slon, elat, elon):
            return _FRESH_ROUTE if slat == 13.70 else None  # route_b succeeds, route_a fails

        with (
            patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file),
            patch("scripts.refresh_route_presets.get_osrm_route", side_effect=osrm_side_effect),
        ):
            result = refresh_presets()

        assert result["route_a"] is False
        assert result["route_b"] is True

    def test_partial_failure_still_writes_successful_presets(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {"route_a": _PRESET_A, "route_b": _PRESET_B})

        def osrm_side_effect(slat, slon, elat, elon):
            return _FRESH_ROUTE if slat == 13.70 else None

        with (
            patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file),
            patch("scripts.refresh_route_presets.get_osrm_route", side_effect=osrm_side_effect),
        ):
            refresh_presets()

        updated = json.loads(presets_file.read_text())
        assert updated["route_b"]["waypoints"] == _FRESH_ROUTE["waypoints"]

    # ---- edge cases ----

    def test_empty_presets_returns_empty_dict(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        _write_presets(presets_file, {})
        with patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file):
            result = refresh_presets()
        assert result == {}

    def test_missing_presets_file_returns_empty_dict(self, tmp_path):
        with patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", tmp_path / "missing.json"):
            result = refresh_presets()
        assert result == {}

    def test_corrupt_presets_file_returns_empty_dict(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        presets_file.write_text("{ not valid {{", encoding="utf-8")
        with patch("scripts.refresh_route_presets.ROUTE_PRESETS_PATH", presets_file):
            result = refresh_presets()
        assert result == {}
