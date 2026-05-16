"""
Tests for src/routing.py

Covers: dense interpolation shape and bounds, cached-route lookup logic,
waypoint sampling, and the OSRM/Google Maps fallback chain (using mocks so
tests run offline without network calls).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.routing import (
    ResolvedRoute,
    find_cached_route,
    generate_dense_interpolation,
    generate_fast_route,
    get_osrm_alternatives,
    resolve_route,
)
from src.config import BKK_LAT_MIN, BKK_LAT_MAX, BKK_LON_MIN, BKK_LON_MAX


# ---------------------------------------------------------------------------
# resolve_route — unified resolver
# ---------------------------------------------------------------------------

_START = (13.75, 100.50)
_END   = (13.80, 100.55)

_OSRM_RESPONSE = {
    "waypoints":   [[13.75, 100.50], [13.775, 100.525], [13.80, 100.55]],
    "distance_km": 7.2,
    "duration_min": 14,
}
_CACHED_RESPONSE = {
    "waypoints":   [[13.75, 100.50], [13.770, 100.520], [13.80, 100.55]],
    "distance_km": 7.5,
    "duration_min": 15,
}
_GMAPS_RESPONSE = {
    "duration_minutes": 12,
    "distance_km":      7.0,
    "has_traffic":      True,
}


class TestResolveRoute:
    def _no_external(self, **overrides):
        """Helper: patch all external calls to return nothing by default."""
        defaults = {
            "src.routing.get_osrm_route":              None,
            "src.routing.find_cached_route":           None,
            "src.routing.get_google_maps_travel_time": None,
        }
        defaults.update(overrides)
        return defaults

    def _resolve(self, patches: dict, client_waypoints=None) -> ResolvedRoute:
        ctx = [patch(target, return_value=value) for target, value in patches.items()]
        with __builtins__["__import__"]("contextlib").ExitStack() as stack:
            for p in ctx:
                stack.enter_context(p)
            return resolve_route(*_START, *_END, client_waypoints=client_waypoints)

    # ---- always returns a ResolvedRoute ----

    def test_always_returns_resolved_route(self):
        with (
            patch("src.routing.get_osrm_route", return_value=None),
            patch("src.routing.find_cached_route", return_value=None),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END)
        assert isinstance(result, ResolvedRoute)

    def test_result_always_has_at_least_two_waypoints(self):
        with (
            patch("src.routing.get_osrm_route", return_value=None),
            patch("src.routing.find_cached_route", return_value=None),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END)
        assert len(result.waypoints) >= 2

    def test_result_distance_is_positive(self):
        with (
            patch("src.routing.get_osrm_route", return_value=None),
            patch("src.routing.find_cached_route", return_value=None),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END)
        assert result.distance_km > 0

    # ---- OSRM path ----

    def test_uses_osrm_waypoints_when_osrm_succeeds(self):
        with (
            patch("src.routing.get_osrm_route", return_value=_OSRM_RESPONSE),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END)
        assert result.waypoints == _OSRM_RESPONSE["waypoints"]
        assert result.traffic_source == "osrm"

    def test_osrm_distance_and_duration_used_when_osrm_succeeds(self):
        with (
            patch("src.routing.get_osrm_route", return_value=_OSRM_RESPONSE),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END)
        assert result.distance_km == 7.2
        assert result.duration_min == 14

    # ---- Cache fallback path ----

    def test_uses_cached_waypoints_when_osrm_fails(self):
        with (
            patch("src.routing.get_osrm_route", return_value=None),
            patch("src.routing.find_cached_route", return_value=_CACHED_RESPONSE),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END)
        assert result.waypoints == _CACHED_RESPONSE["waypoints"]
        assert result.traffic_source == "cached_osrm"

    # ---- Interpolation fallback path ----

    def test_traffic_source_is_estimated_when_all_external_fail(self):
        with (
            patch("src.routing.get_osrm_route", return_value=None),
            patch("src.routing.find_cached_route", return_value=None),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END)
        assert result.traffic_source == "estimated"

    # ---- GMaps timing override ----

    def test_gmaps_overrides_duration_when_available(self):
        with (
            patch("src.routing.get_osrm_route", return_value=_OSRM_RESPONSE),
            patch("src.routing.get_google_maps_travel_time", return_value=_GMAPS_RESPONSE),
        ):
            result = resolve_route(*_START, *_END)
        assert result.duration_min == _GMAPS_RESPONSE["duration_minutes"]

    def test_gmaps_sets_traffic_source_when_has_traffic(self):
        with (
            patch("src.routing.get_osrm_route", return_value=_OSRM_RESPONSE),
            patch("src.routing.get_google_maps_travel_time", return_value=_GMAPS_RESPONSE),
        ):
            result = resolve_route(*_START, *_END)
        assert result.traffic_source == "google_maps_traffic"

    def test_gmaps_sets_google_maps_when_no_traffic_data(self):
        gmaps_no_traffic = {**_GMAPS_RESPONSE, "has_traffic": False}
        with (
            patch("src.routing.get_osrm_route", return_value=_OSRM_RESPONSE),
            patch("src.routing.get_google_maps_travel_time", return_value=gmaps_no_traffic),
        ):
            result = resolve_route(*_START, *_END)
        assert result.traffic_source == "google_maps"

    # ---- Client waypoints path ----

    def test_client_waypoints_used_directly_skipping_osrm(self):
        client_wps = [[13.75, 100.50], [13.77, 100.52], [13.80, 100.55]]
        with (
            patch("src.routing.get_osrm_route") as mock_osrm,
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END, client_waypoints=client_wps)
        mock_osrm.assert_not_called()
        assert result.waypoints == client_wps

    def test_empty_client_waypoints_falls_through_to_osrm(self):
        with (
            patch("src.routing.get_osrm_route", return_value=_OSRM_RESPONSE),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            result = resolve_route(*_START, *_END, client_waypoints=[])
        assert result.traffic_source == "osrm"


# ---------------------------------------------------------------------------
# generate_dense_interpolation
# ---------------------------------------------------------------------------

class TestGenerateDenseInterpolation:
    def test_returns_correct_point_count(self):
        points = generate_dense_interpolation(13.70, 100.45, 13.80, 100.55, num_points=10)
        assert len(points) == 11  # 0..10 inclusive

    def test_first_point_is_start(self):
        points = generate_dense_interpolation(13.70, 100.45, 13.80, 100.55)
        assert points[0] == [13.70, 100.45]

    def test_last_point_is_end(self):
        points = generate_dense_interpolation(13.70, 100.45, 13.80, 100.55)
        assert points[-1] == [13.80, 100.55]

    def test_all_points_are_lat_lon_pairs(self):
        points = generate_dense_interpolation(13.70, 100.45, 13.80, 100.55)
        for p in points:
            assert len(p) == 2
            assert isinstance(p[0], float)
            assert isinstance(p[1], float)

    def test_curve_keeps_points_near_straight_line(self):
        """The sinusoidal curve offset must stay small (< 0.05 degrees)."""
        points = generate_dense_interpolation(13.70, 100.45, 13.80, 100.55)
        mid = points[len(points) // 2]
        straight_lat = (13.70 + 13.80) / 2
        straight_lon = (100.45 + 100.55) / 2
        assert abs(mid[0] - straight_lat) < 0.05
        assert abs(mid[1] - straight_lon) < 0.05


# ---------------------------------------------------------------------------
# find_cached_route
# ---------------------------------------------------------------------------

class TestFindCachedRoute:
    def _mock_presets(self):
        return {
            "test_route": {
                "start": {"lat": 13.75, "lon": 100.50},
                "end":   {"lat": 13.80, "lon": 100.55},
                "waypoints": [[13.75 + i * 0.001, 100.50 + i * 0.001] for i in range(15)],
                "distance": 8.0,
                "duration": 15,
            }
        }

    def test_exact_match_returns_route(self):
        with patch("src.routing._load_route_presets", return_value=self._mock_presets()):
            result = find_cached_route(13.75, 100.50, 13.80, 100.55)
        assert result is not None
        assert "waypoints" in result

    def test_far_away_returns_none(self):
        with patch("src.routing._load_route_presets", return_value=self._mock_presets()):
            result = find_cached_route(0.0, 0.0, 1.0, 1.0)
        assert result is None

    def test_within_threshold_returns_route(self):
        # 1 km off from preset start/end — within default 2.0 km threshold
        with patch("src.routing._load_route_presets", return_value=self._mock_presets()):
            result = find_cached_route(13.751, 100.501, 13.801, 100.551)
        assert result is not None


# ---------------------------------------------------------------------------
# generate_fast_route — fallback chain
# ---------------------------------------------------------------------------

class TestGenerateFastRoute:
    def test_uses_osrm_when_available(self):
        mock_waypoints = [[13.75 + i * 0.001, 100.5] for i in range(20)]
        mock_osrm = {"waypoints": mock_waypoints, "distance_km": 5.0, "duration_min": 10}
        with patch("src.routing.get_osrm_route", return_value=mock_osrm):
            result = generate_fast_route(13.75, 100.50, 13.80, 100.55)
        assert result[0] == mock_waypoints[0]
        assert result[-1] == mock_waypoints[-1]

    def test_falls_back_to_interpolation_when_osrm_and_cache_unavailable(self):
        with (
            patch("src.routing.get_osrm_route", return_value=None),
            patch("src.routing.find_cached_route", return_value=None),
        ):
            result = generate_fast_route(13.75, 100.50, 13.80, 100.55)
        assert isinstance(result, list)
        assert len(result) > 2


# ---------------------------------------------------------------------------
# get_osrm_alternatives — fallback to cache
# ---------------------------------------------------------------------------

class TestGetOsrmAlternatives:
    def test_returns_empty_list_when_both_osrm_and_cache_fail(self):
        with (
            patch("src.routing.requests.get", side_effect=Exception("network error")),
            patch("src.routing.find_cached_route", return_value=None),
        ):
            result = get_osrm_alternatives(13.75, 100.50, 13.80, 100.55)
        assert result == []

    def test_returns_cached_route_when_osrm_fails(self):
        mock_cached = {
            "waypoints": [[13.75 + i * 0.001, 100.5] for i in range(15)],
            "distance_km": 5.0,
            "duration_min": 10,
        }
        with (
            patch("src.routing.requests.get", side_effect=Exception("network error")),
            patch("src.routing.find_cached_route", return_value=mock_cached),
        ):
            result = get_osrm_alternatives(13.75, 100.50, 13.80, 100.55)
        assert len(result) == 1
        assert result[0] == mock_cached
