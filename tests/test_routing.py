"""
Tests for src/routing.py

Covers: dense interpolation shape and bounds, cached-route lookup logic,
waypoint sampling, and the OSRM/Google Maps fallback chain (using mocks so
tests run offline without network calls).
"""

from unittest.mock import MagicMock, patch

import pytest

import json

import src.routing as routing_mod
from src.routing import (
    ResolvedRoute,
    find_cached_route,
    generate_dense_interpolation,
    generate_fast_route,
    get_osrm_alternatives,
    get_route_alternatives,
    reload_route_presets,
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


class TestGoogleRoutesApi:
    def test_startup_does_not_create_legacy_directions_client(self, monkeypatch):
        from src.app import create_app

        monkeypatch.setattr("src.app.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        with patch("googlemaps.Client") as legacy_client:
            create_app()
        legacy_client.assert_not_called()

    def test_traffic_aware_time_uses_compute_routes(self, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        response = MagicMock()
        response.json.return_value = {
            "routes": [{"duration": "720.5s", "distanceMeters": 7012}]
        }
        with patch("src.routing.requests.post", return_value=response) as post:
            result = routing_mod.get_google_maps_travel_time(*_START, *_END)

        assert result == {
            "duration_minutes": 12, "distance_km": 7.01, "has_traffic": True,
        }
        args, kwargs = post.call_args
        assert args == ("https://routes.googleapis.com/directions/v2:computeRoutes",)
        assert kwargs["headers"]["X-Goog-Api-Key"] == "test-server-key"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        mask = kwargs["headers"]["X-Goog-FieldMask"].split(",")
        assert "routes.duration" in mask and "routes.distanceMeters" in mask
        assert "*" not in mask
        assert kwargs["json"] == {
            "origin": {"location": {"latLng": {"latitude": 13.75, "longitude": 100.50}}},
            "destination": {"location": {"latLng": {"latitude": 13.80, "longitude": 100.55}}},
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_AWARE",
            "computeAlternativeRoutes": False,
        }
        assert 0 < kwargs["timeout"] <= 10
        response.raise_for_status.assert_called_once()

    def test_alternatives_include_geometry_and_keep_existing_shape(self, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        route = {
            "duration": "780s", "distanceMeters": 9200,
            "polyline": {"encodedPolyline": "_p~iF~ps|U_ulLnnqC_mqNvxq`@"},
            "description": "Main road",
        }
        response = MagicMock()
        response.json.return_value = {"routes": [route, {**route, "duration": "900s"}]}
        with patch("src.routing.requests.post", return_value=response) as post:
            routes = routing_mod.get_google_maps_routes(*_START, *_END)

        assert len(routes) == 2
        assert {key: value for key, value in routes[0].items() if key != "waypoints"} == {
            "distance_km": 9.2, "duration_minutes": 13,
            "has_traffic": True, "summary": "Main road",
        }
        expected = [[38.5, -120.2], [40.7, -120.95], [43.252, -126.453]]
        assert len(routes[0]["waypoints"]) == len(expected)
        for actual, point in zip(routes[0]["waypoints"], expected):
            assert actual == pytest.approx(point)
        assert routes[1]["duration_minutes"] == 15
        kwargs = post.call_args.kwargs
        assert kwargs["json"]["computeAlternativeRoutes"] is True
        assert "routes.polyline.encodedPolyline" in kwargs["headers"]["X-Goog-FieldMask"]
        assert "routes.description" in kwargs["headers"]["X-Goog-FieldMask"]

    def test_google_traffic_fallback_is_labelled_without_traffic(self, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        response = MagicMock()
        response.json.return_value = {
            "routes": [{"duration": "600s", "distanceMeters": 5000}],
            "fallbackInfo": {"routingMode": "FALLBACK_TRAFFIC_UNAWARE"},
        }
        with patch("src.routing.requests.post", return_value=response) as post:
            result = routing_mod.get_google_maps_travel_time(*_START, *_END)

        assert result["has_traffic"] is False
        assert "fallbackInfo" in post.call_args.kwargs["headers"]["X-Goog-FieldMask"]

    @pytest.mark.parametrize("route", [
        {"duration": "720x", "distanceMeters": 5000},
        {"duration": "-60s", "distanceMeters": 5000},
        {"duration": "600s", "distanceMeters": -1},
    ])
    def test_invalid_timing_is_unavailable_for_fallback(self, monkeypatch, route):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        response = MagicMock()
        response.json.return_value = {"routes": [route]}
        with patch("src.routing.requests.post", return_value=response):
            assert routing_mod.get_google_maps_travel_time(*_START, *_END) is None

    @pytest.mark.parametrize("polyline", ["", "_p~iF~ps|U"])
    def test_incomplete_geometry_is_unavailable_for_fallback(self, monkeypatch, polyline):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        response = MagicMock()
        response.json.return_value = {"routes": [{
            "duration": "600s", "distanceMeters": 5000,
            "polyline": {"encodedPolyline": polyline},
        }]}
        with patch("src.routing.requests.post", return_value=response):
            assert routing_mod.get_google_maps_routes(*_START, *_END) == []

    @pytest.mark.parametrize("status", [400, 403, 429, 500])
    def test_http_errors_preserve_resolver_fallback(self, monkeypatch, caplog, status):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "private-test-key")
        response = MagicMock()
        response.raise_for_status.side_effect = routing_mod.requests.HTTPError(
            f"{status}: private-test-key"
        )
        waypoints = [[13.75, 100.50], [13.80, 100.55]]
        with patch("src.routing.requests.post", return_value=response):
            result = resolve_route(*_START, *_END, client_waypoints=waypoints)
            assert routing_mod.get_google_maps_routes(*_START, *_END) == []

        assert result.waypoints == waypoints
        assert result.traffic_source == "estimated"
        assert result.distance_km > 0 and result.duration_min > 0
        assert "private-test-key" not in caplog.text
        response.json.assert_not_called()

    def test_timeout_returns_no_routes(self, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        with patch("src.routing.requests.post", side_effect=routing_mod.requests.Timeout):
            assert routing_mod.get_google_maps_travel_time(*_START, *_END) is None
            assert routing_mod.get_google_maps_routes(*_START, *_END) == []

    @pytest.mark.parametrize("payload", [{}, {"routes": []}, {"routes": [{}]}])
    def test_empty_or_missing_route_data_returns_no_routes(self, monkeypatch, payload):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        response = MagicMock()
        response.json.return_value = payload
        with patch("src.routing.requests.post", return_value=response):
            assert routing_mod.get_google_maps_travel_time(*_START, *_END) is None
            assert routing_mod.get_google_maps_routes(*_START, *_END) == []

    def test_invalid_json_returns_no_routes(self, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        response = MagicMock()
        response.json.side_effect = ValueError("Invalid JSON")
        with patch("src.routing.requests.post", return_value=response):
            assert routing_mod.get_google_maps_travel_time(*_START, *_END) is None
            assert routing_mod.get_google_maps_routes(*_START, *_END) == []

    def test_missing_key_skips_google_requests(self, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "")
        with patch("src.routing.requests.post") as post:
            assert routing_mod.get_google_maps_travel_time(*_START, *_END) is None
            assert routing_mod.get_google_maps_routes(*_START, *_END) == []
        post.assert_not_called()


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


# ---------------------------------------------------------------------------
# get_route_alternatives — GMaps vs fallback paths
# ---------------------------------------------------------------------------

_REQUIRED_ROUTE_KEYS = {
    "waypoints", "distance_km", "time_min", "risk_score", "risk_level",
    "high_risk_zones", "high_risk_km", "medium_risk_zones", "matched_grids",
    "evaluated_waypoints", "traffic_source", "summary",
}

_MOCK_RISK = {
    "risk_score": 45, "risk_level": "medium",
    "high_risk_zones": 1, "high_risk_km": 0.5,
    "medium_risk_zones": 2, "matched_grids": 3,
    "evaluated_waypoints": 5,
}

_GMAPS_ROUTES = [
    {
        "waypoints": [[13.75 + i * 0.001, 100.50] for i in range(5)],
        "distance_km": 5.0, "duration_minutes": 12,
        "has_traffic": True, "summary": "Route A",
    },
    {
        "waypoints": [[13.75 + i * 0.001, 100.55] for i in range(5)],
        "distance_km": 6.5, "duration_minutes": 15,
        "has_traffic": False, "summary": "Route B",
    },
]

_FAST_WPS = [[13.75 + i * 0.001, 100.50] for i in range(5)]
_SAFE_WPS = [[13.76 + i * 0.001, 100.52] for i in range(5)]

_EVALUATED_FAST = {
    "waypoints": _FAST_WPS, "distance_km": 5.0, "time_min": 7,
    "risk_score": 60, "risk_level": "medium",
    "high_risk_zones": 2, "high_risk_km": 1.0,
    "medium_risk_zones": 1, "matched_grids": 3, "evaluated_waypoints": 5,
}
_EVALUATED_SAFE = {
    "waypoints": _SAFE_WPS, "distance_km": 6.0, "time_min": 9,
    "risk_score": 30, "risk_level": "low",
    "high_risk_zones": 0, "high_risk_km": 0.0,
    "medium_risk_zones": 1, "matched_grids": 3, "evaluated_waypoints": 5,
}

_START_LAT, _START_LON = 13.75, 100.50
_END_LAT, _END_LON = 13.80, 100.55


class TestGetRouteAlternatives:
    def _gmaps_path(self, sample_grid_df):
        """Run get_route_alternatives with GMaps returning two routes."""
        with (
            patch("src.routing.get_google_maps_routes", return_value=_GMAPS_ROUTES),
            patch("src.risk.score_route", return_value=_MOCK_RISK),
        ):
            return get_route_alternatives(_START_LAT, _START_LON, _END_LAT, _END_LON, sample_grid_df)

    def _fallback_path(self, sample_grid_df, gmaps_time=None):
        """Run get_route_alternatives with GMaps returning nothing."""
        with (
            patch("src.routing.get_google_maps_routes", return_value=[]),
            patch("src.routing.generate_fast_route", return_value=_FAST_WPS),
            patch("src.routing.generate_safe_route", return_value=_SAFE_WPS),
            patch("src.risk.evaluate_route", side_effect=[dict(_EVALUATED_FAST), dict(_EVALUATED_SAFE)]),
            patch("src.routing.get_google_maps_travel_time", return_value=gmaps_time),
        ):
            return get_route_alternatives(_START_LAT, _START_LON, _END_LAT, _END_LON, sample_grid_df)

    # ---- return shape ----

    def test_returns_tuple_of_list_and_int(self, sample_grid_df):
        routes, count = self._gmaps_path(sample_grid_df)
        assert isinstance(routes, list)
        assert isinstance(count, int)

    def test_gmaps_path_each_route_has_required_keys(self, sample_grid_df):
        routes, _ = self._gmaps_path(sample_grid_df)
        for route in routes:
            assert _REQUIRED_ROUTE_KEYS.issubset(route.keys())

    def test_gmaps_path_count_equals_number_of_gmaps_routes(self, sample_grid_df):
        routes, count = self._gmaps_path(sample_grid_df)
        assert count == len(_GMAPS_ROUTES)

    # ---- traffic_source on GMaps path ----

    def test_gmaps_traffic_source_is_google_maps_traffic_when_has_traffic(self, sample_grid_df):
        routes, _ = self._gmaps_path(sample_grid_df)
        # _GMAPS_ROUTES[0] has has_traffic=True
        assert routes[0]["traffic_source"] == "google_maps_traffic"

    def test_gmaps_traffic_source_is_google_maps_when_no_traffic(self, sample_grid_df):
        routes, _ = self._gmaps_path(sample_grid_df)
        # _GMAPS_ROUTES[1] has has_traffic=False
        assert routes[1]["traffic_source"] == "google_maps"

    # ---- fallback path (GMaps empty) ----

    def test_fallback_returns_exactly_two_routes(self, sample_grid_df):
        routes, _ = self._fallback_path(sample_grid_df)
        assert len(routes) == 2

    def test_fallback_alternatives_analyzed_is_one(self, sample_grid_df):
        _, count = self._fallback_path(sample_grid_df)
        assert count == 1

    def test_fallback_fast_route_uses_gmaps_timing_when_available(self, sample_grid_df):
        # Fallback returns [route_fast, route_safe]; route_fast is index 0
        gmaps_time = {"duration_minutes": 10, "distance_km": 5.2, "has_traffic": True}
        routes, _ = self._fallback_path(sample_grid_df, gmaps_time=gmaps_time)
        route_fast = routes[0]
        assert route_fast["time_min"] == 10
        assert route_fast["distance_km"] == 5.2

    def test_fallback_fast_route_traffic_source_is_estimated_without_gmaps(self, sample_grid_df):
        routes, _ = self._fallback_path(sample_grid_df, gmaps_time=None)
        route_fast = routes[0]
        assert route_fast["traffic_source"] == "estimated"

    def test_fallback_safe_route_traffic_source_is_always_estimated(self, sample_grid_df):
        # safe route (index 1) always gets "estimated" even when GMaps timing was applied to fast
        gmaps_time = {"duration_minutes": 10, "distance_km": 5.2, "has_traffic": True}
        routes, _ = self._fallback_path(sample_grid_df, gmaps_time=gmaps_time)
        route_safe = routes[1]
        assert route_safe["traffic_source"] == "estimated"


# ---------------------------------------------------------------------------
# reload_route_presets — in-process cache bust
# ---------------------------------------------------------------------------

_PRESET_V1 = {
    "test_route": {
        "start": {"lat": 13.75, "lon": 100.50},
        "end":   {"lat": 13.80, "lon": 100.55},
        "waypoints": [[13.75 + i * 0.001, 100.5] for i in range(15)],
        "distance": 8.0,
        "duration": 15,
    }
}


class TestReloadRoutePresets:
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        routing_mod._route_presets_cache = None
        yield
        routing_mod._route_presets_cache = None

    def test_after_reload_find_cached_route_sees_updated_file(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        presets_file.write_text(json.dumps(_PRESET_V1))

        with patch("src.routing.ROUTE_PRESETS_PATH", presets_file):
            # first call populates the in-process cache
            assert find_cached_route(13.75, 100.50, 13.80, 100.55) is not None
            # overwrite the file with an empty preset map
            presets_file.write_text(json.dumps({}))
            # without reload the old cache is still served
            assert find_cached_route(13.75, 100.50, 13.80, 100.55) is not None
            # after reload the next call reads the now-empty file
            reload_route_presets()
            assert find_cached_route(13.75, 100.50, 13.80, 100.55) is None

    def test_without_reload_disk_change_is_not_visible(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        presets_file.write_text(json.dumps(_PRESET_V1))

        with patch("src.routing.ROUTE_PRESETS_PATH", presets_file):
            find_cached_route(13.75, 100.50, 13.80, 100.55)
            presets_file.write_text(json.dumps({}))
            # no reload — still sees old data
            assert find_cached_route(13.75, 100.50, 13.80, 100.55) is not None

    def test_reload_is_safe_when_cache_already_empty(self):
        routing_mod._route_presets_cache = None
        reload_route_presets()  # must not raise
        assert routing_mod._route_presets_cache is None

    def test_reload_is_idempotent(self, tmp_path):
        presets_file = tmp_path / "route_presets.json"
        presets_file.write_text(json.dumps(_PRESET_V1))
        with patch("src.routing.ROUTE_PRESETS_PATH", presets_file):
            find_cached_route(13.75, 100.50, 13.80, 100.55)
            reload_route_presets()
            reload_route_presets()  # second call must not raise
            assert routing_mod._route_presets_cache is None


# ---------------------------------------------------------------------------
# generate_safe_route — bounded OSRM fan-out
# ---------------------------------------------------------------------------

class TestSafeRouteFanout:
    def test_osrm_calls_are_bounded(self, sample_grid_df):
        osrm = MagicMock(return_value=None)  # every OSRM leg "fails"
        with patch.object(routing_mod, "get_osrm_route", osrm), \
             patch.object(routing_mod, "get_osrm_alternatives", return_value=[]):
            # Pass fast_waypoints so generate_fast_route adds no OSRM calls of its own.
            fast = [[13.69, 100.44], [13.705, 100.455]]
            routing_mod.generate_safe_route(
                13.69, 100.44, 13.705, 100.455, sample_grid_df, fast_waypoints=fast,
            )
        # Each surviving candidate issues exactly two OSRM calls; the cap bounds it.
        assert 0 < osrm.call_count <= 2 * routing_mod._MAX_DETOUR_CANDIDATES
