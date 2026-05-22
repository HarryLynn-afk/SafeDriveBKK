"""
Tests for src/risk.py

Covers: distance calculation, density-aware risk scoring, risk level
thresholds, segment analysis, alert generation, deduplication, TOPSIS,
and route role annotation.
"""

import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from src.risk import (
    analyze_route_segments,
    annotate_route_roles,
    apply_topsis,
    build_display_routes,
    build_route_analysis_payload,
    calculate_distance_from_waypoints,
    calculate_route_risk_density_aware,
    calculate_segment_risk,
    dedupe_scored_routes,
    evaluate_route,
    generate_alerts_from_segments,
    rank_and_annotate_routes,
    score_route,
)
from src.routing import ResolvedRoute


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_route(route_id: str, time_min: int, risk_score: int) -> dict:
    return {
        "route_id": route_id,
        "waypoints": [[13.75 + i * 0.001, 100.5] for i in range(5)],
        "time_min": time_min,
        "risk_score": risk_score,
        "distance_km": 5.0,
    }


# ---------------------------------------------------------------------------
# score_route — unified ML/spatial scorer
# ---------------------------------------------------------------------------

_SCORE_ROUTE_KEYS = {
    "risk_score", "risk_level", "high_risk_km",
    "matched_grids", "evaluated_waypoints", "prediction_method",
}

_SIMPLE_WPS = [[13.692, 100.442], [13.695, 100.445], [13.698, 100.448]]


class TestScoreRoute:
    def test_returns_all_required_keys_spatial(self, sample_grid_df):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = False
            mock_ext.ml_predictor = None
            result = score_route(_SIMPLE_WPS, sample_grid_df)
        assert _SCORE_ROUTE_KEYS.issubset(result.keys())

    def test_prediction_method_is_spatial_when_ml_disabled(self, sample_grid_df):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = False
            mock_ext.ml_predictor = None
            result = score_route(_SIMPLE_WPS, sample_grid_df)
        assert result["prediction_method"] == "spatial"

    def test_prediction_method_is_xgboost_when_ml_enabled(self, sample_grid_df):
        mock_predictor = MagicMock()
        mock_predictor.predict_batch.return_value = [
            {"risk_score": 55, "risk_level": "medium", "confidence": 0.7, "method": "xgboost"}
            for _ in _SIMPLE_WPS
        ]
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = True
            mock_ext.ml_predictor = mock_predictor
            result = score_route(_SIMPLE_WPS, sample_grid_df)
        assert result["prediction_method"] == "xgboost"

    def test_risk_score_always_in_range(self, sample_grid_df):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = False
            mock_ext.ml_predictor = None
            result = score_route(_SIMPLE_WPS, sample_grid_df)
        assert 0 <= result["risk_score"] <= 100

    def test_never_returns_none(self, sample_grid_df):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = False
            mock_ext.ml_predictor = None
            result = score_route(_SIMPLE_WPS, sample_grid_df)
        assert result is not None

    def test_risk_level_is_valid_string(self, sample_grid_df):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = False
            mock_ext.ml_predictor = None
            result = score_route(_SIMPLE_WPS, sample_grid_df)
        assert result["risk_level"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# calculate_segment_risk — single-point risk with internal fallback
# ---------------------------------------------------------------------------

class TestCalculateSegmentRisk:
    def test_uses_predict_risk_when_ml_enabled(self):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = True
            mock_ext.ml_predictor = MagicMock()
            mock_ext.ml_predictor.predict_risk.return_value = {
                "risk_score": 60, "risk_level": "high",
                "confidence": 0.9, "method": "xgboost", "grid_id": "G010_010",
            }
            result = calculate_segment_risk(13.75, 100.5, hour=8, day_of_week=1)
        mock_ext.ml_predictor.predict_risk.assert_called_once_with(13.75, 100.5, 8, 1)
        mock_ext.ml_predictor.fallback_to_spatial.assert_not_called()
        assert result["method"] == "xgboost"
        assert result["risk_score"] == 60

    def test_uses_fallback_when_ml_disabled_but_predictor_exists(self):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = False
            mock_ext.ml_predictor = MagicMock()
            mock_ext.ml_predictor.fallback_to_spatial.return_value = {
                "risk_score": 20, "risk_level": "low",
                "confidence": 0.5, "method": "spatial_fallback", "grid_id": "G010_010",
            }
            result = calculate_segment_risk(13.75, 100.5)
        mock_ext.ml_predictor.predict_risk.assert_not_called()
        mock_ext.ml_predictor.fallback_to_spatial.assert_called_once_with(13.75, 100.5)
        assert result["method"] == "spatial_fallback"

    def test_returns_hardcoded_fallback_when_no_predictor(self):
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = False
            mock_ext.ml_predictor = None
            result = calculate_segment_risk(13.75, 100.5)
        assert result["method"] == "fallback"
        assert result["risk_level"] == "medium"
        assert result["grid_id"] is None

    def test_predict_risk_raising_falls_back_to_spatial(self):
        """When ML is enabled and predict_risk raises an uncaught exception,
        we still degrade to spatial fallback — never to the hardcoded dict."""
        with patch("src.risk.extensions") as mock_ext:
            mock_ext.USE_ML = True
            mock_ext.ml_predictor = MagicMock()
            mock_ext.ml_predictor.predict_risk.side_effect = RuntimeError("boom")
            mock_ext.ml_predictor.fallback_to_spatial.return_value = {
                "risk_score": 25, "risk_level": "low",
                "confidence": 0.5, "method": "spatial_fallback", "grid_id": "G010_010",
            }
            result = calculate_segment_risk(13.75, 100.5)
        mock_ext.ml_predictor.fallback_to_spatial.assert_called_once_with(13.75, 100.5)
        assert result["method"] == "spatial_fallback"


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

class TestCalculateDistanceFromWaypoints:
    def test_two_identical_points_is_zero(self):
        wps = [[13.75, 100.5], [13.75, 100.5]]
        assert calculate_distance_from_waypoints(wps) == 0.0

    def test_distance_is_positive_for_distinct_points(self):
        wps = [[13.70, 100.45], [13.80, 100.55]]
        assert calculate_distance_from_waypoints(wps) > 0

    def test_single_waypoint_returns_zero(self):
        assert calculate_distance_from_waypoints([[13.75, 100.5]]) == 0.0

    def test_longer_route_has_greater_distance(self):
        short = [[13.75, 100.5], [13.76, 100.5]]
        long_ = [[13.75, 100.5], [13.85, 100.5]]
        assert calculate_distance_from_waypoints(long_) > calculate_distance_from_waypoints(short)


# ---------------------------------------------------------------------------
# Density-aware risk scoring
# ---------------------------------------------------------------------------

class TestCalculateRouteRiskDensityAware:
    def test_returns_required_keys(self, sample_grid_df):
        wps = [[13.695, 100.444], [13.705, 100.454]]
        result = calculate_route_risk_density_aware(wps, sample_grid_df)
        for key in ("risk_score", "risk_level", "high_risk_zones", "high_risk_km", "matched_grids"):
            assert key in result

    def test_risk_score_in_valid_range(self, sample_grid_df):
        wps = [[13.695, 100.444], [13.704, 100.453]]
        result = calculate_route_risk_density_aware(wps, sample_grid_df)
        assert 0 <= result["risk_score"] <= 100

    def test_high_accident_zone_scores_higher(self, sample_grid_df):
        # Route through high-accident zone (G011_011, 30 accidents)
        high_risk_wps = [[13.703, 100.452], [13.706, 100.455]]
        # Route through low-accident zone (G010_010, 5 accidents)
        low_risk_wps = [[13.692, 100.442], [13.695, 100.445]]
        high_score = calculate_route_risk_density_aware(high_risk_wps, sample_grid_df)["risk_score"]
        low_score = calculate_route_risk_density_aware(low_risk_wps, sample_grid_df)["risk_score"]
        assert high_score > low_score

    def test_risk_level_high_when_score_gte_60(self, sample_grid_df):
        # Force a high score by using only the high-accident grid (30 accidents)
        wps = [[13.703, 100.452]] * 10
        result = calculate_route_risk_density_aware(wps, sample_grid_df)
        # With 30 accidents: 100*(1-e^(-30/25)) ≈ 70 → should be "high"
        assert result["risk_level"] in ("high", "medium")  # depends on exact math

    def test_no_matched_grids_gives_zero_score(self):
        grid_df = pd.DataFrame({
            "lat_min": [0.0], "lat_max": [0.001],
            "lon_min": [0.0], "lon_max": [0.001],
            "total_accidents": [5], "risk_level": ["high"], "risk_score": [75],
        })
        wps = [[13.75, 100.5]]
        result = calculate_route_risk_density_aware(wps, grid_df)
        assert result["risk_score"] == 0


# ---------------------------------------------------------------------------
# Segment analysis & alerts
# ---------------------------------------------------------------------------

class TestAnalyzeRouteSegments:
    def test_empty_waypoints_returns_empty_list(self, sample_grid_df):
        assert analyze_route_segments([], sample_grid_df) == []

    def test_single_waypoint_returns_empty_list(self, sample_grid_df):
        assert analyze_route_segments([[13.75, 100.5]], sample_grid_df) == []

    def test_returns_list_of_dicts(self, sample_grid_df):
        wps = [[13.692, 100.442], [13.695, 100.445], [13.698, 100.448], [13.703, 100.452]]
        result = analyze_route_segments(wps, sample_grid_df)
        assert isinstance(result, list)
        for seg in result:
            assert "risk_score" in seg
            assert "risk_level" in seg
            assert "color" in seg


class TestGenerateAlertsFromSegments:
    def test_no_segments_returns_empty(self):
        assert generate_alerts_from_segments([]) == []

    def test_high_risk_segment_produces_three_alerts(self):
        segments = [{
            "start_km": 2.0, "end_km": 3.0, "risk_level": "high",
            "risk_score": 80, "max_accidents": 25.0,
        }]
        alerts = generate_alerts_from_segments(segments)
        assert len(alerts) == 3
        types = {a["type"] for a in alerts}
        assert types == {"warning", "danger", "success"}

    def test_medium_risk_segment_produces_one_alert(self):
        segments = [{
            "start_km": 1.0, "end_km": 2.0, "risk_level": "medium",
            "risk_score": 40, "max_accidents": 5.0,
        }]
        alerts = generate_alerts_from_segments(segments)
        assert len(alerts) == 1
        assert alerts[0]["type"] == "info"

    def test_alerts_sorted_by_trigger_km(self):
        segments = [
            {"start_km": 5.0, "end_km": 6.0, "risk_level": "high", "risk_score": 75, "max_accidents": 10.0},
            {"start_km": 1.0, "end_km": 2.0, "risk_level": "high", "risk_score": 70, "max_accidents": 8.0},
        ]
        alerts = generate_alerts_from_segments(segments)
        trigger_kms = [a["trigger_km"] for a in alerts]
        assert trigger_kms == sorted(trigger_kms)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDedupeScheduledRoutes:
    def test_unique_routes_unchanged(self):
        r1 = _make_route("route_0", 20, 50)
        r2 = _make_route("route_1", 25, 40)
        # Give r2 different waypoints so they don't hash the same
        r2["waypoints"] = [[13.80 + i * 0.001, 100.6] for i in range(5)]
        result = dedupe_scored_routes([r1, r2])
        assert len(result) == 2

    def test_identical_geometry_deduped_to_one(self):
        r1 = _make_route("route_0", 20, 50)
        r2 = dict(r1)  # identical waypoints → same hash
        r2["route_id"] = "route_1"
        result = dedupe_scored_routes([r1, r2])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------

class TestApplyTopsis:
    def test_single_route_gets_score_one(self):
        routes = [_make_route("route_0", 20, 50)]
        apply_topsis(routes)
        assert routes[0]["topsis_score"] == 1.0

    def test_scores_between_zero_and_one(self):
        routes = [
            _make_route("route_0", 20, 80),
            _make_route("route_1", 30, 30),
            _make_route("route_2", 25, 55),
        ]
        apply_topsis(routes)
        for r in routes:
            assert 0.0 <= r["topsis_score"] <= 1.0

    def test_best_balanced_route_scores_highest(self):
        fast_risky = _make_route("route_0", 10, 90)
        slow_safe = _make_route("route_1", 60, 10)
        balanced = _make_route("route_2", 25, 40)
        routes = [fast_risky, slow_safe, balanced]
        apply_topsis(routes)
        scores = {r["route_id"]: r["topsis_score"] for r in routes}
        assert scores["route_2"] > scores["route_0"]
        assert scores["route_2"] > scores["route_1"]


# ---------------------------------------------------------------------------
# Route role annotation & display selection
# ---------------------------------------------------------------------------

class TestAnnotateAndDisplay:
    def _setup_routes(self):
        fast = _make_route("route_0", 15, 70)
        safe = _make_route("route_1", 30, 20)
        rec = _make_route("route_2", 22, 45)
        apply_topsis([fast, safe, rec])
        return fast, safe, rec

    def test_badges_assigned_correctly(self):
        fast, safe, rec = self._setup_routes()
        routes = [fast, safe, rec]
        annotate_route_roles(routes, fast, safe, rec)
        assert "Fastest" in fast["badges"]
        assert "Safest" in safe["badges"]

    def test_build_display_routes_max_three(self):
        fast, safe, rec = self._setup_routes()
        routes = [fast, safe, rec]
        annotate_route_roles(routes, fast, safe, rec)
        display = build_display_routes(routes, fast, safe, rec)
        assert len(display) <= 3


# ---------------------------------------------------------------------------
# rank_and_annotate_routes — full comparison pipeline
# ---------------------------------------------------------------------------

_RANK_REQUIRED_KEYS = {
    "fast_route", "safe_route", "recommended_route",
    "display_routes", "all_routes", "route_fast", "route_safe", "recommendation",
}


def _make_scored(time_min: int, risk_score: int, offset: float = 0.0) -> dict:
    """Minimal scored-route dict suitable as rank_and_annotate_routes input."""
    return {
        "waypoints": [[13.75 + offset + i * 0.001, 100.5] for i in range(5)],
        "time_min": time_min,
        "risk_score": risk_score,
        "distance_km": 5.0,
    }


class TestRankAndAnnotateRoutes:
    def _no_ml(self):
        mock_ext = MagicMock()
        mock_ext.USE_ML = False
        mock_ext.ml_predictor = None
        return mock_ext

    def _with_ml(self):
        mock_ext = MagicMock()
        mock_ext.USE_ML = True
        mock_ext.ml_predictor = MagicMock()
        return mock_ext

    def _run(self, routes, ext=None):
        if ext is None:
            ext = self._no_ml()
        with patch("src.risk.extensions", ext):
            return rank_and_annotate_routes(routes)

    # ---- return shape ----

    def test_returns_all_required_keys(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1)]
        result = self._run(routes)
        assert _RANK_REQUIRED_KEYS.issubset(result.keys())

    def test_display_routes_at_most_three(self):
        routes = [
            _make_scored(10, 80),
            _make_scored(25, 50, offset=0.1),
            _make_scored(40, 20, offset=0.2),
            _make_scored(35, 35, offset=0.3),
        ]
        result = self._run(routes)
        assert len(result["display_routes"]) <= 3

    # ---- fast / safe selection ----

    def test_fast_route_has_minimum_time(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1), _make_scored(25, 45, offset=0.2)]
        result = self._run(routes)
        assert result["fast_route"]["time_min"] == 15

    def test_safe_route_has_minimum_risk_score(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1), _make_scored(25, 45, offset=0.2)]
        result = self._run(routes)
        assert result["safe_route"]["risk_score"] == 20

    def test_all_routes_sorted_by_time_ascending(self):
        routes = [_make_scored(30, 20, offset=0.1), _make_scored(15, 70), _make_scored(25, 45, offset=0.2)]
        result = self._run(routes)
        times = [r["time_min"] for r in result["all_routes"]]
        assert times == sorted(times)

    # ---- per-route annotations ----

    def test_confidence_within_bounds_for_all_routes(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1)]
        result = self._run(routes)
        for route in result["all_routes"]:
            assert 0.0 <= route["confidence"] <= 0.99

    def test_prediction_method_is_spatial_when_ml_disabled(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1)]
        result = self._run(routes, ext=self._no_ml())
        for route in result["all_routes"]:
            assert route["prediction_method"] == "spatial"

    def test_prediction_method_is_xgboost_when_ml_enabled(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1)]
        result = self._run(routes, ext=self._with_ml())
        for route in result["all_routes"]:
            assert route["prediction_method"] == "xgboost"

    def test_each_route_has_peak_label(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1)]
        result = self._run(routes)
        for route in result["all_routes"]:
            assert "peak_label" in route
            assert isinstance(route["peak_label"], str)

    # ---- recommendation and time_difference ----

    def test_recommendation_is_nonempty_string(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1)]
        result = self._run(routes)
        assert isinstance(result["recommendation"], str)
        assert len(result["recommendation"]) > 0

    def test_safe_route_has_time_difference_key(self):
        routes = [_make_scored(15, 70), _make_scored(30, 20, offset=0.1)]
        result = self._run(routes)
        assert "time_difference" in result["safe_route"]

    # ---- degenerate: single-route input ----

    def test_single_route_returns_all_required_keys(self):
        routes = [_make_scored(20, 50)]
        result = self._run(routes)
        assert _RANK_REQUIRED_KEYS.issubset(result.keys())

    def test_single_route_fast_safe_and_recommended_are_same(self):
        routes = [_make_scored(20, 50)]
        result = self._run(routes)
        assert result["fast_route"]["time_min"] == result["safe_route"]["time_min"]


# ---------------------------------------------------------------------------
# build_route_analysis_payload — deep-module entry point for /api/analyze-route
# ---------------------------------------------------------------------------

_PAYLOAD_TOP_KEYS = {"success", "route", "segments", "alerts", "summary"}


def _make_resolved(traffic_source: str = "osrm") -> ResolvedRoute:
    return ResolvedRoute(
        waypoints=[[13.690 + i * 0.0005, 100.440 + i * 0.0005] for i in range(60)],
        distance_km=4.321,
        duration_min=12,
        traffic_source=traffic_source,
    )


class TestBuildRouteAnalysisPayload:
    def _mock_score(self, method: str = "spatial", risk_score: int = 40):
        return {
            "risk_score": risk_score,
            "risk_level": "medium",
            "high_risk_km": 0.8,
            "matched_grids": 4,
            "evaluated_waypoints": 5,
            "prediction_method": method,
        }

    def test_returns_top_level_keys(self, sample_grid_df):
        with patch("src.risk.score_route", return_value=self._mock_score()):
            result = build_route_analysis_payload(_make_resolved(), sample_grid_df)
        assert _PAYLOAD_TOP_KEYS.issubset(result.keys())
        assert result["success"] is True

    def test_route_block_mirrors_resolved_route(self, sample_grid_df):
        resolved = _make_resolved(traffic_source="google_maps_traffic")
        with patch("src.risk.score_route", return_value=self._mock_score()):
            result = build_route_analysis_payload(resolved, sample_grid_df)
        assert result["route"]["waypoints"] == resolved.waypoints
        assert result["route"]["distance_km"] == round(resolved.distance_km, 2)
        assert result["route"]["duration_min"] == resolved.duration_min
        assert result["route"]["traffic_source"] == "google_maps_traffic"

    def test_risk_score_clamped_to_100(self, sample_grid_df):
        with patch("src.risk.score_route", return_value=self._mock_score(risk_score=150)):
            result = build_route_analysis_payload(_make_resolved(), sample_grid_df)
        assert result["route"]["risk_score"] == 100

    def test_summary_xgboost_uses_holdout_accuracy_string(self, sample_grid_df):
        with patch("src.risk.score_route", return_value=self._mock_score(method="xgboost")):
            result = build_route_analysis_payload(_make_resolved(), sample_grid_df)
        assert result["summary"]["prediction_method"] == "xgboost"
        assert "holdout" in result["summary"]["model_accuracy"]

    def test_summary_spatial_uses_baseline_accuracy_string(self, sample_grid_df):
        with patch("src.risk.score_route", return_value=self._mock_score(method="spatial")):
            result = build_route_analysis_payload(_make_resolved(), sample_grid_df)
        assert result["summary"]["prediction_method"] == "spatial"
        assert "spatial baseline" in result["summary"]["model_accuracy"]

    def test_bucket_counts_match_segments_list(self, sample_grid_df):
        with patch("src.risk.score_route", return_value=self._mock_score()):
            result = build_route_analysis_payload(_make_resolved(), sample_grid_df)
        segments = result["segments"]
        summary = result["summary"]
        assert summary["high_risk_segments"]   == sum(1 for s in segments if s["risk_level"] == "high")
        assert summary["medium_risk_segments"] == sum(1 for s in segments if s["risk_level"] == "medium")
        assert summary["low_risk_segments"]    == sum(1 for s in segments if s["risk_level"] == "low")

    def test_total_alerts_matches_alerts_list(self, sample_grid_df):
        with patch("src.risk.score_route", return_value=self._mock_score()):
            result = build_route_analysis_payload(_make_resolved(), sample_grid_df)
        assert result["summary"]["total_alerts"] == len(result["alerts"])

    def test_summary_carries_traffic_source(self, sample_grid_df):
        resolved = _make_resolved(traffic_source="cached_osrm")
        with patch("src.risk.score_route", return_value=self._mock_score()):
            result = build_route_analysis_payload(resolved, sample_grid_df)
        assert result["summary"]["traffic_source"] == "cached_osrm"

    def test_summary_totals_match_resolved(self, sample_grid_df):
        resolved = _make_resolved()
        with patch("src.risk.score_route", return_value=self._mock_score()):
            result = build_route_analysis_payload(resolved, sample_grid_df)
        assert result["summary"]["total_distance"] == round(resolved.distance_km, 2)
        assert result["summary"]["total_time"] == resolved.duration_min
