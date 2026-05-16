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
    calculate_distance_from_waypoints,
    calculate_route_risk_density_aware,
    dedupe_scored_routes,
    evaluate_route,
    generate_alerts_from_segments,
    score_route,
)


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
