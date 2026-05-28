"""
Boundary tests for RiskPredictor (the ML core).

Covers the public contract — model-load precedence, predict_risk, predict_batch,
fallback_to_spatial — plus the pure score-mapping helpers. The ML-path tests
skip gracefully when the model file is absent, so the suite passes in either
environment; the spatial-fallback path is always exercised via a predictor
constructed with a non-existent model path.
"""

import pytest

from src.ml_model import RiskPredictor


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: loading the model + CSVs is the expensive part)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def predictor():
    """Default predictor — loads the real .ubj model if present."""
    return RiskPredictor()


@pytest.fixture(scope="module")
def spatial_predictor():
    """Predictor forced into spatial-only mode (no model file at this path)."""
    return RiskPredictor("/nonexistent/path/xgboost_model.pkl")


# A handful of central-Bangkok coordinates that fall inside the grid.
CENTRAL_POINTS = [(13.75, 100.50), (13.7563, 100.5018), (13.72, 100.53)]


# ---------------------------------------------------------------------------
# Pure score-mapping helpers
# ---------------------------------------------------------------------------

class TestStaticHelpers:
    def test_score_to_level_boundaries(self):
        assert RiskPredictor._score_to_level(0) == "low"
        assert RiskPredictor._score_to_level(30) == "low"
        assert RiskPredictor._score_to_level(31) == "medium"
        assert RiskPredictor._score_to_level(70) == "medium"
        assert RiskPredictor._score_to_level(71) == "high"
        assert RiskPredictor._score_to_level(100) == "high"

    def test_probability_to_risk_score_scales_and_clamps(self):
        assert RiskPredictor._probability_to_risk_score(0.0) == 0
        assert RiskPredictor._probability_to_risk_score(1.0) == 100
        assert RiskPredictor._probability_to_risk_score(0.5) == 50
        # clamps out-of-range probabilities
        assert RiskPredictor._probability_to_risk_score(-0.5) == 0
        assert RiskPredictor._probability_to_risk_score(2.0) == 100


# ---------------------------------------------------------------------------
# Model-load precedence (.ubj first, .pkl fallback, else spatial-only)
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_missing_model_is_none(self, spatial_predictor):
        assert spatial_predictor._model is None
        assert spatial_predictor._model_is_booster is False

    def test_default_path_prefers_ubj_when_present(self, predictor):
        # When the .ubj exists it must load as a Booster (never as legacy pkl).
        if predictor._model is None:
            pytest.skip("no model file in this environment")
        assert predictor._model_is_booster is True

    def test_grid_data_loads_even_without_model(self, spatial_predictor):
        # Spatial fallback needs the grid stats regardless of the model.
        assert not spatial_predictor._grid_stats.empty


# ---------------------------------------------------------------------------
# predict_risk
# ---------------------------------------------------------------------------

class TestPredictRisk:
    def test_out_of_bounds(self, predictor):
        result = predictor.predict_risk(0.0, 0.0)
        assert result["method"] == "out_of_bounds"
        assert result["grid_id"] is None
        assert result["risk_score"] == 35

    def test_in_bounds_has_full_contract(self, predictor):
        result = predictor.predict_risk(*CENTRAL_POINTS[0])
        assert set(result) >= {"risk_score", "risk_level", "confidence", "method", "grid_id"}

    def test_risk_score_in_range_and_level_valid(self, predictor):
        for lat, lon in CENTRAL_POINTS:
            r = predictor.predict_risk(lat, lon)
            assert 0 <= r["risk_score"] <= 100
            assert r["risk_level"] in {"low", "medium", "high"}

    def test_spatial_predictor_falls_back(self, spatial_predictor):
        r = spatial_predictor.predict_risk(*CENTRAL_POINTS[0])
        assert r["method"] == "spatial_fallback"

    def test_ml_predictor_uses_xgboost_for_a_matched_grid(self, predictor):
        if predictor._model is None:
            pytest.skip("no model file in this environment")
        methods = {predictor.predict_risk(lat, lon)["method"] for lat, lon in CENTRAL_POINTS}
        assert "xgboost" in methods


# ---------------------------------------------------------------------------
# fallback_to_spatial
# ---------------------------------------------------------------------------

class TestFallbackToSpatial:
    def test_out_of_bounds_default(self, spatial_predictor):
        r = spatial_predictor.fallback_to_spatial(0.0, 0.0)
        assert r["grid_id"] is None
        assert r["risk_score"] == 35
        assert r["confidence"] == 0.1
        assert r["method"] == "spatial_fallback"

    def test_score_is_double_accident_count_capped(self, spatial_predictor):
        gs = spatial_predictor._grid_stats
        row = gs[gs["total_accidents"] > 0].iloc[0]
        lat = (row["lat_min"] + row["lat_max"]) / 2
        lon = (row["lon_min"] + row["lon_max"]) / 2
        expected = min(100, int(int(row["total_accidents"]) * 2.0))
        r = spatial_predictor.fallback_to_spatial(lat, lon)
        assert r["method"] == "spatial_fallback"
        assert r["risk_score"] == expected


# ---------------------------------------------------------------------------
# predict_batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_empty_returns_empty(self, predictor):
        assert predictor.predict_batch([]) == []

    def test_length_matches_input(self, predictor):
        out = predictor.predict_batch(CENTRAL_POINTS)
        assert len(out) == len(CENTRAL_POINTS)

    def test_out_of_bounds_entry_tagged(self, predictor):
        out = predictor.predict_batch([(0.0, 0.0), CENTRAL_POINTS[0]])
        assert out[0]["method"] == "out_of_bounds"

    def test_batch_matches_single_for_same_points(self, predictor):
        batch = predictor.predict_batch(CENTRAL_POINTS)
        for (lat, lon), batched in zip(CENTRAL_POINTS, batch):
            single = predictor.predict_risk(lat, lon)
            assert batched["risk_score"] == single["risk_score"]
            assert batched["grid_id"] == single["grid_id"]
