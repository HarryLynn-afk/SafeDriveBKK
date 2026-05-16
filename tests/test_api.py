"""
Tests for Flask API endpoints.

Uses the Flask test client (no live server, no network).
External calls (OSRM, Google Maps, Groq) are patched so tests run offline.

Covers: temporal-stats, risk-zones, model-info, analytics-data,
        analyze-route (happy path + validation), compare-routes (validation),
        chatbot (happy path + empty message), chatbot suggestions,
        NLP endpoints, geolocate (missing key), compare-predictions.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.routing import ResolvedRoute


# ---------------------------------------------------------------------------
# GET /api/temporal-stats
# ---------------------------------------------------------------------------

class TestTemporalStats:
    def test_returns_200_and_required_keys(self, client):
        resp = client.get("/api/temporal-stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "peak_hours" in data
        assert "dangerous_days" in data
        assert "rush_hour_count" in data

    def test_peak_hours_is_a_list(self, client):
        data = client.get("/api/temporal-stats").get_json()
        assert isinstance(data["peak_hours"], list)


# ---------------------------------------------------------------------------
# GET /api/risk-zones
# ---------------------------------------------------------------------------

class TestRiskZones:
    def test_returns_200(self, client):
        assert client.get("/api/risk-zones").status_code == 200

    def test_count_field_present(self, client):
        data = client.get("/api/risk-zones").get_json()
        assert "count" in data
        assert isinstance(data["count"], int)

    def test_risk_zones_is_list(self, client):
        data = client.get("/api/risk-zones").get_json()
        assert isinstance(data["risk_zones"], list)


# ---------------------------------------------------------------------------
# GET /api/model-info
# ---------------------------------------------------------------------------

class TestModelInfo:
    def test_returns_200(self, client):
        assert client.get("/api/model-info").status_code == 200

    def test_status_field_is_success(self, client):
        data = client.get("/api/model-info").get_json()
        assert data["status"] == "success"

    def test_model_field_present(self, client):
        data = client.get("/api/model-info").get_json()
        assert "model" in data


# ---------------------------------------------------------------------------
# POST /api/analyze-route
# ---------------------------------------------------------------------------

class TestAnalyzeRoute:
    _VALID_PAYLOAD = {
        "start_lat": 13.75,
        "start_lon": 100.50,
        "end_lat":   13.80,
        "end_lon":   100.55,
    }

    def test_valid_request_returns_200(self, client):
        dummy_resolved = ResolvedRoute(
            waypoints=[[13.75, 100.50], [13.775, 100.525], [13.80, 100.55]],
            distance_km=5.0,
            duration_min=10,
            traffic_source="estimated",
        )
        with patch("src.blueprints.routing_api.resolve_route", return_value=dummy_resolved):
            resp = client.post(
                "/api/analyze-route",
                json=self._VALID_PAYLOAD,
                content_type="application/json",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "route" in data
        assert "segments" in data
        assert "alerts" in data

    def test_missing_fields_returns_400(self, client):
        resp = client.post("/api/analyze-route", json={"start_lat": 13.75})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_non_numeric_coords_returns_400(self, client):
        payload = {**self._VALID_PAYLOAD, "start_lat": "not_a_number"}
        resp = client.post("/api/analyze-route", json=payload)
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /api/compare-routes
# ---------------------------------------------------------------------------

class TestCompareRoutes:
    _VALID_PAYLOAD = {
        "start_lat": 13.75,
        "start_lon": 100.50,
        "end_lat":   13.80,
        "end_lon":   100.55,
    }

    def test_same_start_end_returns_400(self, client):
        payload = {
            "start_lat": 13.75, "start_lon": 100.50,
            "end_lat":   13.75, "end_lon":   100.50,
        }
        resp = client.post("/api/compare-routes", json=payload)
        assert resp.status_code == 400

    def test_missing_fields_returns_400(self, client):
        resp = client.post("/api/compare-routes", json={"start_lat": 13.75})
        assert resp.status_code == 400

    def test_valid_request_returns_200_with_routes(self, client):
        with (
            patch("src.routing.get_google_maps_routes", return_value=[]),
            patch("src.routing.get_osrm_route", return_value=None),
            patch("src.routing.find_cached_route", return_value=None),
            patch("src.routing.get_google_maps_travel_time", return_value=None),
        ):
            resp = client.post("/api/compare-routes", json=self._VALID_PAYLOAD)
        assert resp.status_code == 200
        data = resp.get_json()
        assert "fast_route" in data
        assert "safe_route" in data
        assert "recommendation" in data


# ---------------------------------------------------------------------------
# POST /api/chatbot
# ---------------------------------------------------------------------------

class TestChatbot:
    def test_empty_message_returns_400(self, client):
        resp = client.post("/api/chatbot", json={"message": ""})
        assert resp.status_code == 400

    def test_scripted_intent_returns_200(self, client):
        """'most dangerous' triggers a scripted intent — no LLM call needed."""
        resp = client.post(
            "/api/chatbot",
            json={"message": "Which areas are most dangerous?"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "response" in data

    def test_missing_message_key_returns_400(self, client):
        resp = client.post("/api/chatbot", json={})
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/chatbot/suggestions
# ---------------------------------------------------------------------------

class TestChatbotSuggestions:
    def test_returns_200_and_list(self, client):
        resp = client.get("/api/chatbot/suggestions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) > 0


# ---------------------------------------------------------------------------
# GET /api/nlp/cause-analysis
# ---------------------------------------------------------------------------

class TestNlpEndpoints:
    def test_cause_analysis_returns_200(self, client):
        assert client.get("/api/nlp/cause-analysis").status_code == 200

    def test_wordcloud_returns_200(self, client):
        assert client.get("/api/nlp/wordcloud").status_code == 200

    def test_recommendations_returns_200(self, client):
        assert client.get("/api/nlp/recommendations").status_code == 200


# ---------------------------------------------------------------------------
# POST /api/geolocate — no API key configured
# ---------------------------------------------------------------------------

class TestGeolocate:
    def test_returns_500_when_no_key(self, client):
        with patch("src.blueprints.api.GOOGLE_MAPS_SERVER_KEY", ""):
            resp = client.post("/api/geolocate", json={})
        assert resp.status_code == 500
        assert "error" in resp.get_json()


# ---------------------------------------------------------------------------
# POST /api/compare-predictions
# ---------------------------------------------------------------------------

class TestComparePredictions:
    def test_missing_lat_lon_returns_400(self, client):
        resp = client.post("/api/compare-predictions", json={})
        assert resp.status_code == 400

    def test_valid_coords_return_200(self, client):
        resp = client.post(
            "/api/compare-predictions",
            json={"lat": 13.75, "lon": 100.5},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "location" in data
