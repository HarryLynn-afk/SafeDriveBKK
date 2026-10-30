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

    def test_routes_api_response_reaches_route_comparison(self, client, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "test-server-key")
        response = MagicMock()
        response.json.return_value = {"routes": [{
            "duration": "720s", "distanceMeters": 7012,
            "polyline": {"encodedPolyline": "op|rA_|kdRg{Cg{Cg{Cg{C"},
            "description": "Bangkok test route",
        }]}
        with (
            patch("src.routing.requests.post", return_value=response) as post,
            patch("src.routing.requests.get") as osrm,
        ):
            result = client.post("/api/compare-routes", json=self._VALID_PAYLOAD)

        assert result.status_code == 200
        payload = result.get_json()
        assert payload["alternatives_analyzed"] == 1
        route = payload["fast_route"]
        assert route["time_min"] == 12 and route["distance_km"] == 7.01
        assert route["traffic_source"] == "google_maps_traffic"
        assert route["summary"] == "Bangkok test route"
        assert route["waypoints"][0] == pytest.approx([13.75, 100.50])
        assert route["waypoints"][-1] == pytest.approx([13.80, 100.55])
        assert "test-server-key" not in result.get_data(as_text=True)
        post.assert_called_once()
        osrm.assert_not_called()

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

class TestPreviewRoute:
    _PAYLOAD = {"start_lat": 13.75, "start_lon": 100.50,
                "end_lat": 13.80, "end_lon": 100.55}

    def test_preview_uses_server_routes_api_without_alternatives(self, client, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "private-server-key")
        response = MagicMock()
        response.json.return_value = {"routes": [{
            "duration": "720s", "distanceMeters": 7012,
            "polyline": {"encodedPolyline": "op|rA_|kdRg{Cg{Cg{Cg{C"},
        }]}
        with patch("src.routing.requests.post", return_value=response) as post:
            result = client.post("/api/preview-route", json=self._PAYLOAD)
        assert result.status_code == 200
        assert result.get_json()["waypoints"][0] == pytest.approx([13.75, 100.50])
        assert result.get_json()["waypoints"][-1] == pytest.approx([13.80, 100.55])
        assert post.call_args.kwargs["json"]["computeAlternativeRoutes"] is False
        assert "private-server-key" not in result.get_data(as_text=True)

    @pytest.mark.parametrize("payload", [
        {}, [1], {"start_lat": "bad"},
        {"start_lat": 91, "start_lon": 100, "end_lat": 13.8, "end_lon": 100.55},
        {"start_lat": 13.75, "start_lon": 100.5, "end_lat": 13.75, "end_lon": 100.5},
    ])
    def test_invalid_preview_does_not_call_google(self, client, payload):
        with patch("src.routing.requests.post") as post:
            assert client.post("/api/preview-route", json=payload).status_code == 400
        post.assert_not_called()

    def test_preview_falls_back_to_osrm_without_google_key(self, client, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "")
        response = MagicMock()
        response.json.return_value = {"code": "Ok", "routes": [{
            "geometry": {"coordinates": [[100.50, 13.75], [100.55, 13.80]]},
            "distance": 7000, "duration": 720,
        }]}
        with (
            patch("src.routing.requests.post") as post,
            patch("src.routing.requests.get", return_value=response),
        ):
            result = client.post("/api/preview-route", json=self._PAYLOAD)
        assert result.status_code == 200
        assert result.get_json()["waypoints"] == [[13.75, 100.50], [13.80, 100.55]]
        post.assert_not_called()

    def test_no_road_geometry_returns_unavailable(self, client, monkeypatch):
        monkeypatch.setattr("src.config.GOOGLE_MAPS_SERVER_KEY", "")
        with patch("src.routing.requests.get", side_effect=TimeoutError):
            result = client.post("/api/preview-route", json=self._PAYLOAD)
        assert result.status_code == 503
        assert "waypoints" not in result.get_json()


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
