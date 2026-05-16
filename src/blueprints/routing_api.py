"""
Route analysis and comparison API endpoints.

Routes:
    POST /api/analyze-route
    POST /api/compare-routes
"""

import logging

from flask import Blueprint, jsonify, request

from ..risk import (
    analyze_route_segments,
    generate_alerts_from_segments,
    rank_and_annotate_routes,
    score_route,
)
from ..routing import get_route_alternatives, resolve_route
from ..utils.data import load_grid_stats
from ..utils.rate_limit import rate_limit
from ..utils.validators import validate_coordinates

logger = logging.getLogger(__name__)

routing_api_bp = Blueprint("routing_api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_coordinates(payload: dict) -> tuple[float, float, float, float]:
    """Extract, type-cast, and range-validate start/end coordinates."""
    required = {"start_lat", "start_lon", "end_lat", "end_lon"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"Missing fields: {sorted(missing)}")
    try:
        start_lat = float(payload["start_lat"])
        start_lon = float(payload["start_lon"])
        end_lat   = float(payload["end_lat"])
        end_lon   = float(payload["end_lon"])
    except (TypeError, ValueError):
        raise ValueError("All coordinates must be valid numbers")

    validate_coordinates(start_lat, start_lon)
    validate_coordinates(end_lat, end_lon)

    return start_lat, start_lon, end_lat, end_lon


# ---------------------------------------------------------------------------
# POST /api/analyze-route
# ---------------------------------------------------------------------------

@routing_api_bp.route("/analyze-route", methods=["POST"])
@rate_limit(max_calls=60, window_seconds=3600)
def analyze_route():
    try:
        data = request.get_json(silent=True) or {}
        try:
            start_lat, start_lon, end_lat, end_lon = _parse_coordinates(data)
        except ValueError as exc:
            logger.debug("Invalid coordinates in /api/analyze-route: %s", exc)
            return jsonify({"error": "Invalid or missing coordinates."}), 400

        # ── Parse optional pre-computed waypoints from the client ──
        parsed_waypoints: list[list[float]] = [
            [float(p[0]), float(p[1])]
            for p in (data.get("waypoints") or [])
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]

        resolved = resolve_route(
            start_lat, start_lon, end_lat, end_lon,
            client_waypoints=parsed_waypoints or None,
        )
        waypoints = resolved.waypoints
        route_distance_km = resolved.distance_km
        route_duration_min = resolved.duration_min
        traffic_source = resolved.traffic_source

        sampled = waypoints[:: max(1, len(waypoints) // 50)]
        grid_df = load_grid_stats()
        segments = analyze_route_segments(sampled, grid_df, segment_distance_km=1.0)
        alerts = generate_alerts_from_segments(segments, advance_warning_km=0.5)

        overall_risk = score_route(sampled, grid_df)
        prediction_method = overall_risk["prediction_method"]

        high   = sum(1 for s in segments if s["risk_level"] == "high")
        medium = sum(1 for s in segments if s["risk_level"] == "medium")
        low    = sum(1 for s in segments if s["risk_level"] == "low")

        return jsonify({
            "success": True,
            "route": {
                "waypoints": waypoints,
                "distance_km": round(route_distance_km, 2),
                "duration_min": route_duration_min,
                "risk_score": min(100, overall_risk["risk_score"]),
                "risk_level": overall_risk["risk_level"],
                "high_risk_km": overall_risk.get("high_risk_km", 0),
                "traffic_source": traffic_source,
            },
            "segments": segments,
            "alerts": alerts,
            "summary": {
                "total_distance": round(route_distance_km, 2),
                "total_time": route_duration_min,
                "high_risk_segments": high,
                "medium_risk_segments": medium,
                "low_risk_segments": low,
                "total_alerts": len(alerts),
                "prediction_method": prediction_method,
                "model_accuracy": (
                    "83.8% holdout crash capture" if prediction_method == "xgboost"
                    else "83.8% spatial baseline capture"
                ),
                "traffic_source": traffic_source,
            },
        })
    except Exception:
        logger.exception("Unhandled error in /api/analyze-route")
        return jsonify({"error": "An error occurred while analysing the route."}), 500


# ---------------------------------------------------------------------------
# POST /api/compare-routes
# ---------------------------------------------------------------------------

@routing_api_bp.route("/compare-routes", methods=["POST"])
@rate_limit(max_calls=30, window_seconds=3600)
def compare_routes():
    try:
        payload = request.get_json(silent=True) or {}
        try:
            start_lat, start_lon, end_lat, end_lon = _parse_coordinates(payload)
        except ValueError as exc:
            logger.debug("Invalid coordinates in /api/compare-routes: %s", exc)
            return jsonify({"error": "Invalid or missing coordinates."}), 400

        if start_lat == end_lat and start_lon == end_lon:
            return jsonify({"error": "Start and end locations must be different"}), 400

        grid_df = load_grid_stats()
        scored_routes, alternatives_analyzed = get_route_alternatives(
            start_lat, start_lon, end_lat, end_lon, grid_df,
        )
        ranked = rank_and_annotate_routes(scored_routes)
        route_fast = ranked["route_fast"]
        route_safe = ranked["route_safe"]

        return jsonify({
            **ranked,
            "alternatives_analyzed": alternatives_analyzed,
            "debug": {
                "fast_waypoints_sample": route_fast["waypoints"][:3],
                "safe_waypoints_sample": route_safe["waypoints"][:3],
                "fast_risk_details": (
                    f"Calculated from {route_fast.get('matched_grids', 0)} matched grids "
                    f"across {route_fast.get('evaluated_waypoints', 0)} waypoints"
                ),
                "safe_risk_details": (
                    f"Calculated from {route_safe.get('matched_grids', 0)} matched grids "
                    f"across {route_safe.get('evaluated_waypoints', 0)} waypoints"
                ),
                "fast_summary": route_fast.get("summary", ""),
                "safe_summary": route_safe.get("summary", ""),
                "google_maps_alternatives": alternatives_analyzed,
            },
        })
    except Exception:
        logger.exception("Unhandled error in /api/compare-routes")
        return jsonify({"error": "An error occurred while comparing routes."}), 500
