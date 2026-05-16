"""
Data / analytics API endpoints.

Routes:
    GET  /api/risk-zones
    GET  /api/temporal-stats
    GET  /api/analytics-data
    GET  /api/model-info
    POST /api/geolocate
    POST /api/compare-predictions
"""

import logging

import pandas as pd
import requests as http_requests
from flask import Blueprint, jsonify, request

from .. import extensions
from ..config import GOOGLE_MAPS_SERVER_KEY
from ..utils.data import (
    build_analytics_payload,
    calculate_temporal_stats,
    load_bangkok_accidents,
    load_grid_stats,
)
from ..utils.rate_limit import rate_limit
from ..utils.validators import validate_coordinates

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/risk-zones")
def risk_zones():
    try:
        df = load_grid_stats()
        df = df[df["total_accidents"] > 0].copy()
        df = df.where(pd.notna(df), None)
        return jsonify({"count": int(len(df)), "risk_zones": df.to_dict(orient="records")})
    except Exception:
        logger.exception("Error in /api/risk-zones")
        return jsonify({"error": "Failed to load risk zones."}), 500


@api_bp.route("/temporal-stats")
def temporal_stats():
    try:
        return jsonify(calculate_temporal_stats(load_bangkok_accidents()))
    except Exception:
        logger.exception("Error in /api/temporal-stats")
        return jsonify({"error": "Failed to load temporal statistics."}), 500


@api_bp.route("/analytics-data")
def analytics_data():
    try:
        payload = build_analytics_payload(load_bangkok_accidents(), load_grid_stats())
        return jsonify(payload)
    except Exception:
        logger.exception("Error in /api/analytics-data")
        return jsonify({"error": "Failed to load analytics data."}), 500


@api_bp.route("/model-info")
def model_info():
    if extensions.USE_ML and extensions.ml_predictor:
        try:
            top_features = extensions.ml_predictor.get_feature_importance(top_n=5)
        except Exception:
            top_features = {}
        return jsonify({
            "status": "success",
            "model": "xgboost",
            "version": "1.0",
            "accuracy": "83.8% holdout crash capture rate",
            "test_performance": {
                "holdout_split": "Jan-Sep 2024 train / Oct-Dec 2024 validation",
                "capture": "83.8% of later mapped crashes fell inside flagged high-risk grids",
                "lift": "17.5x over random selection",
            },
            "comparison": {
                "spatial_baseline": "83.8% capture rate",
                "xgboost": "Primary temporal holdout model",
                "improvement": "+16.2% over spatial baseline",
            },
            "top_features": top_features,
            "fallback_available": True,
            "prediction_time": "<100ms",
        })
    return jsonify({
        "status": "success",
        "model": "spatial_analysis",
        "accuracy": "83.8%",
        "method": "grid_based_counting",
        "ml_available": False,
    })


@api_bp.route("/geolocate", methods=["POST"])
@rate_limit(max_calls=10, window_seconds=3600)
def geolocate():
    """Server-side IP geolocation via Google Geolocation API.

    Rate-limited to 10 requests/hour per IP to protect API quota.
    """
    if not GOOGLE_MAPS_SERVER_KEY:
        return jsonify(success=False, error="Geolocation is not configured on this server."), 500
    try:
        resp = http_requests.post(
            f"https://www.googleapis.com/geolocation/v1/geolocate?key={GOOGLE_MAPS_SERVER_KEY}",
            json={"considerIp": True},
            timeout=5,
        )
        data = resp.json()
        if "location" not in data:
            return jsonify(success=False, error="Could not determine location."), 400

        # Validate that Google returned sensible coordinate values before
        # passing them back to the client.
        try:
            lat = float(data["location"]["lat"])
            lon = float(data["location"]["lng"])
            validate_coordinates(lat, lon)
        except (KeyError, TypeError, ValueError):
            logger.warning("Google Geolocation returned invalid coordinates: %s", data.get("location"))
            return jsonify(success=False, error="Could not determine location."), 400

        return jsonify(success=True, lat=lat, lon=lon, accuracy=data.get("accuracy", 0))

    except Exception:
        logger.exception("Error in /api/geolocate")
        return jsonify(success=False, error="Geolocation request failed."), 500


@api_bp.route("/compare-predictions", methods=["POST"])
def compare_predictions():
    try:
        data = request.get_json(silent=True) or {}
        try:
            lat = float(data["lat"])
            lon = float(data["lon"])
            validate_coordinates(lat, lon)
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Invalid coordinates in /api/compare-predictions: %s", exc)
            return jsonify({"error": "Invalid or missing coordinates."}), 400

        result: dict = {
            "location": {"lat": lat, "lon": lon},
            "ml_prediction": None,
            "spatial_prediction": None,
            "difference": None,
        }

        if extensions.USE_ML and extensions.ml_predictor:
            try:
                result["ml_prediction"] = extensions.ml_predictor.predict_risk(lat, lon)
            except Exception:
                logger.exception("ML prediction failed in /api/compare-predictions")

        if extensions.ml_predictor:
            try:
                result["spatial_prediction"] = extensions.ml_predictor.fallback_to_spatial(lat, lon)
            except Exception:
                logger.exception("Spatial prediction failed in /api/compare-predictions")

        if result["ml_prediction"] and result["spatial_prediction"]:
            ml_score = result["ml_prediction"]["risk_score"]
            sp_score = result["spatial_prediction"]["risk_score"]
            result["difference"] = {
                "absolute": ml_score - sp_score,
                "percentage": round(((ml_score - sp_score) / max(sp_score, 1)) * 100, 1),
            }

        return jsonify(result)
    except Exception:
        logger.exception("Unhandled error in /api/compare-predictions")
        return jsonify({"error": "An error occurred while comparing predictions."}), 500
