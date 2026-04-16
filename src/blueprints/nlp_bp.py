"""
NLP analysis API endpoints.

Routes:
    GET  /api/nlp/pipeline-preview
    GET  /api/nlp/cause-analysis
    GET  /api/nlp/recommendations
    GET  /api/nlp/wordcloud
    GET  /api/nlp/ml-metrics
    POST /api/nlp/classify
"""

import json

from flask import Blueprint, jsonify, request

from ..config import BASE_DIR, NLP_CACHE_PATH

nlp_bp = Blueprint("nlp", __name__, url_prefix="/api/nlp")


def _load_nlp_cache() -> dict:
    try:
        if NLP_CACHE_PATH.exists():
            with open(NLP_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return {}


@nlp_bp.route("/pipeline-preview")
def pipeline_preview():
    try:
        return jsonify(_load_nlp_cache().get("pipeline_preview", {}))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@nlp_bp.route("/cause-analysis")
def cause_analysis():
    try:
        return jsonify(_load_nlp_cache().get("cause_analysis", {}))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@nlp_bp.route("/recommendations")
def recommendations():
    try:
        return jsonify(_load_nlp_cache().get("recommendations", []))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@nlp_bp.route("/wordcloud")
def wordcloud():
    try:
        return jsonify(_load_nlp_cache().get("word_cloud", {}))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@nlp_bp.route("/ml-metrics")
def ml_metrics():
    try:
        metrics_path = BASE_DIR / "data" / "processed" / "nlp_metrics.json"
        if not metrics_path.exists():
            return jsonify({"error": "Metrics not available"}), 404
        with open(metrics_path, encoding="utf-8") as f:
            return jsonify(json.load(f))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@nlp_bp.route("/classify", methods=["POST"])
def classify():
    try:
        # Lazy import: avoid paying the pythainlp startup cost until first use
        from ..nlp_analysis import get_cause_classifier

        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        return jsonify(get_cause_classifier().classify(text))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
