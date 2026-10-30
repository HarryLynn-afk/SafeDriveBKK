"""
Flask application factory for SafeDrive BKK.

Usage
-----
    from src.app import create_app
    app = create_app()

The WSGI entry point (root app.py) and the dev server both call create_app().
"""

import json
import logging

from flask import Flask, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import (
    BASE_DIR,
    GOOGLE_MAPS_SERVER_KEY,
    GOOGLE_MAPS_SERVER_KEY_SOURCE,
    PORT,
    SECRET_KEY,
)
from . import extensions
from .logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )

    # Trust one layer of reverse proxy (Render/Gunicorn) so request.remote_addr
    # reflects the real client IP — otherwise the rate limiter keys on the proxy
    # IP and throttles all users as one.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

    # ── Security configuration ──────────────────────────────────────────────
    app.config["SECRET_KEY"] = SECRET_KEY
    # Reject request bodies larger than 2 MB to prevent memory-exhaustion attacks.
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB

    _register_request_hooks(app)
    _init_ml()
    _log_routing_config()
    _init_chatbot()

    from .blueprints.main import main_bp
    from .blueprints.api import api_bp
    from .blueprints.routing_api import routing_api_bp
    from .blueprints.nlp_bp import nlp_bp
    from .blueprints.chatbot_bp import chatbot_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(routing_api_bp)
    app.register_blueprint(nlp_bp)
    app.register_blueprint(chatbot_bp)

    return app


# ---------------------------------------------------------------------------
# Request / response hooks
# ---------------------------------------------------------------------------

def _register_request_hooks(app: Flask) -> None:

    @app.before_request
    def enforce_json_content_type():
        """Reject non-JSON POST bodies.

        Browsers cannot send Content-Type: application/json via plain HTML
        forms, so this provides lightweight CSRF mitigation for all JSON
        API endpoints without requiring a token.
        """
        if (
            request.method == "POST"
            and request.content_length
            and request.content_length > 0
            and not request.is_json
        ):
            return jsonify({"error": "Content-Type must be application/json"}), 415

    @app.after_request
    def set_security_headers(response):
        """Attach security headers to every response."""
        is_heatmap = (
            request.endpoint == "static"
            and (request.view_args or {}).get("filename") == "bangkok_risk_heatmap.html"
        )
        # The dashboard embeds this Folium document from our own origin.
        map_scripts = " https://code.jquery.com https://cdnjs.cloudflare.com" if is_heatmap else ""
        map_styles = " https://netdna.bootstrapcdn.com https://cdnjs.cloudflare.com" if is_heatmap else ""
        map_fonts = " https://netdna.bootstrapcdn.com" if is_heatmap else ""
        frame_ancestors = "'self'" if is_heatmap else "'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN" if is_heatmap else "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Strict-Transport-Security: only sent over HTTPS (Render/Gunicorn enforces this)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # CSP: allow Maps JS, Chart.js CDN, and our own assets; block everything else.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'unsafe-inline' cdn.jsdelivr.net maps.googleapis.com{map_scripts}; "
            f"style-src 'self' 'unsafe-inline' fonts.googleapis.com cdn.jsdelivr.net{map_styles}; "
            f"font-src fonts.gstatic.com cdn.jsdelivr.net{map_fonts}; "
            f"frame-ancestors {frame_ancestors}; "
            "img-src 'self' data: https:; "
            "connect-src 'self' router.project-osrm.org *.googleapis.com https://nominatim.openstreetmap.org"
        )
        return response


# ---------------------------------------------------------------------------
# Startup initialisation helpers
# ---------------------------------------------------------------------------

def _init_ml() -> None:
    from .ml_model import RiskPredictor
    try:
        predictor = RiskPredictor(str(BASE_DIR / "models" / "xgboost_model.pkl"))
        extensions.ml_predictor = predictor
        extensions.USE_ML = predictor._model is not None
        if extensions.USE_ML:
            logger.info("ML predictor: XGBoost loaded (predictive performance not verified)")
        else:
            logger.warning("ML predictor: model file missing, spatial fallback active")
    except Exception:
        logger.exception("ML predictor: failed to load, spatial fallback active")
        extensions.ml_predictor = None
        extensions.USE_ML = False


def _log_routing_config() -> None:
    if not GOOGLE_MAPS_SERVER_KEY:
        logger.info("Google Maps: No server API key — using OSRM/estimated travel times.")
        return
    logger.info("Google Routes API: configured via %s (TRAFFIC_AWARE requests)", GOOGLE_MAPS_SERVER_KEY_SOURCE)


def _init_chatbot() -> None:
    chatbot_data_path = BASE_DIR / "static" / "chatbot_data.json"
    try:
        with open(chatbot_data_path, encoding="utf-8") as fh:
            extensions.CHATBOT_DATA = json.load(fh)
        logger.info("Chatbot data loaded from %s", chatbot_data_path)
    except FileNotFoundError:
        extensions.CHATBOT_DATA = {}
        logger.warning("Chatbot data not found at %s", chatbot_data_path)


# ---------------------------------------------------------------------------
# Dev-server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 64)
    print("SafeDrive BKK - Risk Prediction System")
    print("=" * 64)

    _app = create_app()

    if extensions.USE_ML:
        print("Prediction: XGBoost (Machine Learning)")
        print("  Evaluation: predictive performance not verified")
        print("  Features:    44-feature gradient boosting")
    else:
        print("Prediction: Spatial Analysis (Grid-based)")
        print("  Spatial baseline: 83.8% temporal crash capture (not accuracy)")

    print("-" * 64)
    print("Blueprints registered:")
    print("  main        → /, /trip-planner, /analytics, /about, /nlp, /risk-map")
    print("  api         → /api/risk-zones, /api/temporal-stats, /api/analytics-data")
    print("               /api/model-info, /api/geolocate, /api/compare-predictions")
    print("  routing_api → /api/analyze-route, /api/compare-routes")
    print("  nlp         → /api/nlp/*")
    print("  chatbot     → /api/chatbot, /api/chatbot/suggestions")
    print("=" * 64)

    port = PORT
    print(f"Starting on http://0.0.0.0:{port}")
    print()
    _app.run(debug=False, host="0.0.0.0", port=port)
