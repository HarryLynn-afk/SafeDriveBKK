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
    _init_gmaps()
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
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Strict-Transport-Security: only sent over HTTPS (Render/Gunicorn enforces this)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        # CSP: allow Maps JS, Chart.js CDN, and our own assets; block everything else.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.jsdelivr.net maps.googleapis.com; "
            "style-src 'self' 'unsafe-inline' fonts.googleapis.com cdn.jsdelivr.net; "
            "font-src fonts.gstatic.com cdn.jsdelivr.net; "
            "img-src 'self' data: https:; "
            "connect-src 'self' router.project-osrm.org *.googleapis.com"
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
            logger.info("ML predictor: XGBoost loaded (83.8% holdout crash capture, 17.5x lift)")
        else:
            logger.warning("ML predictor: model file missing, spatial fallback active")
    except Exception:
        logger.exception("ML predictor: failed to load, spatial fallback active")
        extensions.ml_predictor = None
        extensions.USE_ML = False


def _init_gmaps() -> None:
    if not GOOGLE_MAPS_SERVER_KEY:
        logger.info("Google Maps: No server API key — using OSRM/estimated travel times.")
        return
    try:
        import googlemaps
        extensions.gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_SERVER_KEY)
        logger.info("Google Maps: loaded via %s (traffic-aware routing enabled)", GOOGLE_MAPS_SERVER_KEY_SOURCE)
    except Exception:
        logger.exception("Google Maps: failed to initialise client")
        extensions.gmaps_client = None


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
        print("  Holdout:     83.8% crash capture, 17.5x lift vs random")
        print("  Improvement: +16.2% vs spatial baseline")
        print("  Features:    44-feature gradient boosting")
    else:
        print("Prediction: Spatial Analysis (Grid-based)")
        print("  Accuracy: 83.8%")

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
