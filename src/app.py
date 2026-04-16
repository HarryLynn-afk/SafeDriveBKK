"""
Flask application factory for SafeDrive BKK.

Usage
-----
    from src.app import create_app
    app = create_app()

The WSGI entry point (root app.py) and the dev server both call create_app().
"""

import json
import os

from flask import Flask

from .config import BASE_DIR, GOOGLE_MAPS_SERVER_KEY
from . import extensions


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )

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
# Startup initialisation helpers
# ---------------------------------------------------------------------------

def _init_ml() -> None:
    from .ml_model import RiskPredictor
    try:
        predictor = RiskPredictor(str(BASE_DIR / "models" / "xgboost_model.pkl"))
        extensions.ml_predictor = predictor
        extensions.USE_ML = predictor._model is not None
        if extensions.USE_ML:
            print("ML predictor: XGBoost loaded (83.8% holdout crash capture, 17.5x lift)")
        else:
            print("ML predictor: model file missing, spatial fallback active")
    except Exception as exc:
        print(f"ML predictor: failed to load ({exc}), spatial fallback active")
        extensions.ml_predictor = None
        extensions.USE_ML = False


def _init_gmaps() -> None:
    if not GOOGLE_MAPS_SERVER_KEY:
        print("Google Maps: No server API key found. Using OSRM/estimated travel times.")
        return
    try:
        import googlemaps
        extensions.gmaps_client = googlemaps.Client(key=GOOGLE_MAPS_SERVER_KEY)
        key_source = "GOOGLE_MAPS_SERVER_KEY" if os.getenv("GOOGLE_MAPS_SERVER_KEY") else "legacy GOOGLE_MAPS_API_KEY"
        print(f"Google Maps: loaded via {key_source} (traffic-aware travel times enabled)")
    except Exception as exc:
        print(f"Google Maps: failed to initialise client ({exc})")
        extensions.gmaps_client = None


def _init_chatbot() -> None:
    chatbot_data_path = BASE_DIR / "static" / "chatbot_data.json"
    try:
        with open(chatbot_data_path, encoding="utf-8") as fh:
            extensions.CHATBOT_DATA = json.load(fh)
        print(f"Chatbot data loaded: {chatbot_data_path}")
    except FileNotFoundError:
        extensions.CHATBOT_DATA = {}
        print(f"Warning: chatbot data not found at {chatbot_data_path}")


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

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting on http://0.0.0.0:{port}")
    print()
    _app.run(debug=False, host="0.0.0.0", port=port)
