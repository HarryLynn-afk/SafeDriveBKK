"""
Flask web app for Bangkok Road Risk system.

Foundation version: basic routes + ML model loading + core API endpoints.
Trip Planner, Analytics, and About pages will be built during the hackathon.
NLP Analysis and Object Detection tracks are pre-built and serve from cache.
"""

import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, send_file

from ml_model import RiskPredictor

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

BANGKOK_ACCIDENTS_PATH = DATA_DIR / "bangkok_accidents_clean.csv"
GRID_STATS_PATH = DATA_DIR / "bangkok_grid_stats.csv"
RISK_HEATMAP_HTML_PATH = BASE_DIR / "static" / "bangkok_risk_heatmap.html"

BKK_LAT_MIN = 13.6
BKK_LAT_MAX = 13.95
BKK_LON_MIN = 100.35
BKK_LON_MAX = 100.9
CELL_SIZE_DEG = 0.009

DAY_NAME_MAP = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

# ---------------------------------------------------------------------------
# ML predictor — initialised once at import time
# ---------------------------------------------------------------------------
try:
    ml_predictor = RiskPredictor(str(BASE_DIR / "models" / "xgboost_model.pkl"))
    USE_ML = ml_predictor._model is not None
    if USE_ML:
        print("ML predictor: XGBoost loaded (39/39 zones, 17.5x lift)")
    else:
        print("ML predictor: model file missing, spatial fallback active")
except Exception as exc:
    print(f"ML predictor: failed to load ({exc}), spatial fallback active")
    ml_predictor = None
    USE_ML = False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _require_file(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


def _load_bangkok_accidents():
    _require_file(BANGKOK_ACCIDENTS_PATH)
    return pd.read_csv(BANGKOK_ACCIDENTS_PATH, encoding="utf-8-sig")


def _load_grid_stats():
    _require_file(GRID_STATS_PATH)
    df = pd.read_csv(GRID_STATS_PATH, encoding="utf-8-sig")
    if "grid_id" not in df.columns:
        raise ValueError("Invalid grid stats file: missing 'grid_id' column")
    return _add_grid_bounds(df)


def _add_grid_bounds(grid_df):
    required = {"lat_min", "lat_max", "lon_min", "lon_max"}
    if required.issubset(grid_df.columns):
        return grid_df.copy()

    parsed = grid_df["grid_id"].astype(str).str.extract(r"^G(\d{3})_(\d{3})$")
    if parsed.isna().any().any():
        raise ValueError("Invalid grid_id format; expected G000_000 style values")

    lat_idx = parsed[0].astype(int)
    lon_idx = parsed[1].astype(int)

    result = grid_df.copy()
    result["lat_min"] = BKK_LAT_MIN + (lat_idx * CELL_SIZE_DEG)
    result["lat_max"] = (result["lat_min"] + CELL_SIZE_DEG).clip(upper=BKK_LAT_MAX)
    result["lon_min"] = BKK_LON_MIN + (lon_idx * CELL_SIZE_DEG)
    result["lon_max"] = (result["lon_min"] + CELL_SIZE_DEG).clip(upper=BKK_LON_MAX)
    return result


def _format_hour_range(hour):
    start_hour = int(hour)
    end_hour = (start_hour + 1) % 24
    return f"{start_hour:02d}:00-{end_hour:02d}:00"


def _calculate_temporal_stats(df):
    hour_counts = (
        pd.to_numeric(df["hour"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_values(ascending=False)
    )
    peak_hours = [
        {
            "hour": int(hour),
            "time_range": _format_hour_range(hour),
            "count": int(count),
        }
        for hour, count in hour_counts.head(3).items()
    ]

    day_counts = (
        pd.to_numeric(df["day_of_week"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_values(ascending=False)
    )
    dangerous_days = [
        {
            "day_index": int(day),
            "day_name": DAY_NAME_MAP.get(int(day), str(day)),
            "count": int(count),
        }
        for day, count in day_counts.head(3).items()
    ]

    return {
        "peak_hours": peak_hours,
        "dangerous_days": dangerous_days,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    try:
        accidents_df = _load_bangkok_accidents()
        grid_df = _load_grid_stats()
        temporal_stats = _calculate_temporal_stats(accidents_df)

        total_accidents = int(len(accidents_df))
        high_risk_zones = int(
            ((grid_df["risk_level"].str.lower() == "high") & (grid_df["total_accidents"] > 0)).sum()
        )
        peak_time = temporal_stats["peak_hours"][0]["time_range"] if temporal_stats["peak_hours"] else "N/A"
        dangerous_day = temporal_stats["dangerous_days"][0]["day_name"] if temporal_stats["dangerous_days"] else "N/A"
        map_available = RISK_HEATMAP_HTML_PATH.exists()

        return render_template(
            "index.html",
            total_accidents=total_accidents,
            high_risk_zones=high_risk_zones,
            peak_time=peak_time,
            dangerous_day=dangerous_day,
            map_available=map_available,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            total_accidents=0,
            high_risk_zones=0,
            peak_time="N/A",
            dangerous_day="N/A",
            map_available=False,
            error_message=str(exc),
        ), 500


@app.route("/heatmap")
def heatmap():
    """Interactive risk heatmap page."""
    return render_template("heatmap.html")


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.route("/api/grid-stats")
def grid_stats():
    """Return grid statistics as JSON."""
    try:
        df = _load_grid_stats()
        df = df[df["total_accidents"] > 0].copy()
        df = df.where(pd.notna(df), None)
        return jsonify({"count": int(len(df)), "grid_stats": df.to_dict(orient="records")})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/heatmap-data")
def heatmap_data():
    """Return pre-generated heatmap data."""
    try:
        heatmap_path = BASE_DIR / "static" / "heatmap_data.json"
        _require_file(heatmap_path)
        with open(heatmap_path) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/model-info")
def model_info():
    """Return information about the active prediction model."""
    if USE_ML and ml_predictor:
        try:
            top_features = ml_predictor.get_feature_importance(top_n=5)
        except Exception:
            top_features = {}

        return jsonify({
            "status": "success",
            "model": "xgboost",
            "version": "1.0",
            "zone_detection": "39/39 zones detected",
            "spatial_capture_rate": "83.8%",
            "lift": "17.5x over random selection",
            "top_features": top_features,
        })

    return jsonify({
        "status": "success",
        "model": "spatial_analysis",
        "spatial_capture_rate": "83.8%",
        "method": "grid_based_counting",
        "ml_available": False,
    })


# ---------------------------------------------------------------------------
# NLP Analysis (Track 1) — serves from cache
# ---------------------------------------------------------------------------
NLP_CACHE_PATH = BASE_DIR / "static" / "nlp_cache.json"


def _load_nlp_cache():
    if NLP_CACHE_PATH.exists():
        with open(NLP_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.route("/nlp")
def nlp():
    """NLP analysis page — Thai text → XGBoost features → recommendations."""
    nlp_data = _load_nlp_cache()
    return render_template("nlp.html", nlp_data=nlp_data)


@app.route("/api/nlp/pipeline-demo")
def nlp_pipeline_demo():
    """Pipeline examples: Thai text → tokenization → XGBoost feature."""
    cache = _load_nlp_cache()
    return jsonify(cache.get("pipeline_demo", {}))


@app.route("/api/nlp/cause-analysis")
def nlp_cause_analysis():
    """Accident cause frequency breakdown."""
    cache = _load_nlp_cache()
    return jsonify(cache.get("cause_analysis", {}))


@app.route("/api/nlp/recommendations")
def nlp_recommendations():
    """Per-zone policy recommendations based on NLP-derived causes."""
    cache = _load_nlp_cache()
    return jsonify(cache.get("recommendations", []))


@app.route("/api/nlp/wordcloud")
def nlp_wordcloud():
    """Thai keyword frequencies for word cloud visualization."""
    cache = _load_nlp_cache()
    return jsonify(cache.get("word_cloud", {}))


# ---------------------------------------------------------------------------
# Object Detection (Track 3) — serves from cache
# ---------------------------------------------------------------------------
DETECTION_CACHE_PATH = BASE_DIR / "static" / "detection_cache.json"


def _load_detection_cache():
    if DETECTION_CACHE_PATH.exists():
        with open(DETECTION_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    # Return skeleton with pipeline data even without images
    return {
        "image_results": [],
        "comparisons": [],
        "vehicle_importance": _get_vehicle_importance_fallback(),
        "pipeline": _get_pipeline_fallback(),
        "total_images": 0,
    }


def _get_vehicle_importance_fallback():
    """Vehicle feature importance from CSV when no detection cache exists."""
    try:
        imp_path = BASE_DIR / "models" / "xgb_feature_importance.csv"
        if not imp_path.exists():
            return {}
        imp = pd.read_csv(imp_path)
        imp_dict = dict(zip(imp["feature"], imp["importance"]))
        vehicle_features = [
            "pickup_count", "pickup_pct", "motorcycle_count", "car_count",
            "motorcycle_pct", "car_pct", "heavy_truck_count", "heavy_truck_pct",
            "max_vehicles",
        ]
        features = []
        total = 0
        for f in vehicle_features:
            v = imp_dict.get(f, 0)
            total += v
            features.append({
                "feature": f,
                "importance_pct": round(v * 100, 3),
                "yolo_detectable": f not in ("pickup_count", "pickup_pct", "max_vehicles"),
            })
        features.sort(key=lambda x: x["importance_pct"], reverse=True)
        return {
            "vehicle_features": features,
            "total_vehicle_importance_pct": round(total * 100, 2),
            "highlight": "pickup_count is #4 most important feature overall at 7.39%",
            "limitation": "COCO has no pickup class; pickups classified as car/truck by YOLO",
        }
    except Exception:
        return {}


def _get_pipeline_fallback():
    """Static pipeline diagram data."""
    return {
        "stages": [
            {"stage": 1, "title": "Camera Feed", "description": "CCTV camera at Bangkok intersection", "icon": "camera"},
            {"stage": 2, "title": "YOLOv8 Detection", "description": "Process frame, detect vehicles by type", "icon": "eye",
             "features_produced": ["car_count", "motorcycle_count", "heavy_truck_count"]},
            {"stage": 3, "title": "Feature Aggregation", "description": "Aggregate counts per grid cell per time window", "icon": "grid",
             "features_produced": ["motorcycle_pct", "car_pct", "heavy_truck_pct"]},
            {"stage": 4, "title": "XGBoost Prediction", "description": "Feed vehicle features into trained model", "icon": "brain"},
            {"stage": 5, "title": "Dynamic Risk Map", "description": "Real-time risk score updates per zone", "icon": "map"},
        ],
        "key_message": "No model retraining needed. YOLO provides fresher values for features the model already understands.",
    }


@app.route("/detection")
def detection():
    """Object detection page — YOLO vehicle detection mapped to XGBoost features."""
    detection_data = _load_detection_cache()
    return render_template("detection.html", detection_data=detection_data)


@app.route("/api/detection/results")
def detection_results():
    """All detection results from cache."""
    cache = _load_detection_cache()
    return jsonify({
        "image_results": cache.get("image_results", []),
        "total_images": cache.get("total_images", 0),
    })


@app.route("/api/detection/compare/<grid_id>")
def detection_compare(grid_id):
    """YOLO vs model comparison for a specific grid zone."""
    cache = _load_detection_cache()
    for comp in cache.get("comparisons", []):
        if comp.get("grid_id") == grid_id:
            return jsonify(comp)
    return jsonify({"error": f"No comparison data for grid {grid_id}"}), 404


@app.route("/api/detection/pipeline")
def detection_pipeline():
    """Future vision pipeline diagram data."""
    cache = _load_detection_cache()
    return jsonify(cache.get("pipeline", {}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print()
    print("=" * 64)
    print("RoadRisk Bangkok - Foundation")
    print("=" * 64)

    if USE_ML:
        print("Prediction: XGBoost (Machine Learning)")
        print("  Detection:   39/39 danger zones")
        print("  Capture:     83.8% spatial capture rate")
        print("  Lift:        17.5x over random")
    else:
        print("Prediction: Spatial Analysis (Grid-based)")
        print("  Capture: 83.8% spatial capture rate")

    print("-" * 64)
    print("Endpoints:")
    print("  GET  /                          Home dashboard")
    print("  GET  /heatmap                   Risk heatmap")
    print("  GET  /nlp                       NLP analysis (Track 1)")
    print("  GET  /detection                 Object detection (Track 3)")
    print("  GET  /api/grid-stats            Grid statistics")
    print("  GET  /api/heatmap-data          Heatmap data")
    print("  GET  /api/model-info            ML model details")
    print("  GET  /api/nlp/*                 NLP endpoints")
    print("  GET  /api/detection/*           Detection endpoints")
    print("=" * 64)
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting on http://0.0.0.0:{port}")
    print()
    app.run(debug=False, host="0.0.0.0", port=port)
