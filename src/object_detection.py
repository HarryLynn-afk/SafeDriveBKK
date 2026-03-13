"""
Object Detection Module for Bangkok Road Risk.

Uses YOLOv8n to detect vehicles in sample traffic images, maps detections
to the same feature names the XGBoost model uses, and provides validation
narratives connecting visual evidence to model predictions.
"""

import json
import logging
from pathlib import Path

import cv2
import pandas as pd

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

ML_FEATURES_PATH = DATA_DIR / "ml_features.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "xgb_feature_importance.csv"
SAMPLE_IMAGES_DIR = STATIC_DIR / "sample_images"
DETECTIONS_DIR = STATIC_DIR / "detections"
DETECTION_CACHE_PATH = STATIC_DIR / "detection_cache.json"
METADATA_PATH = SAMPLE_IMAGES_DIR / "metadata.json"

# Map COCO class IDs to our XGBoost feature names
COCO_TO_XGBOOST = {
    2: {"coco_name": "car", "xgboost_feature": "car_count", "pct_feature": "car_pct"},
    3: {
        "coco_name": "motorcycle",
        "xgboost_feature": "motorcycle_count",
        "pct_feature": "motorcycle_pct",
    },
    5: {"coco_name": "bus", "xgboost_feature": None, "pct_feature": None},
    7: {
        "coco_name": "truck",
        "xgboost_feature": "heavy_truck_count",
        "pct_feature": "heavy_truck_pct",
    },
    0: {"coco_name": "person", "xgboost_feature": None, "pct_feature": None},
}

VEHICLE_CLASS_IDS = {0, 2, 3, 5, 7}


class VehicleDetector:
    """Detects vehicles using YOLOv8n and maps to XGBoost features."""

    def __init__(self):
        from ultralytics import YOLO

        self._model = YOLO("yolov8n.pt")
        self._ml_features = pd.read_csv(ML_FEATURES_PATH, encoding="utf-8-sig")
        self._importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        self._importance_dict = dict(
            zip(self._importance["feature"], self._importance["importance"])
        )
        LOGGER.info("VehicleDetector initialized with YOLOv8n")

    def detect_vehicles(self, image_path):
        """Run YOLO inference on an image and return structured results."""
        image_path = Path(image_path)
        results = self._model(str(image_path), verbose=False)[0]

        vehicle_counts = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0, "person": 0}
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASS_IDS:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_name = COCO_TO_XGBOOST.get(cls_id, {}).get("coco_name", "unknown")

            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
            detections.append(
                {
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                }
            )

        # Save annotated image
        DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        annotated = results.plot()
        out_path = DETECTIONS_DIR / image_path.name
        cv2.imwrite(str(out_path), annotated)

        total = sum(v for k, v in vehicle_counts.items() if k != "person")

        return {
            "image_name": image_path.name,
            "detections": detections,
            "vehicle_counts": vehicle_counts,
            "total_vehicles": total,
            "annotated_image": f"detections/{image_path.name}",
        }

    def map_to_xgboost_features(self, vehicle_counts):
        """Convert YOLO counts to XGBoost feature format."""
        total = sum(v for k, v in vehicle_counts.items() if k != "person")
        if total == 0:
            total = 1  # avoid division by zero

        return {
            "motorcycle_count": vehicle_counts.get("motorcycle", 0),
            "car_count": vehicle_counts.get("car", 0),
            "heavy_truck_count": vehicle_counts.get("truck", 0),
            "motorcycle_pct": round(
                vehicle_counts.get("motorcycle", 0) / total * 100, 1
            ),
            "car_pct": round(vehicle_counts.get("car", 0) / total * 100, 1),
            "heavy_truck_pct": round(
                vehicle_counts.get("truck", 0) / total * 100, 1
            ),
            "pickup_count": 0,  # COCO has no pickup class - honest limitation
            "pickup_pct": 0,
            "_note": "COCO dataset has no 'pickup' class. Pickups may appear as car or truck.",
        }

    def compare_with_model(self, image_path, grid_id):
        """Side-by-side: YOLO snapshot vs XGBoost historical data for a zone."""
        detection = self.detect_vehicles(image_path)
        yolo_features = self.map_to_xgboost_features(detection["vehicle_counts"])

        # Get historical model data for this grid
        grid_row = self._ml_features[self._ml_features["grid_id"] == grid_id]
        if grid_row.empty:
            return {
                "error": f"Grid {grid_id} not found",
                "detection": detection,
                "yolo_features": yolo_features,
            }

        row = grid_row.iloc[0]
        model_features = {
            "motorcycle_count": int(row.get("motorcycle_count", 0)),
            "car_count": int(row.get("car_count", 0)),
            "heavy_truck_count": int(row.get("heavy_truck_count", 0)),
            "pickup_count": int(row.get("pickup_count", 0)),
            "motorcycle_pct": round(float(row.get("motorcycle_pct", 0)), 1),
            "car_pct": round(float(row.get("car_pct", 0)), 1),
            "heavy_truck_pct": round(float(row.get("heavy_truck_pct", 0)), 1),
            "pickup_pct": round(float(row.get("pickup_pct", 0)), 1),
        }

        # Build narrative
        yolo_mc_pct = yolo_features["motorcycle_pct"]
        model_mc_pct = model_features["motorcycle_pct"]
        risk_score = int(row.get("total_accidents", 0))

        narrative = (
            f"YOLO detected {yolo_mc_pct}% motorcycles in this snapshot. "
            f"Historical data shows {model_mc_pct}% for zone {grid_id}. "
            f"This zone has {risk_score} recorded accidents."
        )

        return {
            "grid_id": grid_id,
            "detection": detection,
            "yolo_features": yolo_features,
            "model_features": model_features,
            "total_accidents": risk_score,
            "dangerous": bool(row.get("dangerous", 0)),
            "narrative": narrative,
        }

    def get_vehicle_feature_importance(self):
        """Return importance of all vehicle-related features in XGBoost."""
        vehicle_features = [
            "pickup_count",
            "pickup_pct",
            "motorcycle_count",
            "car_count",
            "motorcycle_pct",
            "car_pct",
            "heavy_truck_count",
            "heavy_truck_pct",
            "max_vehicles",
        ]

        features = []
        total_importance = 0
        for f in vehicle_features:
            imp = self._importance_dict.get(f, 0)
            total_importance += imp
            features.append(
                {
                    "feature": f,
                    "importance_pct": round(imp * 100, 3),
                    "yolo_detectable": f
                    not in ("pickup_count", "pickup_pct", "max_vehicles"),
                }
            )

        features.sort(key=lambda x: x["importance_pct"], reverse=True)

        return {
            "vehicle_features": features,
            "total_vehicle_importance_pct": round(total_importance * 100, 2),
            "highlight": "pickup_count is #4 most important feature overall at 7.39%",
            "limitation": "COCO has no pickup class; pickups classified as car/truck by YOLO",
        }

    def build_realtime_pipeline_diagram(self):
        """Structured data for the future-vision pipeline diagram."""
        return {
            "stages": [
                {
                    "stage": 1,
                    "title": "Camera Feed",
                    "description": "CCTV camera at Bangkok intersection",
                    "icon": "camera",
                },
                {
                    "stage": 2,
                    "title": "YOLOv8 Detection",
                    "description": "Process frame, detect vehicles by type",
                    "icon": "eye",
                    "features_produced": [
                        "car_count",
                        "motorcycle_count",
                        "heavy_truck_count",
                    ],
                },
                {
                    "stage": 3,
                    "title": "Feature Aggregation",
                    "description": "Aggregate counts per grid cell per time window",
                    "icon": "grid",
                    "features_produced": [
                        "motorcycle_pct",
                        "car_pct",
                        "heavy_truck_pct",
                    ],
                },
                {
                    "stage": 4,
                    "title": "XGBoost Prediction",
                    "description": "Feed vehicle features into trained model",
                    "icon": "brain",
                    "note": "Same 44-feature model, just fresher vehicle data",
                },
                {
                    "stage": 5,
                    "title": "Dynamic Risk Map",
                    "description": "Real-time risk score updates per zone",
                    "icon": "map",
                },
            ],
            "key_message": "No model retraining needed. YOLO provides fresher values for features the model already understands.",
        }

    def get_vehicle_accident_analysis(self):
        """Compute motorcycle % vs severity correlation from historical accident data."""
        grid_stats_path = BASE_DIR / "data" / "processed" / "bangkok_grid_stats.csv"
        try:
            grid_stats = pd.read_csv(grid_stats_path, encoding="utf-8-sig")
            stats_dict = dict(zip(grid_stats["grid_id"], grid_stats["risk_score"]))
        except Exception:
            stats_dict = {}

        dangerous = self._ml_features[self._ml_features["dangerous"] == 1].copy()

        # Bucket zones by motorcycle_pct
        buckets = [
            ("<30%", 0, 30),
            ("30–50%", 30, 50),
            ("50–70%", 50, 70),
            (">70%", 70, 101),
        ]
        correlation = []
        for label, low, high in buckets:
            subset = dangerous[
                (dangerous["motorcycle_pct"] >= low) & (dangerous["motorcycle_pct"] < high)
            ]
            correlation.append({
                "bucket": label,
                "zone_count": int(len(subset)),
                "avg_accidents": round(float(subset["total_accidents"].mean()), 1) if len(subset) else 0,
                "avg_severity": round(float(subset["severity_mean"].mean()), 2) if len(subset) else 0,
            })

        # Key finding: compare >70% vs <30%
        top = next((c for c in correlation if c["bucket"] == ">70%"), None)
        bot = next((c for c in correlation if c["bucket"] == "<30%"), None)
        if top and bot and bot["avg_accidents"] > 0:
            ratio = round(top["avg_accidents"] / bot["avg_accidents"], 1)
            key_finding = (
                f"Zones with >70% motorcycle share average {top['avg_accidents']} accidents "
                f"vs {bot['avg_accidents']} for <30% — a {ratio}× difference."
            )
        else:
            key_finding = "High motorcycle share correlates with increased accident counts."

        # Top 8 dangerous zones by motorcycle_pct
        top_zones = dangerous.nlargest(8, "motorcycle_pct")[
            ["grid_id", "motorcycle_pct", "total_accidents"]
        ].copy()
        top_zones["risk_score"] = top_zones["grid_id"].map(stats_dict).fillna(0)
        high_motorcycle_zones = [
            {
                "grid_id": row["grid_id"],
                "motorcycle_pct": round(float(row["motorcycle_pct"]), 1),
                "total_accidents": int(row["total_accidents"]),
                "risk_score": round(float(row["risk_score"]), 1),
            }
            for _, row in top_zones.iterrows()
        ]

        return {
            "correlation": correlation,
            "key_finding": key_finding,
            "high_motorcycle_zones": high_motorcycle_zones,
        }

    def generate_cache(self, image_dir=None):
        """Process all sample images and save cache."""
        if image_dir is None:
            image_dir = SAMPLE_IMAGES_DIR

        image_dir = Path(image_dir)
        if not image_dir.exists():
            LOGGER.warning("Sample images directory not found: %s", image_dir)
            return {}

        # Load metadata for grid_id mapping
        metadata = []
        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                metadata = json.load(f)

        metadata_map = {m["image"]: m for m in metadata}

        # Load risk scores for grid_id enrichment
        grid_stats_path = DATA_DIR / "bangkok_grid_stats.csv"
        stats_dict = {}
        if grid_stats_path.exists():
            stats_df = pd.read_csv(grid_stats_path)
            stats_dict = dict(zip(stats_df["grid_id"], stats_df["risk_score"]))

        # Process each image
        image_results = []
        comparisons = []

        for img_path in sorted(image_dir.glob("*.jpg")) + sorted(
            image_dir.glob("*.png")
        ):
            LOGGER.info("Processing %s...", img_path.name)
            detection = self.detect_vehicles(img_path)
            yolo_features = self.map_to_xgboost_features(
                detection["vehicle_counts"]
            )
            detection["xgboost_features"] = yolo_features

            # Enrich with metadata (grid_id, description, risk_score)
            meta = metadata_map.get(img_path.name, {})
            grid_id = meta.get("grid_id", "")
            detection["grid_id"] = grid_id
            detection["description"] = meta.get("description", "")
            detection["risk_score"] = round(float(stats_dict.get(grid_id, 0)), 1) if grid_id else 0
            image_results.append(detection)

            # If we have a grid_id mapping, do comparison
            if grid_id:
                comparison = self.compare_with_model(img_path, grid_id)
                comparison["description"] = meta.get("description", "")
                comparisons.append(comparison)

        cache = {
            "image_results": image_results,
            "comparisons": comparisons,
            "vehicle_importance": self.get_vehicle_feature_importance(),
            "vehicle_analysis": self.get_vehicle_accident_analysis(),
            "pipeline": self.build_realtime_pipeline_diagram(),
            "total_images": len(image_results),
        }

        DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(DETECTION_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        LOGGER.info(
            "Detection cache saved to %s (%d images processed)",
            DETECTION_CACHE_PATH,
            len(image_results),
        )
        return cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    detector = VehicleDetector()
    cache = detector.generate_cache()
    print(f"Cache generated: {cache.get('total_images', 0)} images processed")
