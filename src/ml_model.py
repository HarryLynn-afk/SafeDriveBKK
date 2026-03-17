"""
ML-powered risk predictor for the Flask app.

Wraps the trained XGBoost model behind a simple predict_risk / predict_batch
interface.  Falls back to spatial accident-count scoring when the model file
is missing or a grid lookup fails.
"""

import json
import logging
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid / spatial constants  (must match grid_aggregation & feature_engineering)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 13.6, 13.95
LON_MIN, LON_MAX = 100.35, 100.9
CELL_SIZE_DEG = 0.009

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

FEATURE_NAMES_PATH = DATA_DIR / "feature_info.json"
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
ML_FEATURES_PATH = DATA_DIR / "ml_features.csv"
GRID_STATS_PATH = DATA_DIR / "bangkok_grid_stats.csv"

_GRID_ID_RE = re.compile(r"^G(\d{3})_(\d{3})$")


class RiskPredictor:
    """ML-powered risk predictor using XGBoost model.

    Provides risk scores and levels for route segments.
    """

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = MODEL_DIR / "xgboost_model.pkl"
        else:
            model_path = Path(model_path)

        # Feature ordering expected by the model
        self._feature_names = self._load_feature_names()

        # Pre-built lookup: grid_id -> feature dict
        self._grid_features = self._load_grid_features()

        # Grid stats for spatial fallback
        self._grid_stats = self._load_grid_stats()

        # Try loading model; fall back to spatial-only if unavailable
        self._model = None
        if model_path.exists():
            try:
                self._model = joblib.load(model_path)
                LOGGER.info("Loaded XGBoost model from %s", model_path)
            except Exception:
                LOGGER.exception("Failed to load model from %s", model_path)
        else:
            LOGGER.warning("Model file not found at %s — spatial fallback only", model_path)

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _load_feature_names(self):
        """Read feature order from X_train.csv header (matches training order)."""
        try:
            return list(pd.read_csv(X_TRAIN_PATH, nrows=0).columns)
        except Exception:
            LOGGER.warning("X_train.csv unavailable, falling back to feature_info.json")
        try:
            with open(FEATURE_NAMES_PATH) as f:
                return json.load(f)["feature_names"]
        except Exception:
            LOGGER.exception("Could not load feature names")
            return []

    def _load_grid_features(self):
        """Load ml_features.csv and index by grid_id for O(1) lookup."""
        lookup = {}
        try:
            df = pd.read_csv(ML_FEATURES_PATH)
            for _, row in df.iterrows():
                gid = row["grid_id"]
                lookup[gid] = row
        except Exception:
            LOGGER.exception("Could not load ml_features.csv")
        return lookup

    def _load_grid_stats(self):
        """Load bangkok_grid_stats.csv with bounds for spatial fallback."""
        try:
            df = pd.read_csv(GRID_STATS_PATH)
            parsed = df["grid_id"].str.extract(r"^G(\d{3})_(\d{3})$")
            lat_idx = parsed[0].astype(int)
            lon_idx = parsed[1].astype(int)
            df["lat_min"] = LAT_MIN + lat_idx * CELL_SIZE_DEG
            df["lat_max"] = (df["lat_min"] + CELL_SIZE_DEG).clip(upper=LAT_MAX)
            df["lon_min"] = LON_MIN + lon_idx * CELL_SIZE_DEG
            df["lon_max"] = (df["lon_min"] + CELL_SIZE_DEG).clip(upper=LON_MAX)
            return df
        except Exception:
            LOGGER.exception("Could not load grid stats")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Grid helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _latlon_to_grid_id(lat, lon):
        """Convert lat/lon to the matching grid_id string (or None)."""
        if lat < LAT_MIN or lat > LAT_MAX or lon < LON_MIN or lon > LON_MAX:
            return None
        lat_idx = int((lat - LAT_MIN) / CELL_SIZE_DEG)
        lon_idx = int((lon - LON_MIN) / CELL_SIZE_DEG)
        # Clamp to valid indices
        max_lat_idx = int((LAT_MAX - LAT_MIN) / CELL_SIZE_DEG) - 1
        max_lon_idx = int((LON_MAX - LON_MIN) / CELL_SIZE_DEG) - 1
        lat_idx = min(lat_idx, max_lat_idx)
        lon_idx = min(lon_idx, max_lon_idx)
        return f"G{lat_idx:03d}_{lon_idx:03d}"

    def _extract_features(self, grid_id):
        """Return a 1-row DataFrame with the 44 model features for *grid_id*.

        Returns None when the grid has no feature data.
        """
        row = self._grid_features.get(grid_id)
        if row is None:
            return None

        values = []
        for feat in self._feature_names:
            val = row.get(feat, np.nan)
            values.append(float(val) if not pd.isna(val) else np.nan)

        return pd.DataFrame([values], columns=self._feature_names)

    @staticmethod
    def _probability_to_risk_score(prob):
        """Map model probability (0-1) to a 0-100 risk score."""
        return min(100, max(0, int(round(prob * 100))))

    @staticmethod
    def _score_to_level(score):
        if score <= 30:
            return "low"
        if score <= 70:
            return "medium"
        return "high"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_risk(self, lat, lon, hour=12, day_of_week=1):
        """Predict risk for a single location.

        Returns a dict with risk_score (0-100), risk_level, confidence, and
        the method used ('xgboost' or 'spatial_fallback').
        """
        grid_id = self._latlon_to_grid_id(lat, lon)

        # No grid match → return a cautious medium with low confidence
        if grid_id is None:
            return {
                "risk_score": 35,
                "risk_level": "medium",
                "confidence": 0.1,
                "method": "out_of_bounds",
                "grid_id": None,
            }

        # Try ML prediction
        if self._model is not None:
            features_df = self._extract_features(grid_id)
            if features_df is not None:
                try:
                    prob = float(self._model.predict_proba(features_df)[0, 1])
                    score = self._probability_to_risk_score(prob)
                    return {
                        "risk_score": score,
                        "risk_level": self._score_to_level(score),
                        "confidence": round(prob if prob >= 0.5 else 1.0 - prob, 4),
                        "method": "xgboost",
                        "grid_id": grid_id,
                    }
                except Exception:
                    LOGGER.exception("XGBoost prediction failed for %s", grid_id)

        # Fallback
        return self.fallback_to_spatial(lat, lon)

    def predict_batch(self, locations):
        """Predict risk for multiple locations efficiently.

        *locations*: list of (lat, lon) or (lat, lon, hour, day) tuples.
        Returns a list of prediction dicts.
        """
        if not locations:
            return []

        # Separate into ML-eligible and fallback buckets
        grid_ids = []
        feature_rows = []
        indices_ml = []  # position in the output list
        results = [None] * len(locations)

        for i, loc in enumerate(locations):
            lat, lon = float(loc[0]), float(loc[1])
            grid_id = self._latlon_to_grid_id(lat, lon)

            if grid_id is None:
                results[i] = {
                    "risk_score": 35,
                    "risk_level": "medium",
                    "confidence": 0.1,
                    "method": "out_of_bounds",
                    "grid_id": None,
                }
                continue

            if self._model is not None:
                feat_df = self._extract_features(grid_id)
                if feat_df is not None:
                    feature_rows.append(feat_df.iloc[0].values)
                    grid_ids.append(grid_id)
                    indices_ml.append(i)
                    continue

            # Fallback for this location
            results[i] = self.fallback_to_spatial(lat, lon)

        # Vectorised ML prediction for the batch
        if feature_rows and self._model is not None:
            try:
                batch_df = pd.DataFrame(feature_rows, columns=self._feature_names)
                probas = self._model.predict_proba(batch_df)[:, 1]
                for j, idx in enumerate(indices_ml):
                    prob = float(probas[j])
                    score = self._probability_to_risk_score(prob)
                    results[idx] = {
                        "risk_score": score,
                        "risk_level": self._score_to_level(score),
                        "confidence": round(prob if prob >= 0.5 else 1.0 - prob, 4),
                        "method": "xgboost",
                        "grid_id": grid_ids[j],
                    }
            except Exception:
                LOGGER.exception("Batch XGBoost prediction failed")
                for j, idx in enumerate(indices_ml):
                    lat, lon = float(locations[idx][0]), float(locations[idx][1])
                    results[idx] = self.fallback_to_spatial(lat, lon)

        return results

    def get_feature_importance(self, top_n=10):
        """Return top-N feature importances from the loaded model."""
        if self._model is None:
            return {}
        importances = self._model.feature_importances_
        pairs = sorted(
            zip(self._feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: round(float(val), 5) for name, val in pairs[:top_n]}

    def fallback_to_spatial(self, lat, lon):
        """Simple spatial fallback using raw accident counts."""
        grid_id = self._latlon_to_grid_id(lat, lon)

        if grid_id is not None and not self._grid_stats.empty:
            match = self._grid_stats[self._grid_stats["grid_id"] == grid_id]
            if not match.empty:
                accidents = int(match.iloc[0]["total_accidents"])
                score = min(100, int(accidents * 2.0))
                return {
                    "risk_score": score,
                    "risk_level": self._score_to_level(score),
                    "confidence": 0.4,
                    "method": "spatial_fallback",
                    "grid_id": grid_id,
                }

        return {
            "risk_score": 35,
            "risk_level": "medium",
            "confidence": 0.1,
            "method": "spatial_fallback",
            "grid_id": grid_id,
        }
