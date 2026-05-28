"""
NLP Analysis Module for Bangkok Road Risk.

Processes Thai accident text columns, shows how text features feed into XGBoost,
and generates actionable per-zone policy recommendations.

Includes ML-based text classification (TF-IDF + Logistic Regression) trained on
2,466 Thai accident reports, replacing dictionary keyword matching.
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from .config import BKK_LAT_MIN as LAT_MIN, BKK_LON_MIN as LON_MIN, CELL_SIZE_DEG

LOGGER = logging.getLogger(__name__)

# Joblib artifacts were serialized with `nlp_analysis.thai_tokenizer`.
# Register the current module under that name so they can be loaded both as
# `src.nlp_analysis` and `nlp_analysis`.
sys.modules.setdefault("nlp_analysis", sys.modules[__name__])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

ACCIDENTS_PATH = DATA_DIR / "bangkok_accidents_clean.csv"
ML_FEATURES_PATH = DATA_DIR / "ml_features.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "xgb_feature_importance.csv"
NLP_CACHE_PATH = STATIC_DIR / "nlp_cache.json"
NLP_METRICS_PATH = DATA_DIR / "nlp_metrics.json"

# Thai column names
COL_CAUSE = "มูลเหตุสันนิษฐาน"
COL_TYPE = "ลักษณะการเกิดเหตุ"
COL_WEATHER = "สภาพอากาศ"
COL_LOCATION = "บริเวณที่เกิดเหตุ"
COL_VEHICLE = "รถคันที่1"

# Cause categories -> English + XGBoost feature name
CAUSE_TRANSLATIONS = {
    "ขับรถเร็วเกินอัตรากำหนด": {
        "english": "Speeding",
        "feature": "cause_ขับรถเร็วเกินอัตรากำหนด",
    },
    "คน/รถ/สัตว์ตัดหน้ากระชั้นชิด": {
        "english": "Cutting in front",
        "feature": "cause_คน/รถ/สัตว์ตัดหน้ากระชั้นชิด",
    },
    "ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร": {
        "english": "Running red light / traffic sign violation",
        "feature": "cause_ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร",
    },
    "ยางเสื่อมสภาพ/ยางแตก": {
        "english": "Tire blowout / worn tires",
        "feature": "cause_ยางเสื่อมสภาพ/ยางแตก",
    },
    "หลับใน": {
        "english": "Drowsy driving",
        "feature": "cause_หลับใน",
    },
}

TYPE_TRANSLATIONS = {
    "ชนท้าย": {"english": "Rear-end collision", "feature": "type_ชนท้าย"},
    "ชนด้านข้าง": {"english": "Side collision", "feature": "type_ชนด้านข้าง"},
    "พลิกคว่ำ/ตกถนนในทางตรง": {
        "english": "Rollover / road departure",
        "feature": "type_พลิกคว่ำ/ตกถนนในทางตรง",
    },
    "ชนสิ่งกีดขวาง (บนผิวจราจร)": {
        "english": "Hitting obstacle on road",
        "feature": "type_ชนสิ่งกีดขวาง (บนผิวจราจร)",
    },
    "อื่นๆ": {"english": "Other type", "feature": "type_อื่นๆ"},
}

# Recommendation lookup: cause -> policy action
POLICY_RECOMMENDATIONS = {
    "ขับรถเร็วเกินอัตรากำหนด": "Deploy automated speed enforcement cameras and install rumble strips",
    "คน/รถ/สัตว์ตัดหน้ากระชั้นชิด": "Add traffic signals or roundabout, improve sight lines at intersection",
    "ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร": "Install red-light cameras and increase signal visibility",
    "ยางเสื่อมสภาพ/ยางแตก": "Increase roadside vehicle inspection checkpoints",
    "หลับใน": "Install alertness rumble strips and add rest area signage",
}

VEHICLE_RECOMMENDATIONS = {
    "high_motorcycle": "Dedicate motorcycle lane and enforce helmet laws",
    "high_truck": "Restrict heavy vehicle hours and add truck-specific signage",
}

WEATHER_RECOMMENDATIONS = {
    "high_rain": "Improve drainage and install anti-skid road surface",
}

# Cause group English names (must match train_text_classifier.py)
def thai_tokenizer(text):
    """Tokenize Thai text using pythainlp newmm engine.

    Defined here so joblib can resolve it when loading the TF-IDF vectorizer.
    """
    from pythainlp.tokenize import word_tokenize

    tokens = word_tokenize(str(text), engine="newmm")
    return [t for t in tokens if len(t.strip()) > 1]


CAUSE_GROUP_ENGLISH = {
    "speeding": "Speeding",
    "cutting_in_front": "Cutting in front",
    "drowsy_driving": "Drowsy driving",
    "equipment_failure": "Equipment failure",
    "traffic_violation": "Traffic violation",
    "other": "Other causes",
}


class CauseClassifier:
    """ML-based Thai text classifier for accident cause prediction.

    Uses TF-IDF + Logistic Regression trained on 2,466 Thai accident reports.
    Falls back to dictionary matching if model files are unavailable.
    """

    def __init__(self):
        self.classifier = None
        self.vectorizer = None
        self.available = False
        self.metrics = {}
        self._load()

    def _load(self):
        clf_path = MODELS_DIR / "cause_classifier.pkl"
        vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        try:
            self.classifier = joblib.load(clf_path)
            self.vectorizer = joblib.load(vec_path)
            self.available = True
            LOGGER.info("ML cause classifier loaded from %s", clf_path)
        except Exception as exc:
            LOGGER.warning("ML classifier not available (%s), using dictionary fallback", exc)

        try:
            if NLP_METRICS_PATH.exists():
                with open(NLP_METRICS_PATH, encoding="utf-8") as f:
                    self.metrics = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    def classify(self, text, confidence_threshold=0.5):
        """Classify Thai text into a cause category.

        Returns dict with cause, confidence, method, and english_name.
        """
        if not text or not str(text).strip():
            return {
                "cause": "other",
                "english_name": "Other causes",
                "confidence": 0.0,
                "method": "empty_input",
            }

        if self.available:
            try:
                tfidf = self.vectorizer.transform([str(text)])
                prediction = self.classifier.predict(tfidf)[0]
                probas = self.classifier.predict_proba(tfidf)[0]
                confidence = float(max(probas))

                if confidence >= confidence_threshold:
                    return {
                        "cause": prediction,
                        "english_name": CAUSE_GROUP_ENGLISH.get(prediction, prediction),
                        "confidence": round(confidence, 4),
                        "method": "ml_classifier",
                        "all_probabilities": {
                            cls: round(float(p), 4)
                            for cls, p in zip(self.classifier.classes_, probas)
                        },
                    }
            except Exception as exc:
                LOGGER.warning("ML classify error: %s", exc)

        # Dictionary fallback
        return self._dictionary_classify(text)

    @staticmethod
    def _dictionary_classify(text):
        """Fallback dictionary-based classification."""
        text = str(text)
        for thai, info in CAUSE_TRANSLATIONS.items():
            if thai in text:
                return {
                    "cause": thai,
                    "english_name": info["english"],
                    "confidence": 1.0,
                    "method": "dictionary_fallback",
                }
        return {
            "cause": "other",
            "english_name": "Other causes",
            "confidence": 0.0,
            "method": "dictionary_fallback",
        }

    def classify_batch(self, texts):
        """Classify multiple Thai texts at once."""
        return [self.classify(t) for t in texts]

    def get_metrics(self):
        """Return stored training metrics."""
        return self.metrics


# Module-level singleton (loaded once on import)
_cause_classifier = None


def get_cause_classifier():
    """Get or create the singleton CauseClassifier instance."""
    global _cause_classifier
    if _cause_classifier is None:
        _cause_classifier = CauseClassifier()
    return _cause_classifier


class NLPAnalyzer:
    """Analyzes Thai accident text and maps to XGBoost features."""

    def __init__(self):
        self._accidents = pd.read_csv(ACCIDENTS_PATH, encoding="utf-8-sig")
        self._ml_features = pd.read_csv(ML_FEATURES_PATH, encoding="utf-8-sig")
        self._importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
        self._importance_dict = dict(
            zip(self._importance["feature"], self._importance["importance"])
        )
        LOGGER.info(
            "NLPAnalyzer loaded: %d accidents, %d grid features",
            len(self._accidents),
            len(self._ml_features),
        )

    def get_text_feature_pipeline_preview(self):
        """Show 3-5 sample records flowing through the NLP -> XGBoost pipeline.

        Now includes ML classifier prediction alongside the dictionary mapping.
        """
        try:
            from pythainlp.tokenize import word_tokenize
        except ImportError:
            word_tokenize = lambda t, **kw: list(t)

        classifier = get_cause_classifier()
        samples = []
        seen_causes = set()

        for _, row in self._accidents.iterrows():
            cause = row.get(COL_CAUSE, "")
            if pd.isna(cause) or cause in seen_causes:
                continue
            if cause in CAUSE_TRANSLATIONS:
                seen_causes.add(cause)
                info = CAUSE_TRANSLATIONS[cause]
                tokens = word_tokenize(cause, engine="newmm")

                # Build full report text for ML classification
                report_parts = []
                for col in [COL_CAUSE, COL_LOCATION, COL_TYPE, COL_WEATHER, COL_VEHICLE]:
                    val = row.get(col, "")
                    if pd.notna(val) and str(val).strip():
                        report_parts.append(str(val).strip())
                report_text = " ".join(report_parts)

                ml_result = classifier.classify(report_text)

                samples.append(
                    {
                        "raw_thai": cause,
                        "tokens": tokens,
                        "category_english": info["english"],
                        "xgboost_feature": info["feature"],
                        "feature_importance": round(
                            self._importance_dict.get(info["feature"], 0) * 100, 3
                        ),
                        "ml_prediction": ml_result.get("cause", ""),
                        "ml_confidence": ml_result.get("confidence", 0),
                        "ml_method": ml_result.get("method", ""),
                    }
                )
            if len(samples) >= 5:
                break

        return {
            "pipeline_examples": samples,
            "total_features_in_model": 44,
            "ml_classifier_available": classifier.available,
        }

    def get_cause_frequency_analysis(self):
        """Cause breakdown matching feature_engineering.py categorization."""
        if COL_CAUSE not in self._accidents.columns:
            return {"categories": [], "total_records": 0}

        cause_counts = self._accidents[COL_CAUSE].value_counts()
        top5 = cause_counts.head(5).index.tolist()
        total = len(self._accidents)

        categories = []
        for cause_thai in top5:
            count = int(cause_counts[cause_thai])
            info = CAUSE_TRANSLATIONS.get(
                cause_thai,
                {"english": cause_thai, "feature": f"cause_{cause_thai}"},
            )
            categories.append(
                {
                    "thai_name": cause_thai,
                    "english_name": info["english"],
                    "count": count,
                    "percentage": round(count / total * 100, 1),
                    "xgboost_feature": info["feature"],
                    "feature_importance": round(
                        self._importance_dict.get(info["feature"], 0) * 100, 3
                    ),
                }
            )

        other_count = int(cause_counts.iloc[5:].sum()) if len(cause_counts) > 5 else 0
        if other_count > 0:
            categories.append(
                {
                    "thai_name": "อื่นๆ",
                    "english_name": "Other causes",
                    "count": other_count,
                    "percentage": round(other_count / total * 100, 1),
                    "xgboost_feature": "cause_other",
                    "feature_importance": round(
                        self._importance_dict.get("cause_other", 0) * 100, 3
                    ),
                }
            )

        return {"categories": categories, "total_records": total}

    def get_nlp_feature_importance_summary(self):
        """Sum importance by feature category."""
        groups = {
            "severity": ["severity_mean", "severity_total", "has_fatality"],
            "spatial": [
                "grid_density",
                "distance_from_center_km",
                "accidents_n",
                "accidents_s",
                "accidents_e",
                "accidents_w",
                "accidents_ne",
                "accidents_nw",
                "accidents_se",
                "accidents_sw",
                "accidents_neighbors_total",
                "accidents_neighbors_mean",
                "accidents_neighbors_max",
            ],
            "temporal": [
                "rush_hour_pct",
                "weekend_pct",
                "peak_hour",
                "peak_day",
                "month_spread",
            ],
            "vehicle": [
                "motorcycle_count",
                "car_count",
                "pickup_count",
                "heavy_truck_count",
                "motorcycle_pct",
                "car_pct",
                "pickup_pct",
                "heavy_truck_pct",
                "max_vehicles",
            ],
            "nlp_text": [
                f
                for f in self._importance_dict
                if f.startswith("cause_")
                or f.startswith("type_")
                or f in ("clear_weather_pct", "rain_weather_pct")
            ],
        }

        summary = {}
        for group, features in groups.items():
            total = sum(self._importance_dict.get(f, 0) for f in features)
            summary[group] = {
                "importance_pct": round(total * 100, 2),
                "feature_count": len(features),
                "features": [
                    {"name": f, "importance": round(self._importance_dict.get(f, 0) * 100, 3)}
                    for f in features
                    if self._importance_dict.get(f, 0) > 0
                ],
            }

        return summary

    def get_zone_recommendations(self, top_n=10):
        """Generate actionable policy recommendations for dangerous zones."""
        dangerous = self._ml_features[self._ml_features["dangerous"] == 1].copy()
        dangerous = dangerous.nlargest(top_n, "total_accidents")

        cause_cols = [c for c in dangerous.columns if c.startswith("cause_")]
        type_cols = [c for c in dangerous.columns if c.startswith("type_")]

        recommendations = []
        for _, row in dangerous.iterrows():
            grid_id = row["grid_id"]

            # Find dominant cause
            dominant_cause = "other"
            dominant_cause_pct = 0
            for col in cause_cols:
                val = row.get(col, 0)
                if val > dominant_cause_pct:
                    dominant_cause_pct = val
                    dominant_cause = col.replace("cause_", "")

            # Find dominant type
            dominant_type = "other"
            dominant_type_pct = 0
            for col in type_cols:
                val = row.get(col, 0)
                if val > dominant_type_pct:
                    dominant_type_pct = val
                    dominant_type = col.replace("type_", "")

            # Get location context from accidents in this grid
            location_context = self._get_location_context(grid_id)

            # Build recommendation
            rec = POLICY_RECOMMENDATIONS.get(
                dominant_cause, "Conduct detailed safety audit of this zone"
            )

            # Add vehicle-specific recommendation
            motorcycle_pct = row.get("motorcycle_pct", 0)
            if motorcycle_pct > 30:
                rec += ". " + VEHICLE_RECOMMENDATIONS["high_motorcycle"]

            rain_pct = row.get("rain_weather_pct", 0)
            if rain_pct > 20:
                rec += ". " + WEATHER_RECOMMENDATIONS["high_rain"]

            cause_info = CAUSE_TRANSLATIONS.get(dominant_cause, {})
            type_info = TYPE_TRANSLATIONS.get(dominant_type, {})

            recommendations.append(
                {
                    "grid_id": grid_id,
                    "total_accidents": int(row.get("total_accidents", 0)),
                    "center_lat": float(row.get("center_lat", 0)),
                    "center_lon": float(row.get("center_lon", 0)),
                    "dominant_cause_thai": dominant_cause,
                    "dominant_cause_english": cause_info.get("english", dominant_cause),
                    "dominant_cause_pct": round(dominant_cause_pct, 1),
                    "dominant_type_thai": dominant_type,
                    "dominant_type_english": type_info.get("english", dominant_type),
                    "dominant_type_pct": round(dominant_type_pct, 1),
                    "motorcycle_pct": round(motorcycle_pct, 1),
                    "location_context": location_context,
                    "recommendation": rec,
                }
            )

        return recommendations

    def _get_location_context(self, grid_id):
        """Extract most common location description for accidents in a grid."""
        # Parse grid indices
        parts = grid_id.replace("G", "").split("_")
        lat_idx, lon_idx = int(parts[0]), int(parts[1])
        lat_min = LAT_MIN + lat_idx * CELL_SIZE_DEG
        lat_max = lat_min + CELL_SIZE_DEG
        lon_min = LON_MIN + lon_idx * CELL_SIZE_DEG
        lon_max = lon_min + CELL_SIZE_DEG

        mask = (
            (self._accidents["LATITUDE"] >= lat_min)
            & (self._accidents["LATITUDE"] < lat_max)
            & (self._accidents["LONGITUDE"] >= lon_min)
            & (self._accidents["LONGITUDE"] < lon_max)
        )
        subset = self._accidents[mask]
        if COL_LOCATION in subset.columns and len(subset) > 0:
            locations = subset[COL_LOCATION].dropna().value_counts()
            if len(locations) > 0:
                return str(locations.index[0])
        return "Unknown location"

    def get_word_cloud_data(self):
        """Tokenize Thai text and return word frequencies."""
        word_freq = {}
        try:
            from pythainlp.tokenize import word_tokenize
            from pythainlp.corpus import thai_stopwords
            stopwords = thai_stopwords()

            for col in [COL_CAUSE, COL_LOCATION, COL_TYPE]:
                if col not in self._accidents.columns:
                    continue
                for text in self._accidents[col].dropna().unique():
                    tokens = word_tokenize(str(text), engine="newmm")
                    # Skip if tokenizer returned character-level output
                    if tokens and (sum(len(t) for t in tokens) / len(tokens)) < 2.5:
                        continue
                    for token in tokens:
                        token = token.strip()
                        if len(token) <= 1 or token in stopwords or token == " ":
                            continue
                        word_freq[token] = word_freq.get(token, 0) + 1
        except ImportError:
            pass

        # If word-level tokenization produced nothing, fall back to raw cause/type values
        if not word_freq:
            for col in [COL_CAUSE, COL_TYPE]:
                if col not in self._accidents.columns:
                    continue
                for val, count in self._accidents[col].dropna().value_counts().items():
                    text = str(val).strip()
                    if text:
                        word_freq[text] = int(count)

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50]
        return {"words": [{"text": w, "frequency": f} for w, f in sorted_words]}

    def get_accident_type_analysis(self):
        """Accident type breakdown with XGBoost feature mapping."""
        if COL_TYPE not in self._accidents.columns:
            return {"types": [], "total_records": 0}

        type_counts = self._accidents[COL_TYPE].value_counts()
        total = len(self._accidents)
        types = []

        for type_thai, count in type_counts.head(6).items():
            info = TYPE_TRANSLATIONS.get(
                type_thai, {"english": type_thai, "feature": f"type_{type_thai}"}
            )
            types.append(
                {
                    "thai_name": type_thai,
                    "english_name": info["english"],
                    "count": int(count),
                    "percentage": round(int(count) / total * 100, 1),
                    "xgboost_feature": info["feature"],
                    "feature_importance": round(
                        self._importance_dict.get(info["feature"], 0) * 100, 3
                    ),
                }
            )

        return {"types": types, "total_records": total}

    def get_accident_samples(self, n=30):
        """Return n real accident records with full NLP analysis for the interactive analyzer."""
        try:
            from pythainlp.tokenize import word_tokenize

            def _tokenize(text):
                tokens = word_tokenize(text, engine="newmm")
                # Guard against character-level fallback: if average token length < 2.5
                # the engine returned individual characters — revert to whole-text
                if tokens and (sum(len(t) for t in tokens) / len(tokens)) < 2.5:
                    return [text]
                return [t for t in tokens if t.strip()]
        except ImportError:
            def _tokenize(text):
                return [text]

        # Collect samples with variety across cause categories
        cause_buckets = {cause: [] for cause in CAUSE_TRANSLATIONS}
        cause_buckets["other"] = []

        for idx, row in self._accidents.iterrows():
            cause = row.get(COL_CAUSE, "")
            if pd.isna(cause) or not str(cause).strip():
                continue
            bucket = cause if cause in CAUSE_TRANSLATIONS else "other"
            if len(cause_buckets[bucket]) < 6:
                cause_buckets[bucket].append((idx, row))

        # Flatten, ensuring variety first, then pad if needed
        selected = []
        for bucket_rows in cause_buckets.values():
            selected.extend(bucket_rows)
        selected = selected[:n]

        # Pad with random records if we have fewer than n
        if len(selected) < n:
            remaining = self._accidents.dropna(subset=[COL_CAUSE])
            used_ids = {idx for idx, _ in selected}
            for idx, row in remaining.iterrows():
                if idx not in used_ids:
                    selected.append((idx, row))
                if len(selected) >= n:
                    break

        samples = []
        for idx, row in selected:
            cause = str(row.get(COL_CAUSE, "")).strip()
            tokens = _tokenize(cause) if cause else []
            info = CAUSE_TRANSLATIONS.get(cause, {"english": "Other", "feature": "cause_other"})
            location = row.get(COL_LOCATION, "")
            location = str(location) if not pd.isna(location) else "Unknown"
            samples.append({
                "accident_id": int(idx),
                "raw_thai": cause,
                "tokens": tokens,
                "category_english": info["english"],
                "xgboost_feature": info["feature"],
                "feature_importance": round(
                    self._importance_dict.get(info["feature"], 0) * 100, 3
                ),
                "location": location,
            })

        return samples

    def get_map_data(self, top_n=200):
        """Return top N dangerous zones with cause data for the interactive Leaflet map."""
        GRID_STATS_PATH = DATA_DIR / "bangkok_grid_stats.csv"

        dangerous = self._ml_features[self._ml_features["dangerous"] == 1].copy()
        dangerous = dangerous.nlargest(top_n, "total_accidents")

        stats_dict = {}
        if GRID_STATS_PATH.exists():
            stats_df = pd.read_csv(GRID_STATS_PATH)
            stats_dict = dict(zip(stats_df["grid_id"], stats_df["risk_score"]))

        cause_cols = [c for c in dangerous.columns if c.startswith("cause_")]

        zones = []
        for _, row in dangerous.iterrows():
            dominant_col = max(cause_cols, key=lambda c: float(row.get(c, 0))) if cause_cols else "cause_other"
            dominant_thai = dominant_col.replace("cause_", "")
            dominant_pct = round(float(row.get(dominant_col, 0)), 1)
            dominant_english = CAUSE_TRANSLATIONS.get(dominant_thai, {}).get("english", "Other")

            zones.append({
                "grid_id": row["grid_id"],
                "lat": float(row["center_lat"]),
                "lon": float(row["center_lon"]),
                "total_accidents": int(row["total_accidents"]),
                "risk_score": round(float(stats_dict.get(row["grid_id"], 0)), 1),
                "dominant_cause": dominant_thai,
                "dominant_cause_english": dominant_english,
                "dominant_cause_pct": dominant_pct,
                "motorcycle_pct": round(float(row.get("motorcycle_pct", 0)), 1),
            })

        return zones

    def get_ml_classifier_metrics(self):
        """Return ML text classifier metrics for display."""
        classifier = get_cause_classifier()
        return classifier.get_metrics()

    def generate_cache(self):
        """Generate and save all NLP analysis results to cache file."""
        LOGGER.info("Generating NLP cache...")

        cache = {
            "pipeline_preview": self.get_text_feature_pipeline_preview(),
            "cause_analysis": self.get_cause_frequency_analysis(),
            "type_analysis": self.get_accident_type_analysis(),
            "feature_importance": self.get_nlp_feature_importance_summary(),
            "recommendations": self.get_zone_recommendations(top_n=10),
            "word_cloud": self.get_word_cloud_data(),
            "accident_samples": self.get_accident_samples(30),
            "map_data": self.get_map_data(200),
            "ml_classifier_metrics": self.get_ml_classifier_metrics(),
        }

        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        with open(NLP_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

        LOGGER.info("NLP cache saved to %s", NLP_CACHE_PATH)
        return cache


# ---------------------------------------------------------------------------
# Cache loader — in-process singleton with staleness detection
# ---------------------------------------------------------------------------

_nlp_cache: dict | None = None


def load_nlp_cache() -> dict:
    """Return NLP analysis results from nlp_cache.json.

    Reads from disk once per process lifetime; subsequent calls return the
    in-process copy. If the accident data file is newer than the cache file,
    logs a warning and returns the stale cache — regeneration is intentionally
    left to the offline step (`python -m src.nlp_analysis`) so web requests
    are never blocked by a multi-minute pipeline run.

    Always returns a dict (empty if the cache file is missing or unreadable).
    """
    global _nlp_cache
    if _nlp_cache is not None:
        return _nlp_cache

    try:
        if NLP_CACHE_PATH.exists():
            with open(NLP_CACHE_PATH, encoding="utf-8") as f:
                _nlp_cache = json.load(f)
            if ACCIDENTS_PATH.exists():
                if ACCIDENTS_PATH.stat().st_mtime > NLP_CACHE_PATH.stat().st_mtime:
                    LOGGER.warning(
                        "NLP cache is stale — accident data is newer than %s. "
                        "Run `python -m src.nlp_analysis` to regenerate.",
                        NLP_CACHE_PATH,
                    )
            return _nlp_cache
    except (json.JSONDecodeError, OSError):
        LOGGER.warning("Failed to load NLP cache from %s", NLP_CACHE_PATH)

    _nlp_cache = {}
    return _nlp_cache


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = NLPAnalyzer()
    cache = analyzer.generate_cache()
    print(f"Cache generated with {len(cache)} sections")
    print(f"  - {len(cache['cause_analysis']['categories'])} cause categories")
    print(f"  - {len(cache['recommendations'])} zone recommendations")
    print(f"  - {len(cache['word_cloud']['words'])} word cloud entries")
