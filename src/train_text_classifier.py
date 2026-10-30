"""
Train a TF-IDF + Logistic Regression text classifier for Thai accident reports.

This classifier processes Thai accident report text through NLP tokenization
and TF-IDF vectorization to classify accident cause categories — replacing
dictionary keyword matching with a trained ML model.

Input: Full Thai accident report text (cause description + location + type + weather)
Output: Cause category with confidence score

Pipeline: Thai text → pythainlp tokenization → TF-IDF features → LogisticRegression

Output files:
  models/cause_classifier.pkl       — trained LogisticRegression model
  models/tfidf_vectorizer.pkl       — fitted TF-IDF vectorizer
  data/processed/nlp_metrics.json   — classification metrics for display
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

ACCIDENTS_PATH = DATA_DIR / "bangkok_accidents_clean.csv"

# Thai column names
COL_CAUSE = "มูลเหตุสันนิษฐาน"
COL_TYPE = "ลักษณะการเกิดเหตุ"
COL_WEATHER = "สภาพอากาศ"
COL_LOCATION = "บริเวณที่เกิดเหตุ"
COL_VEHICLE = "รถคันที่1"

# All Thai text columns to combine as input
TEXT_COLUMNS = [COL_CAUSE, COL_LOCATION, COL_TYPE, COL_WEATHER, COL_VEHICLE]

# Map 27 raw causes -> 6 categories (top 5 + other)
CAUSE_GROUPS = {
    "ขับรถเร็วเกินอัตรากำหนด": "speeding",
    "คน/รถ/สัตว์ตัดหน้ากระชั้นชิด": "cutting_in_front",
    "หลับใน": "drowsy_driving",
    "ยางเสื่อมสภาพ/ยางแตก": "equipment_failure",
    "อุปกรณ์ยานพาหนะบกพร่อง": "equipment_failure",
    "ระบบห้ามล้อขัดข้อง/ระบบเบรกชำรุด": "equipment_failure",
    "ฝ่าฝืนสัญญาณไฟ/เครื่องหมายจราจร": "traffic_violation",
    "ไม่ให้สัญญาณชะลอ/เลี้ยว": "traffic_violation",
    "ขับรถย้อนศร": "traffic_violation",
    "เปลี่ยนช่องทางกะทันหัน": "traffic_violation",
    "แซงรถอย่างผิดกฎหมาย": "traffic_violation",
    "ขับรถตามกระชั้นชิด": "traffic_violation",
    "ไม่ยอมให้รถที่มีสิทธิ์ไปก่อน": "traffic_violation",
    "ไม่ให้สัญญาณเข้าจอด/ออกจากที่จอด": "traffic_violation",
}

CAUSE_ENGLISH = {
    "speeding": "Speeding",
    "cutting_in_front": "Cutting in front",
    "drowsy_driving": "Drowsy driving",
    "equipment_failure": "Equipment failure",
    "traffic_violation": "Traffic violation",
    "other": "Other causes",
}

# Import tokenizer from nlp_analysis so joblib stores the correct reference
from nlp_analysis import thai_tokenizer


def build_report_text(row):
    """Combine all Thai text columns into a single accident report string."""
    parts = []
    for col in TEXT_COLUMNS:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    return " ".join(parts)


def main():
    print("=" * 60)
    print("Thai Accident Report Text Classifier — Training")
    print("=" * 60)

    # Load data
    df = pd.read_csv(ACCIDENTS_PATH, encoding="utf-8-sig")
    print(f"\nLoaded {len(df)} accident records")

    # Drop rows with missing cause
    df = df.dropna(subset=[COL_CAUSE])
    print(f"Records with cause labels: {len(df)}")

    # Map causes to grouped categories
    df["cause_group"] = df[COL_CAUSE].map(CAUSE_GROUPS).fillna("other")

    print("\nCause group distribution:")
    dist = df["cause_group"].value_counts()
    for cause, count in dist.items():
        print(f"  {cause:25s} {count:5d}  ({count / len(df) * 100:5.1f}%)")

    # Build composite report text from all Thai columns
    df["report_text"] = df.apply(build_report_text, axis=1)

    # Remove empty texts
    df = df[df["report_text"].str.strip().str.len() > 0]
    print(f"\nRecords with valid text: {len(df)}")

    X = df["report_text"].values
    y = df["cause_group"].values

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # TF-IDF with Thai tokenization
    print("\nFitting TF-IDF vectorizer with Thai tokenization...")
    vectorizer = TfidfVectorizer(
        tokenizer=thai_tokenizer,
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=None,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"TF-IDF matrix: {X_train_tfidf.shape}")

    # Train Logistic Regression
    print("\nTraining Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver="lbfgs",
    )
    classifier.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted"
    )

    print(f"\n{'=' * 60}")
    print("TEST SET RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"Weighted Precision: {precision:.4f} ({precision * 100:.1f}%)")
    print(f"Weighted Recall:    {recall:.4f} ({recall * 100:.1f}%)")
    print(f"Weighted F1-Score:  {f1:.4f} ({f1 * 100:.1f}%)")

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    print("Confusion Matrix:")
    print(f"Classes: {list(classifier.classes_)}")
    print(cm)

    # Cross-validation on full dataset
    print("\n5-fold Cross-Validation...")
    X_all_tfidf = vectorizer.transform(X)
    cv_scores = cross_val_score(classifier, X_all_tfidf, y, cv=5, scoring="accuracy")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Top TF-IDF features per class
    feature_names = vectorizer.get_feature_names_out()
    top_features_per_class = {}
    for i, cls in enumerate(classifier.classes_):
        coef = classifier.coef_[i]
        top_idx = np.argsort(coef)[-5:][::-1]
        top_features_per_class[cls] = [
            {"token": feature_names[j], "weight": round(float(coef[j]), 3)}
            for j in top_idx
        ]

    # Per-class metrics
    per_class = {}
    for cls in classifier.classes_:
        if cls in report:
            per_class[cls] = {
                "english_name": CAUSE_ENGLISH.get(cls, cls),
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1_score": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
                "top_features": top_features_per_class.get(cls, []),
            }

    # Save model and vectorizer
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODELS_DIR / "cause_classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.pkl")
    print(f"\nModel saved: {MODELS_DIR / 'cause_classifier.pkl'}")
    print(f"Vectorizer saved: {MODELS_DIR / 'tfidf_vectorizer.pkl'}")

    # Save metrics JSON
    metrics = {
        "model_type": "TF-IDF + Logistic Regression",
        "description": (
            "Thai text classifier trained on 2,466 accident reports using "
            "pythainlp, TF-IDF, and Logistic Regression. The recorded cause is "
            "included in the input and used to derive the label. Scores describe "
            "categorization with the cause supplied, not prediction of unknown "
            "causes. Cross-validation reuses a pre-fitted vectorizer; it is not "
            "an independent end-to-end evaluation."
        ),
        "tokenizer": "pythainlp newmm (Thai word segmentation)",
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "total_samples": int(len(X)),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "num_classes": int(len(classifier.classes_)),
        "classes": list(classifier.classes_),
        "class_english": CAUSE_ENGLISH,
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "input_columns": TEXT_COLUMNS,
        "class_distribution": {k: int(v) for k, v in dist.items()},
    }

    metrics_path = DATA_DIR / "nlp_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved: {metrics_path}")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    return metrics


if __name__ == "__main__":
    main()
