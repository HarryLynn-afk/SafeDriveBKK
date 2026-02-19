"""
Random Forest Baseline for Grid Risk Classification

Trains a Random Forest on the temporal train split (Jan-Sep),
evaluates on test split (Oct-Dec), and compares to the 83.8%
spatial baseline capture rate.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

SPATIAL_BASELINE_CAPTURE = 83.8  # from evidence_metrics.py holdout


def load_split():
    """Load the pre-built train/test split."""
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """Train Random Forest with class balancing."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_train, y_train, X_test, y_test, feature_names):
    """Calculate all metrics and return results dict."""
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Overall metrics
    train_acc = accuracy_score(y_train, y_pred_train) * 100
    test_acc = accuracy_score(y_test, y_pred_test) * 100
    test_prec = precision_score(y_test, y_pred_test, zero_division=0) * 100
    test_rec = recall_score(y_test, y_pred_test, zero_division=0) * 100
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0) * 100
    test_auc = roc_auc_score(y_test, y_proba_test) * 100

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    # Dangerous class focus
    n_dangerous_test = int(y_test.sum())
    caught = int(tp)

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    # Baseline comparison — capture rate
    # Spatial baseline captures 83.8% of test accidents in high-risk grids.
    # Our recall on dangerous class is the comparable metric.
    beats_baseline = test_rec > SPATIAL_BASELINE_CAPTURE

    results = {
        "model": "RandomForest",
        "n_estimators": 200,
        "max_depth": 15,
        "class_weight": "balanced",
        "train_accuracy": round(train_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "test_precision_dangerous": round(test_prec, 2),
        "test_recall_dangerous": round(test_rec, 2),
        "test_f1_dangerous": round(test_f1, 2),
        "test_auc_roc": round(test_auc, 2),
        "dangerous_caught": f"{caught}/{n_dangerous_test}",
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "spatial_baseline_capture": SPATIAL_BASELINE_CAPTURE,
        "beats_baseline": bool(beats_baseline),
        "top_15_features": [
            {"feature": name, "importance": round(float(imp), 5)}
            for name, imp in feat_imp[:15]
        ],
    }

    return results, y_pred_test, y_proba_test, feat_imp


def save_outputs(model, results):
    """Save model and results."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model with joblib (more efficient than pickle for numpy arrays)
    import joblib
    model_path = MODEL_DIR / "random_forest.pkl"
    joblib.dump(model, model_path)
    LOGGER.info("Saved model to %s", model_path)

    results_path = MODEL_DIR / "rf_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved results to %s", results_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("RANDOM FOREST BASELINE — GRID RISK CLASSIFICATION")
    print("=" * 70)

    # Load data
    X_train, y_train, X_test, y_test = load_split()
    feature_names = list(X_train.columns)

    print(f"\nTrain: {X_train.shape[0]} grids, {int(y_train.sum())} dangerous")
    print(f"Test:  {X_test.shape[0]} grids, {int(y_test.sum())} dangerous")
    print(f"Features: {len(feature_names)}")

    # Train
    print("\nTraining Random Forest (200 trees, balanced weights)...")
    model = train_model(X_train, y_train)
    print("Training complete.")

    # Evaluate
    results, y_pred_test, y_proba_test, feat_imp = evaluate(
        model, X_train, y_train, X_test, y_test, feature_names
    )

    # Save
    save_outputs(model, results)

    # Print report
    cm = results["confusion_matrix"]
    print(f"\n{'RESULTS':=^70}")

    print(f"\n--- Overall Metrics ---")
    print(f"  Train accuracy:     {results['train_accuracy']:.2f}%")
    print(f"  Test accuracy:      {results['test_accuracy']:.2f}%")
    print(f"  Test AUC-ROC:       {results['test_auc_roc']:.2f}%")

    print(f"\n--- Dangerous Class (what matters) ---")
    print(f"  Precision:  {results['test_precision_dangerous']:.2f}%")
    print(f"  Recall:     {results['test_recall_dangerous']:.2f}%  ({results['dangerous_caught']} grids)")
    print(f"  F1-score:   {results['test_f1_dangerous']:.2f}%")

    print(f"\n--- Confusion Matrix ---")
    print(f"                 Predicted Safe  Predicted Dangerous")
    print(f"  Actual Safe     {cm['true_negative']:>10}     {cm['false_positive']:>10}")
    print(f"  Actual Danger   {cm['false_negative']:>10}     {cm['true_positive']:>10}")

    print(f"\n--- Baseline Comparison ---")
    print(f"  Spatial baseline capture rate: {SPATIAL_BASELINE_CAPTURE}%")
    print(f"  Random Forest recall:          {results['test_recall_dangerous']:.2f}%")
    delta = results["test_recall_dangerous"] - SPATIAL_BASELINE_CAPTURE
    sign = "+" if delta >= 0 else ""
    print(f"  Difference:                    {sign}{delta:.2f}%")
    print(f"  Beats baseline?                {'YES' if results['beats_baseline'] else 'NO'}")

    print(f"\n--- Top 15 Features by Importance ---")
    for i, item in enumerate(results["top_15_features"], 1):
        bar = "#" * int(item["importance"] * 200)
        print(f"  {i:2d}. {item['feature']:<35s}  {item['importance']:.5f}  {bar}")

    # Full classification report
    print(f"\n--- sklearn Classification Report ---")
    print(classification_report(
        y_test, y_pred_test,
        target_names=["safe", "dangerous"],
        digits=4,
    ))

    print(f"\nSaved: models/random_forest.pkl")
    print(f"Saved: models/rf_results.json")


if __name__ == "__main__":
    main()
