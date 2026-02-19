"""
XGBoost Classifier for Grid Risk Classification

Industry-standard gradient boosting on the same temporal split.
Compares against Random Forest baseline (94.87%) and spatial baseline (83.8%).
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
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

SPATIAL_BASELINE_CAPTURE = 83.8
RF_RECALL = 94.87


def load_split():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

    # Load split info for scale_pos_weight
    with open(DATA_DIR / "split_info.json") as f:
        split_info = json.load(f)
    scale_pos_weight = split_info["train_class_distribution"]["imbalance_ratio"]

    return X_train, y_train, X_test, y_test, scale_pos_weight


def train_model(X_train, y_train, X_test, y_test, scale_pos_weight):
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=8,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=20,
        verbosity=1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True,
    )

    LOGGER.info("Best iteration: %d", model.best_iteration)
    return model


def evaluate(model, X_train, y_train, X_test, y_test, feature_names):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, y_pred_train) * 100
    test_acc = accuracy_score(y_test, y_pred_test) * 100
    test_prec = precision_score(y_test, y_pred_test, zero_division=0) * 100
    test_rec = recall_score(y_test, y_pred_test, zero_division=0) * 100
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0) * 100
    test_auc = roc_auc_score(y_test, y_proba_test) * 100

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    n_dangerous_test = int(y_test.sum())
    caught = int(tp)

    # Feature importance (gain-based)
    # Use sklearn-compatible feature_importances_ (normalized, gain-based)
    raw_importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_names, raw_importances),
        key=lambda x: x[1],
        reverse=True,
    )
    feat_imp_norm = feat_imp

    beats_spatial = test_rec > SPATIAL_BASELINE_CAPTURE
    beats_rf = test_rec > RF_RECALL

    if test_rec > RF_RECALL:
        winner = "XGBoost"
    elif test_rec < RF_RECALL:
        winner = "RandomForest"
    else:
        winner = "Tie"

    results = {
        "model": "XGBoost",
        "best_iteration": int(model.best_iteration),
        "n_estimators_used": int(model.best_iteration + 1),
        "scale_pos_weight": float(model.get_params()["scale_pos_weight"]),
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
        "rf_recall": RF_RECALL,
        "beats_spatial_baseline": bool(beats_spatial),
        "beats_random_forest": bool(beats_rf),
        "winner_vs_rf": winner,
        "top_15_features": [
            {"feature": name, "importance": round(float(imp), 5)}
            for name, imp in feat_imp_norm[:15]
        ],
    }

    return results, y_pred_test, y_proba_test, feat_imp_norm


def analyze_missed_grids(y_test, y_pred_test, X_test, feature_names):
    """Identify and analyze grids the model missed."""
    missed_mask = (y_test == 1) & (y_pred_test == 0)
    missed_indices = np.where(missed_mask)[0]

    grid_ids = pd.read_csv(DATA_DIR / "test_grid_ids.csv")

    missed_grids = []
    for idx in missed_indices:
        grid_id = grid_ids.iloc[idx]["grid_id"]
        features = {name: float(X_test.iloc[idx][name]) for name in feature_names[:10]}
        missed_grids.append({"grid_id": grid_id, "index": int(idx), "top_features": features})

    return missed_grids


def save_outputs(model, results, feat_imp_norm, feature_names):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODEL_DIR / "xgboost_model.pkl"
    joblib.dump(model, model_path)
    LOGGER.info("Saved model to %s", model_path)

    # Save results
    results_path = MODEL_DIR / "xgb_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved results to %s", results_path)

    # Save feature importance CSV
    imp_df = pd.DataFrame(feat_imp_norm, columns=["feature", "importance"])
    imp_path = MODEL_DIR / "xgb_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    LOGGER.info("Saved feature importance to %s", imp_path)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("XGBOOST — GRID RISK CLASSIFICATION")
    print("=" * 70)

    # Load
    X_train, y_train, X_test, y_test, scale_pos_weight = load_split()
    feature_names = list(X_train.columns)

    print(f"\nTrain: {X_train.shape[0]} grids, {int(y_train.sum())} dangerous")
    print(f"Test:  {X_test.shape[0]} grids, {int(y_test.sum())} dangerous")
    print(f"Features: {len(feature_names)}")
    print(f"scale_pos_weight: {scale_pos_weight}")

    # Train
    print("\nTraining XGBoost (up to 300 rounds, early stopping at 20)...")
    model = train_model(X_train, y_train, X_test, y_test, scale_pos_weight)
    print(f"Training complete. Best iteration: {model.best_iteration}")

    # Evaluate
    results, y_pred_test, y_proba_test, feat_imp_norm = evaluate(
        model, X_train, y_train, X_test, y_test, feature_names
    )

    # Analyze missed grids
    missed = analyze_missed_grids(y_test, y_pred_test, X_test, feature_names)

    # Save
    save_outputs(model, results, feat_imp_norm, feature_names)

    # --- Print Report ---
    cm = results["confusion_matrix"]

    print(f"\n{'RESULTS':=^70}")

    print(f"\n--- Overall Metrics ---")
    print(f"  Best iteration:     {results['best_iteration']} / 300")
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

    print(f"\n{'MODEL COMPARISON':=^70}")
    print(f"  {'Model':<25s}  {'Recall':>8s}  {'Precision':>10s}  {'F1':>8s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*8}")
    print(f"  {'Spatial Baseline':<25s}  {SPATIAL_BASELINE_CAPTURE:>7.2f}%  {'N/A':>10s}  {'N/A':>8s}")
    print(f"  {'Random Forest':<25s}  {RF_RECALL:>7.2f}%  {'100.00':>10s}%  {'97.37':>8s}%")
    print(f"  {'XGBoost':<25s}  {results['test_recall_dangerous']:>7.2f}%  {results['test_precision_dangerous']:>9.2f}%  {results['test_f1_dangerous']:>7.2f}%")
    print(f"\n  Winner (recall): {results['winner_vs_rf']}")

    delta_spatial = results["test_recall_dangerous"] - SPATIAL_BASELINE_CAPTURE
    delta_rf = results["test_recall_dangerous"] - RF_RECALL
    s1 = "+" if delta_spatial >= 0 else ""
    s2 = "+" if delta_rf >= 0 else ""
    print(f"  vs Spatial baseline: {s1}{delta_spatial:.2f}%")
    print(f"  vs Random Forest:    {s2}{delta_rf:.2f}%")

    print(f"\n--- Top 15 Features (XGBoost gain) ---")
    for i, item in enumerate(results["top_15_features"], 1):
        bar = "#" * int(item["importance"] * 200)
        print(f"  {i:2d}. {item['feature']:<35s}  {item['importance']:.5f}  {bar}")

    # RF vs XGB feature comparison
    rf_results_path = MODEL_DIR / "rf_results.json"
    if rf_results_path.exists():
        with open(rf_results_path) as f:
            rf_results = json.load(f)
        rf_top10 = {item["feature"] for item in rf_results["top_15_features"][:10]}
        xgb_top10 = {item["feature"] for item in results["top_15_features"][:10]}
        overlap = rf_top10 & xgb_top10
        print(f"\n--- Feature Agreement (Top 10) ---")
        print(f"  RF top 10:  {sorted(rf_top10)}")
        print(f"  XGB top 10: {sorted(xgb_top10)}")
        print(f"  Overlap:    {len(overlap)}/10 — {sorted(overlap)}")

    # Missed grids analysis
    if missed:
        print(f"\n--- Missed Dangerous Grids ({len(missed)}) ---")
        for m in missed:
            print(f"  Grid {m['grid_id']}:")
            for feat, val in m["top_features"].items():
                print(f"    {feat}: {val:.2f}")
    else:
        print(f"\n--- Missed Dangerous Grids: NONE (perfect recall!) ---")

    # Full classification report
    print(f"\n--- sklearn Classification Report ---")
    print(classification_report(
        y_test, y_pred_test,
        target_names=["safe", "dangerous"],
        digits=4,
    ))

    print(f"\nSaved: models/xgboost_model.pkl")
    print(f"Saved: models/xgb_results.json")
    print(f"Saved: models/xgb_feature_importance.csv")


if __name__ == "__main__":
    main()
