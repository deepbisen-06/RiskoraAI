import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss
)
from ml_engine.features import ALL_FEATURE_COLS

def evaluate_models(
    artifact_dir: str = "ml_engine/artifacts",
    c_fn_default: float = 350.0,
    c_fp_default: float = 175.0
):
    print("Loading artifacts for evaluation...")
    preprocessor = joblib.load(os.path.join(artifact_dir, "preprocessor.joblib"))
    calibrated_xgb = joblib.load(os.path.join(artifact_dir, "model.joblib"))
    baseline_lr = joblib.load(os.path.join(artifact_dir, "baseline_lr.joblib"))
    xgb_uncalibrated = joblib.load(os.path.join(artifact_dir, "xgb_uncalibrated.joblib"))

    val_df = joblib.load(os.path.join(artifact_dir, "val_split.joblib"))
    test_df = joblib.load(os.path.join(artifact_dir, "test_split.joblib"))

    X_val_trans = preprocessor.transform(val_df[ALL_FEATURE_COLS])
    y_val = val_df['is_returned'].values

    X_test_trans = preprocessor.transform(test_df[ALL_FEATURE_COLS])
    y_test = test_df['is_returned'].values

    # Predict probabilities
    val_probs_xgb = calibrated_xgb.predict_proba(X_val_trans)[:, 1]
    test_probs_xgb = calibrated_xgb.predict_proba(X_test_trans)[:, 1]

    val_probs_lr = baseline_lr.predict_proba(X_val_trans)[:, 1]
    test_probs_lr = baseline_lr.predict_proba(X_test_trans)[:, 1]

    # 1. THRESHOLD OPTIMIZATION ON VALIDATION SET ONLY
    print(f"Sweeping business cost curves on VALIDATION set (C_fn=INR {c_fn_default}, C_fp=INR {c_fp_default})...")
    thresholds = np.linspace(0.15, 0.85, 71)
    val_sweep_results = []
    best_profit = -np.inf
    optimal_threshold = 0.50

    for t in thresholds:
        preds = (val_probs_xgb >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        # Net Profit Saved vs no intervention = (RTOs avoided * C_FN) - (Good customers falsely blocked * C_FP)
        net_profit_saved = (tp * c_fn_default) - (fp * c_fp_default)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        
        val_sweep_results.append({
            "threshold": round(float(t), 2),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "net_profit_saved_inr": round(float(net_profit_saved), 2)
        })

        if net_profit_saved > best_profit:
            best_profit = net_profit_saved
            optimal_threshold = round(float(t), 2)

    low_threshold = round(optimal_threshold * 0.45, 2)
    print(f"Optimal Operating Threshold T* found on Validation: {optimal_threshold} (Low Threshold T_low: {low_threshold})")
    print(f"Validation Profit Saved at T*: INR {best_profit:,.2f}")

    # 2. HELD-OUT TEST SET EVALUATION (TOUCHED ONCE)
    print("Evaluating final metrics on held-out TEST set at T*...")
    test_preds_xgb = (test_probs_xgb >= optimal_threshold).astype(int)
    test_preds_lr = (test_probs_lr >= optimal_threshold).astype(int)

    # XGBoost metrics
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds_xgb, labels=[0, 1]).ravel()
    prec_xgb = precision_score(y_test, test_preds_xgb, zero_division=0)
    rec_xgb = recall_score(y_test, test_preds_xgb, zero_division=0)
    f1_xgb = f1_score(y_test, test_preds_xgb, zero_division=0)
    roc_auc_xgb = roc_auc_score(y_test, test_probs_xgb)
    brier_xgb = brier_score_loss(y_test, test_probs_xgb)

    p_curve, r_curve, _ = precision_recall_curve(y_test, test_probs_xgb)
    pr_auc_xgb = auc(r_curve, p_curve)

    fpr_curve, tpr_curve, _ = roc_curve(y_test, test_probs_xgb)

    test_profit_saved = (tp * c_fn_default) - (fp * c_fp_default)

    # Baseline LR metrics
    tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y_test, test_preds_lr, labels=[0, 1]).ravel()
    prec_lr = precision_score(y_test, test_preds_lr, zero_division=0)
    rec_lr = recall_score(y_test, test_preds_lr, zero_division=0)
    f1_lr = f1_score(y_test, test_preds_lr, zero_division=0)
    roc_auc_lr = roc_auc_score(y_test, test_probs_lr)
    p_curve_lr, r_curve_lr, _ = precision_recall_curve(y_test, test_probs_lr)
    pr_auc_lr = auc(r_curve_lr, p_curve_lr)
    test_profit_saved_lr = (tp_lr * c_fn_default) - (fp_lr * c_fp_default)

    # 3. Global Feature Importances
    feature_names = []
    # numerical features
    feature_names.extend(preprocessor.named_transformers_['num'].get_feature_names_out())
    # categorical features
    feature_names.extend(preprocessor.named_transformers_['cat'].get_feature_names_out())
    
    importances = xgb_uncalibrated.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        {"feature": str(feature_names[i]), "importance": round(float(importances[i]), 4)}
        for i in sorted_idx[:10]
    ]

    # 4. Downsample curves for JSON storage
    pr_step = max(1, len(r_curve) // 40)
    pr_points = [
        {"recall": round(float(r), 4), "precision": round(float(p), 4)}
        for r, p in zip(r_curve[::pr_step], p_curve[::pr_step])
    ]

    roc_step = max(1, len(fpr_curve) // 40)
    roc_points = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
        for f, t in zip(fpr_curve[::roc_step], tpr_curve[::roc_step])
    ]

    # Load datacard if exists
    datacard_path = os.path.join(artifact_dir, "datacard.json")
    datacard = {}
    if os.path.exists(datacard_path):
        with open(datacard_path, "r", encoding="utf-8") as f:
            datacard = json.load(f)

    # 5. Assemble Comprehensive Metrics Payload
    metrics_payload = {
        "model_name": "Calibrated XGBoost (Isotonic)",
        "baseline_name": "Logistic Regression Baseline",
        "optimal_threshold": optimal_threshold,
        "low_threshold": low_threshold,
        "cost_assumptions": {
            "c_fn_rto_cost_inr": c_fn_default,
            "c_fp_friction_cost_inr": c_fp_default,
            "upi_conversion_efficiency": 0.35
        },
        "held_out_test_metrics": {
            "test_sample_size": len(test_df),
            "return_prevalence": round(float(y_test.mean()), 4),
            "precision": round(float(prec_xgb), 4),
            "recall": round(float(rec_xgb), 4),
            "f1_score": round(float(f1_xgb), 4),
            "roc_auc": round(float(roc_auc_xgb), 4),
            "pr_auc": round(float(pr_auc_xgb), 4),
            "brier_score": round(float(brier_xgb), 4),
            "net_profit_saved_inr": round(float(test_profit_saved), 2),
            "confusion_matrix": {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn)
            }
        },
        "baseline_comparison": {
            "precision": round(float(prec_lr), 4),
            "recall": round(float(rec_lr), 4),
            "f1_score": round(float(f1_lr), 4),
            "roc_auc": round(float(roc_auc_lr), 4),
            "pr_auc": round(float(pr_auc_lr), 4),
            "net_profit_saved_inr": round(float(test_profit_saved_lr), 2),
            "lift_over_baseline": {
                "pr_auc_delta": round(float(pr_auc_xgb - pr_auc_lr), 4),
                "profit_saved_delta_inr": round(float(test_profit_saved - test_profit_saved_lr), 2)
            }
        },
        "validation_threshold_sweep": val_sweep_results,
        "pr_curve": pr_points,
        "roc_curve": roc_points,
        "top_global_features": top_features,
        "datacard": datacard
    }

    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    
    print(f"Metrics and evaluation results saved to {metrics_path}")
    print(f"Test PR-AUC: {pr_auc_xgb:.4f} (Baseline: {pr_auc_lr:.4f})")
    print(f"Test Net Profit Saved: INR {test_profit_saved:,.2f}")
    return metrics_payload

if __name__ == "__main__":
    evaluate_models()
