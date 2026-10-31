# src/evaluator.py
from typing import Dict, Any
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.database import fetch_recent_metrics
from src.features import extract_features
from src.detector import AnomalyDetector
from src.forecaster import forecast_resource_exhaustion


def run_scientific_evaluation(train_size: int = 300) -> Dict[str, Any]:
    """
    Executes a zero-leakage chronological Train/Test evaluation on synthetic data:
    - Fits scaler & Isolation Forest strictly on rows [0 : train_size] (clean baseline).
    - Extracts features independently on the test split [train_size : end].
    - Compares ML Anomaly Detection vs. Traditional Static Threshold Rules.
    """
    raw_df = fetch_recent_metrics("synthetic", limit=1000)
    if len(raw_df) <= train_size:
        raise ValueError(
            f"Need more than {train_size} synthetic rows. Run `python -m src.simulator` first."
        )

    # Zero-leakage chronological split BEFORE rolling feature extraction
    raw_train = raw_df.iloc[:train_size].copy()
    raw_test = raw_df.iloc[train_size:].copy()

    df_train_feat = extract_features(raw_train)
    df_test_feat = extract_features(raw_test)

    detector = AnomalyDetector()
    train_meta = detector.train(df_train_feat, save=True)
    analyzed_test = detector.analyze(df_test_feat)

    y_true = analyzed_test["is_labeled_anomaly"].values
    y_ml = analyzed_test["ml_anomaly"].values
    y_rule = analyzed_test["rule_alert"].values

    # Compute ML Metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_ml, labels=[0, 1]).ravel()
    ml_metrics = {
        "precision": round(float(precision_score(y_true, y_ml, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_ml, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_ml, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # Compute Traditional Static Rule Metrics (for Viva comparison!)
    rule_metrics = {
        "precision": round(float(precision_score(y_true, y_rule, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_rule, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_rule, zero_division=0)), 4),
    }

    # Per-Scenario Detection Rate Breakdown
    scenario_breakdown = []
    for scenario, group in analyzed_test.groupby("scenario_label"):
        total = len(group)
        ml_caught = int(group["ml_anomaly"].sum())
        rule_caught = int(group["rule_alert"].sum())
        scenario_breakdown.append({
            "scenario": scenario,
            "total_ticks": total,
            "ml_detected": ml_caught,
            "ml_detection_rate_pct": round(100.0 * ml_caught / total, 1),
            "static_rule_detected": rule_caught,
            "static_rule_rate_pct": round(100.0 * rule_caught / total, 1),
        })

    # Validate Predictive Forecaster on the Memory Leak Test Split
    mem_forecast = forecast_resource_exhaustion(analyzed_test, metric="memory_percent")

    return {
        "train_meta": train_meta,
        "ml_metrics": ml_metrics,
        "rule_metrics": rule_metrics,
        "scenario_breakdown": pd.DataFrame(scenario_breakdown),
        "memory_forecast": mem_forecast,
        "analyzed_test_df": analyzed_test,
    }


if __name__ == "__main__":
    results = run_scientific_evaluation()
    print("\n" + "=" * 65)
    print("  CLOUD-SENTINEL: SCIENTIFIC ML vs. STATIC RULES BENCHMARK")
    print("=" * 65)
    print(f"Trained on {results['train_meta']['samples_trained']} clean baseline ticks "
          f"({results['train_meta']['features_count']} rolling features).")
    print("\n[1] Overall Test Split Performance (300 Unseen Ticks):")
    print(f"  • Isolation Forest ML -> Precision: {results['ml_metrics']['precision']*100:.1f}% | "
          f"Recall: {results['ml_metrics']['recall']*100:.1f}% | "
          f"F1-Score: {results['ml_metrics']['f1_score']*100:.1f}%")
    print(f"  • Static Rules (85%)  -> Precision: {results['rule_metrics']['precision']*100:.1f}% | "
          f"Recall: {results['rule_metrics']['recall']*100:.1f}% | "
          f"F1-Score: {results['rule_metrics']['f1_score']*100:.1f}%")
    cm = results["ml_metrics"]["confusion_matrix"]
    print(f"  • ML Confusion Matrix -> TP: {cm['tp']} | FP: {cm['fp']} | TN: {cm['tn']} | FN: {cm['fn']}")

    print("\n[2] Detection Rate by Scenario (Why ML Beats Fixed Thresholds):")
    print(results["scenario_breakdown"].to_string(index=False))

    fc = results["memory_forecast"]
    print("\n[3] Predictive Analytics (Time-to-Exhaustion Forecaster):")
    print(f"  • Status: {fc['status']} | Current RAM: {fc['current_value']}% | "
          f"Growth Rate: +{fc['slope_per_min']}%/min (R²={fc['r_squared']})")
    if fc["tte_minutes"] is not None:
        print(f"  • PREDICTED CRASH WINDOW: RAM hits {fc['critical_limit']}% in {fc['tte_minutes']} minutes!")
    print("=" * 65)