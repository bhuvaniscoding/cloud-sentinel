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


def run_scientific_evaluation(source: str = "synthetic", train_size: int = 300) -> Dict[str, Any]:
    """
    Executes a zero-leakage chronological Train/Test evaluation on `source`:
    - Fits scaler & Isolation Forest strictly on rows [0 : train_size] (clean baseline).
    - Extracts features independently on the test split [train_size : end].
    - Compares ML Anomaly Detection vs. Traditional Static Threshold Rules.
    """
    raw_df = fetch_recent_metrics(source, limit=1000)
    if len(raw_df) <= train_size:
        raise ValueError(
            f"Need more than {train_size} rows for source='{source}'."
        )

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

    tn, fp, fn, tp = confusion_matrix(y_true, y_ml, labels=[0, 1]).ravel()
    ml_metrics = {
        "precision": round(float(precision_score(y_true, y_ml, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_ml, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_ml, zero_division=0)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    rule_metrics = {
        "precision": round(float(precision_score(y_true, y_rule, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_rule, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_rule, zero_division=0)), 4),
    }

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

    mem_forecast = forecast_resource_exhaustion(analyzed_test, metric="memory_percent")

    return {
        "source": source,
        "train_meta": train_meta,
        "ml_metrics": ml_metrics,
        "rule_metrics": rule_metrics,
        "scenario_breakdown": pd.DataFrame(scenario_breakdown),
        "memory_forecast": mem_forecast,
        "analyzed_test_df": analyzed_test,
    }


if __name__ == "__main__":
    results = run_scientific_evaluation("synthetic")
    print(f"Synthetic F1: {results['ml_metrics']['f1_score']*100:.1f}%")