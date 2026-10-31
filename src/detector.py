# src/detector.py
from pathlib import Path
from typing import Dict, Any, List
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config.settings import (
    MODEL_DIR,
    CONTAMINATION_RATE,
    RANDOM_SEED,
    RAW_METRICS,
    THRESHOLDS,
    PERSISTENCE_WINDOW,
)
from src.features import get_feature_columns

SCALER_PATH = MODEL_DIR / "scaler.joblib"
MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
META_PATH = MODEL_DIR / "baseline_meta.joblib"


class AnomalyDetector:
    """
    Unsupervised Isolation Forest + Rule Baseline + Transparent Z-Score Attribution.
    Strictly separates:
      1. rule_alert: Static threshold violations (Traditional monitoring)
      2. ml_anomaly: Point anomalies detected by Isolation Forest
      3. incident_alert: Persistence-filtered anomalies (reduces alert fatigue)
      4. root_cause_summary: Rule-based Z-score attribution (transparent, non-black-box)
    """

    def __init__(self, contamination: float = CONTAMINATION_RATE):
        self.contamination = contamination
        self.feature_cols: List[str] = get_feature_columns()
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=150,
            contamination=self.contamination,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        self.score_threshold_: float = 0.0
        self.baseline_stats_: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def train(self, df_train: pd.DataFrame, save: bool = True) -> Dict[str, Any]:
        """Fits scaler, Isolation Forest, and baseline Z-score statistics on clean training data."""
        if df_train.empty:
            raise ValueError("Training DataFrame cannot be empty.")

        X_train = df_train[self.feature_cols].fillna(0.0).values
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled)

        # Invert decision_function so HIGHER score = MORE anomalous
        raw_scores = -self.model.decision_function(X_scaled)
        # Set threshold at 98.5th percentile of clean baseline to prevent false positives on normal load
        self.score_threshold_ = float(np.percentile(raw_scores, 98.5))

        # Store raw metric mean & std for transparent root-cause summaries
        self.baseline_stats_ = {}
        for col in RAW_METRICS:
            mean_val = float(df_train[col].mean())
            std_val = max(float(df_train[col].std()), 0.5)
            self.baseline_stats_[col] = {"mean": mean_val, "std": std_val}

        self.is_fitted = True

        if save:
            joblib.dump(self.scaler, SCALER_PATH)
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(
                {
                    "score_threshold": self.score_threshold_,
                    "baseline_stats": self.baseline_stats_,
                    "feature_cols": self.feature_cols,
                },
                META_PATH,
            )

        return {
            "samples_trained": len(df_train),
            "features_count": len(self.feature_cols),
            "calibrated_threshold": round(self.score_threshold_, 4),
        }

    def load(self) -> bool:
        """Loads saved model artifacts from disk if available."""
        if SCALER_PATH.exists() and MODEL_PATH.exists() and META_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
            self.model = joblib.load(MODEL_PATH)
            meta = joblib.load(META_PATH)
            self.score_threshold_ = meta["score_threshold"]
            self.baseline_stats_ = meta["baseline_stats"]
            self.feature_cols = meta["feature_cols"]
            self.is_fitted = True
            return True
        return False

    def analyze(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Runs both Static Rule Checks and ML Anomaly Inference on engineered features.
        """
        if df_features.empty:
            return df_features.copy()

        if not self.is_fitted and not self.load():
            raise RuntimeError("Model is not trained yet. Call train() first.")

        result = df_features.copy()
        X = result[self.feature_cols].fillna(0.0).values
        X_scaled = self.scaler.transform(X)

        # 1. ML Anomaly Score (Higher = more anomalous)
        anomaly_scores = -self.model.decision_function(X_scaled)
        result["anomaly_score"] = np.round(anomaly_scores, 4)

        # Calculate Z-scores across raw metrics relative to learned baseline
        z_matrix = pd.DataFrame(index=result.index)
        for col in RAW_METRICS:
            mean_val = self.baseline_stats_[col]["mean"]
            std_val = self.baseline_stats_[col]["std"]
            z_matrix[col] = (result[col] - mean_val) / std_val

        max_positive_z = z_matrix.max(axis=1)

        # Point ML Anomaly: Exceeds calibrated IForest threshold OR exhibits severe multi-sigma deviation
        raw_ml_flag = (
            (anomaly_scores > self.score_threshold_) & (max_positive_z > 2.2)
        ) | (max_positive_z > 3.8)
        result["ml_anomaly"] = raw_ml_flag.astype(int)

        # 2. Persistence Filter (Operational Incident Alert if >=2 anomalies in last PERSISTENCE_WINDOW ticks)
        result["incident_alert"] = (
            result["ml_anomaly"]
            .rolling(window=PERSISTENCE_WINDOW, min_periods=1)
            .sum()
            >= 2
        ).astype(int)

        # 3. Static Rule Baseline (Traditional fixed threshold monitoring)
        rule_flags = pd.Series(False, index=result.index)
        rule_reasons = []
        for idx, row in result.iterrows():
            triggered = []
            for metric, limit in THRESHOLDS.items():
                if metric in row and row[metric] >= limit:
                    triggered.append(f"{metric}={row[metric]} (>={limit})")
            if triggered:
                rule_flags.loc[idx] = True
            rule_reasons.append(", ".join(triggered) if triggered else "None")

        result["rule_alert"] = rule_flags.astype(int)
        result["rule_violation_detail"] = rule_reasons

        # 4. Transparent Rule-Based Root Cause Summary (Top deviating metrics by Z-score)
        summaries = []
        for idx in result.index:
            if result.loc[idx, "ml_anomaly"] == 1:
                row_z = z_matrix.loc[idx].sort_values(ascending=False)
                top_contributors = [
                    f"{m} (+{val:.1f}σ)" for m, val in row_z.items() if val >= 2.0
                ][:3]
                summaries.append(
                    "Deviating metrics: " + ", ".join(top_contributors)
                    if top_contributors
                    else "Multivariate pattern shift"
                )
            else:
                summaries.append("Normal baseline behavior")

        result["root_cause_summary"] = summaries
        return result