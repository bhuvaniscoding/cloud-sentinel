# src/features.py
from typing import List
import pandas as pd
from config.settings import RAW_METRICS, ROLLING_WINDOW_SIZE

def get_feature_columns() -> List[str]:
    """Returns the exact list of engineered feature columns fed into the ML model."""
    cols = []
    for col in RAW_METRICS:
        cols.extend([
            col,
            f"{col}_roll_mean",
            f"{col}_roll_std",
            f"{col}_delta",
        ])
    return cols

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms chronologically sorted telemetry into sliding-window state vectors.
    Uses strictly past observations (causal rolling window) so zero future data leaks.
    """
    if df.empty:
        return df.copy()

    # Ensure strict chronological ordering before rolling math
    feat_df = df.sort_values("id", ascending=True).reset_index(drop=True).copy()

    for col in RAW_METRICS:
        # 1. Causal Rolling Mean (smooths transient 1-tick noise)
        feat_df[f"{col}_roll_mean"] = (
            feat_df[col]
            .rolling(window=ROLLING_WINDOW_SIZE, min_periods=1)
            .mean()
        )
        # 2. Causal Rolling Standard Deviation (captures sudden volatility/burstiness)
        feat_df[f"{col}_roll_std"] = (
            feat_df[col]
            .rolling(window=ROLLING_WINDOW_SIZE, min_periods=1)
            .std()
            .fillna(0.0)
        )
        # 3. First-Order Velocity / Delta (captures rate of jump or leak slope)
        feat_df[f"{col}_delta"] = feat_df[col].diff().fillna(0.0)

    return feat_df