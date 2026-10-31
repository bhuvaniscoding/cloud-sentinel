# src/forecaster.py
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from config.settings import COLLECTION_INTERVAL_SEC, FORECAST_HORIZON_MIN


def forecast_resource_exhaustion(
    df: pd.DataFrame,
    metric: str = "memory_percent",
    critical_limit: float = 95.0,
    lookback_ticks: int = 75,
    forecast_steps: int = 60,
) -> Dict[str, Any]:
    """
    Combines Linear Trend Regression (with R^2 confidence validation) and
    Holt's Linear Exponential Smoothing to forecast future utilization and
    estimate Time-to-Exhaustion (TTE) in minutes.
    """
    if df.empty or metric not in df.columns or len(df) < 15:
        return {
            "metric": metric,
            "status": "INSUFFICIENT_DATA",
            "current_value": 0.0,
            "slope_per_min": 0.0,
            "r_squared": 0.0,
            "tte_minutes": None,
            "warning_active": False,
            "forecast_values": [],
        }

    recent = df.tail(lookback_ticks).copy()
    y = recent[metric].astype(float).values
    n = len(y)
    x = np.arange(n, dtype=float)

    # 1. Ordinary Least Squares Linear Trend (y = m*x + c)
    slope_per_tick, intercept = np.polyfit(x, y, 1)

    # Compute R^2 (Coefficient of Determination) to filter out random noise spikes
    y_pred_line = slope_per_tick * x + intercept
    ss_res = float(np.sum((y - y_pred_line) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    # 2. Holt's Linear Exponential Smoothing for short-term trajectory projection
    alpha, beta = 0.35, 0.15
    level = y[0]
    trend = y[1] - y[0]
    for val in y[1:]:
        prev_level = level
        level = alpha * val + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend

    # Blended positive slope (combines global regression slope and local Holt trend)
    effective_slope_tick = 0.6 * slope_per_tick + 0.4 * trend
    ticks_per_min = 60.0 / COLLECTION_INTERVAL_SEC
    slope_per_min = effective_slope_tick * ticks_per_min
    current_val = float(y[-1])

    # Generate projected future trajectory for dashboard plotting
    future_steps = np.arange(1, forecast_steps + 1, dtype=float)
    forecast_curve: List[float] = list(
        np.clip(current_val + effective_slope_tick * future_steps, 0.0, 100.0).round(2)
    )

    # Calculate Time-to-Exhaustion (TTE)
    tte_minutes = None
    warning_active = False
    status = "STABLE"

    if current_val >= critical_limit:
        tte_minutes = 0.0
        warning_active = True
        status = "CRITICAL_EXHAUSTED"
    elif slope_per_min > 0.15 and r_squared >= 0.55:
        tte_minutes = round((critical_limit - current_val) / slope_per_min, 2)
        if tte_minutes <= FORECAST_HORIZON_MIN:
            warning_active = True
            status = "EXHAUSTION_WARNING"
        else:
            status = "GRADUAL_GROWTH"

    return {
        "metric": metric,
        "status": status,
        "current_value": round(current_val, 2),
        "critical_limit": critical_limit,
        "slope_per_min": round(slope_per_min, 2),
        "r_squared": round(r_squared, 3),
        "tte_minutes": tte_minutes,
        "warning_active": warning_active,
        "forecast_values": forecast_curve,
    }