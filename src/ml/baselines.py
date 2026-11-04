"""
ArthaMart — Forecast Baselines (Phase 4, Step 4.1)
====================================================
Two classical baselines the Random Forest must beat:

    naive        rolling 7-day mean of the most recent history
    exp_smooth   Holt–Winters (additive trend + weekly seasonality,
                 statsmodels) with graceful fallback to naive
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def naive_forecast(history: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Constant forecast = mean of the last `window` days."""
    base = float(history.tail(window).mean()) if len(history) else 0.0
    return np.full(horizon, max(base, 0.0))


def exp_smooth_forecast(history: pd.Series, horizon: int,
                        seasonal_period: int = 7) -> np.ndarray:
    """
    Holt–Winters forecast. Falls back to naive when the series is too short
    or the optimiser fails (zero-sales products etc.).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    s = history.astype(float)
    if len(s) < 2 * seasonal_period or s.sum() <= 0:
        return naive_forecast(history, horizon)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                s, trend="add", seasonal="add",
                seasonal_periods=seasonal_period,
                initialization_method="estimated",
            ).fit(optimized=True)
            fc = model.forecast(horizon)
        return np.clip(np.asarray(fc, dtype=float), 0.0, None)
    except Exception:
        return naive_forecast(history, horizon)


def naive_test_predictions(history: pd.Series, test_days: int,
                           window: int = 7) -> np.ndarray:
    """
    Baseline predictions over the historical test window (same evaluation
    protocol as the RF): for each test day, the mean of the `window` days
    immediately before it.
    """
    y = history.astype(float).values
    n = len(y)
    start = n - test_days
    preds = np.empty(test_days)
    for i in range(test_days):
        t = start + i
        lo = max(0, t - window)
        preds[i] = max(np.mean(y[lo:t]), 0.0) if t > lo else 0.0
    return preds
