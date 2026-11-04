"""
ArthaMart — Forecast error metrics (Phase 4).
"""
from __future__ import annotations

import numpy as np


def rmse(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, predicted) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(a - p)))


def mape(actual, predicted, eps: float = 1e-8) -> float:
    """Mean absolute percentage error. Zero actuals are skipped (standard
    retail practice — MAPE on stockout/zero days is undefined)."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    mask = a > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100.0)


def all_metrics(actual, predicted) -> dict:
    return {"rmse": rmse(actual, predicted),
            "mae": mae(actual, predicted),
            "mape": mape(actual, predicted)}
