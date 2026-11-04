"""
ArthaMart — Reorder Point & Safety Stock (Phase 5, Step 5.2)
===============================================================
    safety_stock = z × σ_demand × √lead_time
    ROP          = avg_daily_demand × lead_time + safety_stock

    z          from target service level (config: inventory.service_level_z)
    σ_demand   forecast-residual RMSE (ml_forecast_output) — falls back to
               std of the last 28 days of sales when no model output exists
    lead_time  days (config: inventory.lead_time_days)
"""
from __future__ import annotations

import math

from ..config import get


def safety_stock(sigma_demand: float, lead_time: float | None = None,
                 z: float | None = None) -> float:
    lt = lead_time if lead_time is not None \
        else float(get("inventory.lead_time_days", 3))
    zz = z if z is not None else float(get("inventory.service_level_z", 1.65))
    return zz * max(float(sigma_demand), 0.0) * math.sqrt(max(lt, 0.0))


def reorder_point(avg_daily_demand: float, sigma_demand: float,
                  lead_time: float | None = None,
                  z: float | None = None) -> float:
    lt = lead_time if lead_time is not None \
        else float(get("inventory.lead_time_days", 3))
    return avg_daily_demand * lt + safety_stock(sigma_demand, lt, z)


def stock_status(stock_on_hand: float, rop: float, ss: float) -> str:
    """CRITICAL (≤ safety stock) · REORDER (≤ ROP) · OK."""
    if stock_on_hand <= ss:
        return "CRITICAL"
    if stock_on_hand <= rop:
        return "REORDER"
    return "OK"
