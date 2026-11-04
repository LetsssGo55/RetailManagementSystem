"""
ArthaMart — Economic Order Quantity (Phase 5, Step 5.1)
=========================================================
    EOQ = √(2 × D × S / H)

    D   annual demand        = 7-day forecast total × 52 weeks
    S   ordering cost ₹      (config: inventory.ordering_cost)
    H   holding cost ₹/unit/yr = holding_cost_pct × unit cost
"""
from __future__ import annotations

import math

from ..config import get


def annual_demand_from_forecast(forecast_7d_total: float) -> float:
    """Weekly demand (7-day forecast) × 52 weeks."""
    return max(float(forecast_7d_total), 0.0) * 52.0


def holding_cost_per_unit(unit_cost: float,
                          holding_pct: float | None = None) -> float:
    """holding_pct is a FRACTION — config 0.18 means 18% of unit cost/year."""
    pct = holding_pct if holding_pct is not None \
        else float(get("inventory.holding_cost_pct", 0.18))
    return max(float(unit_cost) * pct, 1e-9)


def eoq(annual_demand: float, ordering_cost: float | None = None,
        holding_per_unit: float | None = None) -> float:
    """Classic Wilson EOQ. Returns 0 when demand is zero."""
    if annual_demand <= 0:
        return 0.0
    s = ordering_cost if ordering_cost is not None \
        else float(get("inventory.ordering_cost", 150.0))
    h = holding_per_unit if holding_per_unit is not None else 1.0
    if h <= 0:
        return 0.0
    return math.sqrt(2.0 * annual_demand * s / h)


def recommended_order_qty(annual_demand: float, unit_cost: float) -> int:
    """Ceiling-integer EOQ for purchase orders (min 1 unit if demand>0)."""
    q = eoq(annual_demand, holding_per_unit=holding_cost_per_unit(unit_cost))
    return max(int(math.ceil(q)), 1) if annual_demand > 0 else 0
