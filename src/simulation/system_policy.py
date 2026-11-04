"""
ArthaMart — System Policy (Phase 9, Step 9.2)
================================================
The ArthaMart intelligent policy:

    • reorder at forecast-driven ROP, in EOQ quantities (Phase 5)
    • bulk prices = today's cost × (1 + adjusted margin) (Phase 6),
      festival-aware, clamped to ±8pp
    • cluster-tier markdown ladder — SLOW items discount earlier (Phase 6)
    • bundle / cross-promote demand uplift from FP-Growth rules (Phase 4)
"""
from __future__ import annotations

from ..config import get
from ..pricing.mrp_pricing import TIER_SHIFT_DAYS, markdown_for_days

FESTIVAL_MONTHS = set(get("calendar.festival_months", [8, 10, 11, 1]))
SEASONAL_PP = float(get("pricing.seasonal_pp", 2.0))


class SystemPolicy:
    """ML-driven policy. All plan inputs are computed once up-front
    (the way the live system does it) and reused across the 180 days."""

    name = "system"

    def __init__(self, plans: dict, uplift: dict):
        """
        plans  : product_id → dict(rop, eoq, margin_adj)
        uplift : product_id → (uplift_pct, bundle_discount_pct)
        """
        self.plans = plans
        self.uplift = uplift

    def price(self, regime: str, base_price: float, mrp: float,
              days_to_expiry: int | None, cost_today: float,
              base_margin_pct: float, static_bulk_price: float,
              tier: str = "MEDIUM", month: int = 1,
              stock: float = 0.0, avg_daily: float = 0.0) -> tuple[float, bool]:
        """Returns (price, markdown_active).

        Intelligence gate: the expiry ladder fires ONLY when surplus units
        are at risk — i.e. stock exceeds the demand that will naturally
        occur before expiry (dte × avg_daily). Otherwise margin is not
        sacrificed: the batch sells through at full price in time.
        """
        if regime == "MRP":
            if days_to_expiry is None:
                return mrp, False
            if stock <= days_to_expiry * avg_daily:
                return mrp, False               # sells through before expiry
            shift = TIER_SHIFT_DAYS.get(tier, 0)
            effective = max(days_to_expiry + shift, 0)
            discount = markdown_for_days(effective)
            if discount <= 0:
                return mrp, False
            return mrp * (1 - discount / 100.0), True

        # BULK: cost-plus with clamped margin adjustment
        rng = float(get("pricing.bulk_margin_adjust_range", 8))
        plan = self.plans or {}
        adj = 0.0
        seasonal = SEASONAL_PP if month in FESTIVAL_MONTHS else 0.0
        adj = max(min(seasonal, rng), -rng)
        margin = base_margin_pct + adj
        floor = cost_today * float(get("pricing.price_floor_pct", 90)) / 100.0
        return max(cost_today * (1 + margin / 100.0), floor), False

    def reorder(self, product_id: int, stock: int) -> int:
        plan = self.plans.get(product_id)
        if plan is None:
            return 0
        if stock <= plan["rop"]:
            return max(int(round(plan["eoq"])), 1)
        return 0

    def demand_uplift(self, product_id: int) -> tuple[float, float]:
        """(demand uplift fraction, bundle discount fraction)."""
        return self.uplift.get(product_id, (0.0, 0.0))
