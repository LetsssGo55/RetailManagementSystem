"""
ArthaMart — MRP-locked Pricing (Phase 6, Step 6.1)
=====================================================
Packaged goods sell at MRP until expiry pressure builds, then walk down
the markdown ladder (config: pricing.mrp_markdown_ladder):

    > 14 days to expiry   →  0%
     8–14 days            → 10%
     4–7  days            → 20%
     0–3  days            → 35%

Cluster tier shifts timing (slow movers need earlier clearance):
    SLOW   −4 days  (markdown starts earlier)
    MEDIUM  0
    FAST   +2 days  (holds full price longer)

The discounted price can never fall below the cost floor
(config: pricing.price_floor_pct = 90% of wholesale cost).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import get

TIER_SHIFT_DAYS = dict(get("pricing.tier_shift_days",
                           {"SLOW": -4, "MEDIUM": 0, "FAST": 2}))


@dataclass
class MrpPrice:
    mrp: float
    price: float
    discount_pct: float
    effective_days: int
    floor_applied: bool
    reason: str


def _ladder() -> list[dict]:
    ladder = list(get("pricing.mrp_markdown_ladder", []))
    return sorted(ladder, key=lambda r: r["max_days"])


def markdown_for_days(days_to_expiry: int) -> float:
    """Discount % from the ladder for effective days remaining."""
    for rung in _ladder():
        if days_to_expiry <= rung["max_days"]:
            return float(rung["discount"])
    return 0.0


def price_mrp(mrp: float, wholesale_cost: float, days_to_expiry: int,
              tier: str | None = None) -> MrpPrice:
    """Apply tier shift + ladder + cost floor to an MRP product."""
    shift = TIER_SHIFT_DAYS.get((tier or "MEDIUM").upper(), 0)
    effective = max(int(days_to_expiry) + shift, 0)
    discount = markdown_for_days(effective)

    raw = mrp * (1.0 - discount / 100.0)
    floor_pct = float(get("pricing.price_floor_pct", 90))
    floor = wholesale_cost * floor_pct / 100.0
    floor_applied = raw < floor
    price = floor if floor_applied else raw

    tier_note = f" ({tier} tier shift {shift:+d}d)" if shift else ""
    if discount == 0:
        reason = f"MRP — {effective}d to expiry, no markdown{tier_note}"
    else:
        reason = (f"expiry markdown {discount:.0f}% at {effective}d "
                  f"remaining{tier_note}")
        if floor_applied:
            reason += f" (floored at {floor_pct:.0f}% of cost)"

    return MrpPrice(mrp=mrp, price=round(price, 2), discount_pct=discount,
                    effective_days=effective, floor_applied=floor_applied,
                    reason=reason)
