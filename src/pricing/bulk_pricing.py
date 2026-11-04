"""
ArthaMart — Bulk / Loose Pricing (Phase 6, Step 6.2)
======================================================
Market-linked goods follow the mandi:

    price = wholesale_cost_now × (1 + adjusted_margin)

    adjusted_margin = base_margin + demand_adj + seasonal_adj
                      clamped to base ± bulk_margin_adjust_range

    demand_adj   forecast7 vs trailing-28-day baseline: +1pp per 5% uplift
    seasonal_adj +2pp in festival months
    cost spikes  pass straight through (cost is multiplicative); the daily
                 price move is capped at ±15% vs the last logged price
    floor        never below price_floor_pct of wholesale cost
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import get

# max price move vs previous day (fraction); margin points from demand/festivals
DAILY_MOVE_CAP = float(get("pricing.daily_move_cap_pct", 15)) / 100.0
DEMAND_PP_PER_5PCT = float(get("pricing.demand_pp_per_5pct", 1.0))
SEASONAL_PP = float(get("pricing.seasonal_pp", 2.0))


@dataclass
class BulkPrice:
    cost: float
    price: float
    margin_pct: float
    demand_adj: float
    seasonal_adj: float
    floor_applied: bool
    capped: bool
    reason: str


def demand_adjustment(forecast7: float | None, baseline7: float | None) -> float:
    """Margin points from demand pressure: +1pp per +5% uplift (±clamp by
    the caller). e.g. ratio 1.05 → +1pp, ratio 1.20 → +4pp."""
    if not forecast7 or not baseline7 or baseline7 <= 0:
        return 0.0
    ratio = forecast7 / baseline7
    return (ratio - 1.0) * 20.0


def seasonal_adjustment(is_festival_month: bool) -> float:
    return SEASONAL_PP if is_festival_month else 0.0


def price_bulk(cost: float, base_margin_pct: float,
               demand_adj: float = 0.0, seasonal_adj: float = 0.0,
               prev_price: float | None = None) -> BulkPrice:
    """Cost-plus price with clamped margin adjustment, cap and floor."""
    rng = float(get("pricing.bulk_margin_adjust_range", 8))
    adj = max(min(demand_adj + seasonal_adj, rng), -rng)
    margin = base_margin_pct + adj        # may go negative; floor guards it

    price = cost * (1.0 + margin / 100.0)

    floor_pct = float(get("pricing.price_floor_pct", 90))
    floor = cost * floor_pct / 100.0
    floor_applied = price < floor
    if floor_applied:
        price = floor

    capped = False
    if prev_price and prev_price > 0:
        move = (price - prev_price) / prev_price
        if abs(move) > DAILY_MOVE_CAP:
            price = prev_price * (1 + DAILY_MOVE_CAP * (1 if move > 0 else -1))
            capped = True

    bits = [f"cost ₹{cost:.2f} × (1 + {margin:.1f}% margin)"]
    if demand_adj:
        bits.append(f"demand adj {demand_adj:+.1f}pp")
    if seasonal_adj:
        bits.append(f"festival +{seasonal_adj:.0f}pp")
    if floor_applied:
        bits.append("cost floor applied")
    if capped:
        bits.append(f"daily move capped ±{DAILY_MOVE_CAP:.0%}")

    return BulkPrice(cost=cost, price=round(price, 2), margin_pct=margin,
                     demand_adj=demand_adj, seasonal_adj=seasonal_adj,
                     floor_applied=floor_applied, capped=capped,
                     reason=", ".join(bits))
