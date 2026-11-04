"""
ArthaMart — Naive Policy (Phase 9, Step 9.1)
===============================================
The "old way" a shop operates without intelligence:

    • reorder a FIXED quantity (50) whenever stock < threshold (20)
    • STATIC prices — MRP forever; bulk priced once from day-0 cost
    • markdown only when ≤ 3 days to expiry (single 35% step)

Config: simulation.naive_*
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import get


@dataclass
class NaiveDecision:
    price: float
    reorder_qty: int
    markdown_active: bool


class NaivePolicy:
    """Fixed-rule baseline policy."""

    name = "naive"

    def __init__(self):
        self.reorder_qty = int(get("simulation.naive_reorder_qty", 50))
        self.threshold = int(get("simulation.naive_reorder_threshold", 20))
        self.markdown_days = int(get("simulation.naive_markdown_days", 3))

    def price(self, regime: str, base_price: float, mrp: float,
              days_to_expiry: int | None, cost_today: float,
              base_margin_pct: float, static_bulk_price: float) -> tuple[float, bool]:
        """Returns (price, markdown_active). Static — cost spikes are NOT
        passed through; only the last-3-days fire-sale touches price."""
        if regime == "MRP":
            if days_to_expiry is not None and days_to_expiry <= self.markdown_days:
                return mrp * 0.65, True           # flat 35% fire sale
            return mrp, False
        return static_bulk_price, False

    def reorder(self, stock: int) -> int:
        return self.reorder_qty if stock < self.threshold else 0
