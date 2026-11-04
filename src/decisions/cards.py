"""
ArthaMart — Decision Card data model (Phase 8, Step 8.1)
==========================================================
Every insight in the system is surfaced as a quantified decision card:

    🚨 URGENT_MARKDOWN    near-expiry MRP item (SLOW/MEDIUM tier)
    📦 INVEST_REORDER     FAST item about to stock out
    ⏸️  HOLD               slow mover over its reorder point
    🎁 BUNDLE             FP-Growth association worth acting on

Design rule from the synopsis: no bare numbers — every card carries an
icon, a plain-language action, a reason, and a ₹ impact figure.
"""
from __future__ import annotations

from dataclasses import dataclass

URGENT_MARKDOWN = "URGENT_MARKDOWN"
INVEST_REORDER = "INVEST_REORDER"
HOLD = "HOLD"
BUNDLE = "BUNDLE"

ICONS = {
    URGENT_MARKDOWN: "🚨",
    INVEST_REORDER: "📦",
    HOLD: "⏸️",
    BUNDLE: "🎁",
}

TITLES = {
    URGENT_MARKDOWN: "Urgent Markdown",
    INVEST_REORDER: "Invest-Reorder",
    HOLD: "Hold",
    BUNDLE: "Bundle Opportunity",
}


@dataclass
class DecisionCard:
    card_type: str
    product_id: int | None
    product_name: str
    action: str                # what the shopkeeper should do
    reason: str                # why — quantified context
    impact_inr: float          # ₹ figure used for sorting
    detail: dict | None = None  # extra fields per card type

    @property
    def icon(self) -> str:
        return ICONS[self.card_type]

    @property
    def title(self) -> str:
        return TITLES[self.card_type]

    def render(self) -> str:
        return (f"{self.icon} {self.title} — {self.product_name}\n"
                f"   do   : {self.action}\n"
                f"   why  : {self.reason}\n"
                f"   value: ₹{self.impact_inr:,.0f}")
