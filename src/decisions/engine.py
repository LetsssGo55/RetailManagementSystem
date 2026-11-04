"""
ArthaMart — Decision Card Engine (Phase 8, Step 8.2)
======================================================
Pulls the persisted outputs of Phases 4–6 and emits decision cards:

    URGENT_MARKDOWN  ← pricing ladder (expiry) + cluster tier + stock
    INVEST_REORDER   ← Phase-5 plan (REORDER/CRITICAL) ∩ FAST tier
    HOLD             ← SLOW tier comfortably above ROP (don't tie up cash)
    BUNDLE           ← bundle_recommendations (BUNDLE / CROSS-PROMOTE)

Cards are sorted by ₹ impact, descending.
"""
from __future__ import annotations

import sqlite3

from ..config import get
from ..inventory.reorder import compute_plan
from ..pricing.engine import estimate_days_to_expiry, price_product
from .cards import (BUNDLE, HOLD, INVEST_REORDER, URGENT_MARKDOWN,
                    DecisionCard)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _plan_with_tiers(conn: sqlite3.Connection):
    """Phase-5 plan joined with cluster tier (MEDIUM where unassigned)."""
    plan = compute_plan(conn)
    tiers = {r["product_id"]: r["tier"] for r in conn.execute(
        "SELECT product_id, tier FROM ml_cluster_assignment")}
    plan["tier"] = [tiers.get(int(pid), "MEDIUM") for pid in plan["product_id"]]
    return plan


# ---------------------------------------------------------------------------
# card generators
# ---------------------------------------------------------------------------

def markdown_cards(conn: sqlite3.Connection) -> list[DecisionCard]:
    """Near-expiry MRP items in SLOW/MEDIUM tiers that need a markdown."""
    cards = []
    rows = conn.execute(
        """
        SELECT p.product_id, p.name, p.mrp, p.shelf_life_days,
               c.tier, COALESCE(i.stock_on_hand, 0) AS stock
        FROM products p
        LEFT JOIN ml_cluster_assignment c USING (product_id)
        LEFT JOIN inventory_current i USING (product_id)
        WHERE p.regime = 'MRP' AND p.is_active = 1
        """
    ).fetchall()

    for r in rows:
        tier = r["tier"] or "MEDIUM"
        if tier not in ("SLOW", "MEDIUM"):
            continue
        dte = estimate_days_to_expiry(conn, r["product_id"],
                                      r["shelf_life_days"])
        if dte is None:
            continue
        priced = price_product(conn, r["product_id"],
                               days_to_expiry=dte)
        if priced.discount_pct <= 0 or r["stock"] <= 0:
            continue
        recovered = float(r["stock"]) * priced.price
        cards.append(DecisionCard(
            card_type=URGENT_MARKDOWN,
            product_id=r["product_id"],
            product_name=r["name"],
            action=(f"Apply {priced.discount_pct:.0f}% markdown now "
                    f"(MRP ₹{priced.base_price:.0f} → ₹{priced.price:.0f})"),
            reason=(f"{dte} days to expiry ({tier} tier), "
                    f"{int(r['stock'])} units at risk of total write-off"),
            impact_inr=round(recovered, 2),
            detail={"days_to_expiry": dte, "stock": int(r["stock"]),
                    "discount_pct": priced.discount_pct, "tier": tier},
        ))
    return cards


def reorder_cards(conn: sqlite3.Connection) -> list[DecisionCard]:
    """FAST movers about to stock out — protect the forecast revenue."""
    plan = _plan_with_tiers(conn)
    prices = {r["product_id"]: r["price"] for r in conn.execute(
        "SELECT product_id, price FROM price_history_log "
        "WHERE price_date = (SELECT MAX(price_date) FROM price_history_log)")}
    cards = []
    for row in plan.itertuples():
        if row.status not in ("REORDER", "CRITICAL"):
            continue
        if row.tier != "FAST":
            continue
        unit_price = prices.get(int(row.product_id))
        if not unit_price:
            unit_price = float(row.wholesale_cost) * 1.2
        protected = float(row.forecast7) * float(unit_price)
        cards.append(DecisionCard(
            card_type=INVEST_REORDER,
            product_id=int(row.product_id),
            product_name=row.name,
            action=(f"Order {max(int(round(row.eoq)), 1)} units now "
                    f"(EOQ); stock {int(row.stock_on_hand)} ≤ "
                    f"ROP {row.rop:.0f}"),
            reason=(f"FAST mover: ₹{protected:,.0f} of 7-day forecast "
                    f"revenue at risk of stockout"),
            impact_inr=round(protected, 2),
            detail={"eoq": round(float(row.eoq), 1),
                    "rop": round(float(row.rop), 1),
                    "stock": int(row.stock_on_hand),
                    "status": row.status},
        ))
    return cards


def hold_cards(conn: sqlite3.Connection) -> list[DecisionCard]:
    """SLOW movers sitting above ROP — don't order more; cash is better
    deployed elsewhere. Impact = monthly holding cost of the excess stock."""
    plan = _plan_with_tiers(conn)
    holding_pct = float(get("inventory.holding_cost_pct", 0.18))
    cards = []
    for row in plan.itertuples():
        if row.tier != "SLOW" or row.status != "OK":
            continue
        excess = float(row.stock_on_hand) - float(row.rop)
        if excess <= 0:
            continue
        monthly_holding = excess * float(row.wholesale_cost) * holding_pct / 12
        cover = (float(row.stock_on_hand) / float(row.avg_daily)
                 if row.avg_daily > 0 else 999)
        cards.append(DecisionCard(
            card_type=HOLD,
            product_id=int(row.product_id),
            product_name=row.name,
            action="Do not reorder this week",
            reason=(f"SLOW mover with {cover:.0f} days of cover — "
                    f"{int(excess)} units above ROP tie up cash "
                    f"(₹{monthly_holding:,.0f}/month holding cost)"),
            impact_inr=round(monthly_holding, 2),
            detail={"stock": int(row.stock_on_hand),
                    "rop": round(float(row.rop), 1),
                    "days_of_cover": round(cover, 1)},
        ))
    return cards


def bundle_cards(conn: sqlite3.Connection) -> list[DecisionCard]:
    """FP-Growth association rules worth acting on."""
    discount = float(get("basket.bundle_discount_pct", 10))
    names = dict(conn.execute("SELECT product_id, name FROM products"))
    rows = conn.execute(
        "SELECT * FROM bundle_recommendations "
        "WHERE action IN ('BUNDLE', 'CROSS-PROMOTE') "
        "ORDER BY weekly_impact DESC").fetchall()
    cards = []
    for r in rows:
        a = names.get(r["antecedent_id"], f"#{r['antecedent_id']}")
        c = names.get(r["consequent_id"], f"#{r['consequent_id']}")
        if r["action"] == "BUNDLE":
            action = (f"Sell '{a}' + '{c}' as a bundle at {discount:.0f}% "
                      "off the pair to clear slow stock")
        else:
            action = (f"Cross-promote '{a}' with '{c}' "
                      "(adjacent shelf placement, no discount)")
        cards.append(DecisionCard(
            card_type=BUNDLE,
            product_id=r["antecedent_id"],
            product_name=f"{a} → {c}",
            action=action,
            reason=(f"lift {r['lift']:.2f}, confidence "
                    f"{r['confidence']:.0%} from {get_basket_count(conn)} "
                    "baskets (FP-Growth)"),
            impact_inr=round(float(r["weekly_impact"]), 2),
            detail={"lift": r["lift"], "confidence": r["confidence"],
                    "action_type": r["action"]},
        ))
    return cards


def get_basket_count(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(DISTINCT order_id) FROM order_line_items "
        "WHERE product_id IS NOT NULL").fetchone()[0]


# ---------------------------------------------------------------------------
# engine entry point
# ---------------------------------------------------------------------------

def generate_all(conn: sqlite3.Connection) -> list[DecisionCard]:
    """All decision cards, sorted by ₹ impact (descending)."""
    cards: list[DecisionCard] = []
    cards += markdown_cards(conn)
    cards += reorder_cards(conn)
    cards += hold_cards(conn)
    cards += bundle_cards(conn)
    cards.sort(key=lambda c: c.impact_inr, reverse=True)
    return cards
