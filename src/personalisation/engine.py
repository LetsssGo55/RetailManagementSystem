"""
ArthaMart — Personalisation Engine
====================================
Completes the "…and Personalisation" pillar of the project title.

Segmentation (config: personalisation.*):
    VIP      total spend ≥ vip_threshold_spend (₹5,000)
    AT_RISK  inactive ≥ at_risk_days_inactive (21 days)
    REGULAR  ≥ 3 orders
    NEW      everyone else

Recommendations (collaborative, explainable):
    1. lift-weighted cross-sell from FP-Growth rules whose antecedent the
       customer already bought (lift ≥ cross_sell_lift_threshold)
    2. velocity prior so the list always fills to top_n
    3. already-purchased products are never recommended
    Customers below min_purchase_history get FAST-tier bestsellers instead.

Outputs persist to `personalisation_profiles` and refresh the summary
columns on `customers`.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3

import pandas as pd

from ..config import get

TIERS = list(get("personalisation.segmentation_tiers",
                 ["NEW", "REGULAR", "VIP", "AT_RISK"]))


# ---------------------------------------------------------------------------
# stats + segmentation
# ---------------------------------------------------------------------------

def customer_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """Spend / visits / recency / preferred category per customer.

    Implemented with two flat set-based queries + pandas aggregation —
    a per-row correlated subquery here is O(customers × line_items) and
    takes minutes on 3k orders; this version runs in milliseconds.
    """
    customers = pd.read_sql_query(
        "SELECT customer_id, name FROM customers ORDER BY customer_id", conn)
    lines = pd.read_sql_query(
        "SELECT o.customer_id, li.line_total, p.category "
        "FROM customer_orders o "
        "JOIN order_line_items li USING (order_id) "
        "JOIN products p USING (product_id) "
        "WHERE li.product_id IS NOT NULL", conn)
    orders = pd.read_sql_query(
        "SELECT customer_id, order_id, order_date FROM customer_orders", conn)

    spend = lines.groupby("customer_id")["line_total"].sum()
    visits = orders.groupby("customer_id")["order_id"].nunique()
    last_visit = orders.groupby("customer_id")["order_date"].max()
    cat_spend = (lines.groupby(["customer_id", "category"])["line_total"]
                 .sum().reset_index())
    preferred = (cat_spend.sort_values("line_total")
                 .groupby("customer_id").tail(1)
                 .set_index("customer_id")["category"])

    df = customers.set_index("customer_id")
    df["spend"] = spend.reindex(df.index).fillna(0.0)
    df["visits"] = visits.reindex(df.index).fillna(0).astype(int)
    df["last_visit"] = last_visit.reindex(df.index)
    df["preferred_cat"] = preferred.reindex(df.index)
    return df.reset_index()


def segment_for(spend: float, visits: int, recency_days: float) -> str:
    vip = float(get("personalisation.vip_threshold_spend", 5000))
    risk = float(get("personalisation.at_risk_days_inactive", 21))
    if spend >= vip:
        return "VIP"
    if recency_days > risk and spend > 0:
        return "AT_RISK"
    if visits >= 3:
        return "REGULAR"
    return "NEW"


def compute_segments(conn, ref_date: str | None = None) -> pd.DataFrame:
    df = customer_stats(conn)
    # "now" = end of the business data window (demo/ad-hoc orders placed on
    # later dates must not skew recency)
    ref = ref_date or conn.execute(
        "SELECT MAX(date) FROM daily_sales_history").fetchone()[0]
    ref = pd.to_datetime(ref or _dt.date.today().isoformat())
    df["recency_days"] = [
        (ref - pd.to_datetime(d)).days if d is not None else 9999
        for d in df["last_visit"]]
    df["segment"] = [segment_for(sp, vi, rc) for sp, vi, rc in
                     zip(df["spend"], df["visits"], df["recency_days"])]
    return df


# ---------------------------------------------------------------------------
# recommendations
# ---------------------------------------------------------------------------

def _purchased(conn, customer_id: int) -> set[int]:
    rows = conn.execute(
        "SELECT DISTINCT li.product_id FROM order_line_items li "
        "JOIN customer_orders o USING (order_id) "
        "WHERE o.customer_id = ? AND li.product_id IS NOT NULL",
        (customer_id,)).fetchall()
    return {r[0] for r in rows}


def recommend(conn, customer_id: int, top_n: int | None = None) -> list[dict]:
    """Top-N personalised picks with a plain-language reason each."""
    top_n = top_n or int(get("personalisation.top_n", 5))
    min_hist = int(get("personalisation.min_purchase_history", 2))
    lift_min = float(get("personalisation.cross_sell_lift_threshold", 1.5))

    bought = _purchased(conn, customer_id)
    names = dict(conn.execute("SELECT product_id, name FROM products"))
    velocity = dict(conn.execute(
        "SELECT product_id, velocity FROM ml_cluster_assignment"))

    if len(bought) < min_hist:
        # cold start → bestsellers from the FAST tier
        rows = conn.execute(
            "SELECT p.product_id, p.name FROM ml_cluster_assignment c "
            "JOIN products p USING (product_id) WHERE c.tier = 'FAST' "
            "ORDER BY c.velocity DESC LIMIT ?", (top_n,)).fetchall()
        return [{"product_id": r[0], "name": r[1], "score": 0.0,
                 "reason": "Bestseller in your store (new customer)"}
                for r in rows if r[0] not in bought][:top_n]

    scores: dict[int, float] = {}
    reasons: dict[int, str] = {}
    rules = conn.execute(
        "SELECT antecedent_id, consequent_id, confidence, lift "
        "FROM bundle_recommendations WHERE lift >= ?", (lift_min,)).fetchall()
    for r in rules:
        if r["antecedent_id"] in bought and r["consequent_id"] not in bought:
            add = float(r["confidence"]) * float(r["lift"])
            scores[r["consequent_id"]] = (
                scores.get(r["consequent_id"], 0.0) + add)
            reasons.setdefault(
                r["consequent_id"],
                f"Shoppers who buy {names.get(r['antecedent_id'], '?')} "
                f"often add this (lift {r['lift']:.1f}×)")

    # velocity prior so the list always fills
    for pid, vel in velocity.items():
        if pid not in bought and pid not in scores:
            scores[pid] = 0.05 * float(vel)
            reasons.setdefault(pid, "Fast-moving staple in your store")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [{"product_id": pid, "name": names.get(pid, f"#{pid}"),
             "score": round(sc, 3), "reason": reasons.get(pid, "")}
            for pid, sc in ranked][:top_n]


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def build_profiles(conn, save: bool = True) -> pd.DataFrame:
    """Segment everyone + persist profiles and customer summary columns."""
    df = compute_segments(conn)
    if save:
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in df.itertuples():
            recs = recommend(conn, int(r.customer_id))
            conn.execute(
                "INSERT OR REPLACE INTO personalisation_profiles "
                "(customer_id, segment, preferred_cat, rec_products, "
                " last_updated) VALUES (?, ?, ?, ?, ?)",
                (int(r.customer_id), r.segment, r.preferred_cat or "-",
                 ",".join(str(x["product_id"]) for x in recs), now))
            conn.execute(
                "UPDATE customers SET segment = ?, total_spend = ?, "
                "visit_count = ?, last_visit = ? WHERE customer_id = ?",
                (r.segment, float(r.spend), int(r.visits), r.last_visit,
                 int(r.customer_id)))
        conn.commit()
    return df
