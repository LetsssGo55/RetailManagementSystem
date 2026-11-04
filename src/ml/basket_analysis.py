"""
ArthaMart — Market Basket Analysis: FP-Growth (Phase 4, Step 4.3)
===================================================================
1. Baskets = customer orders → sets of product names (order_line_items)
2. One-hot encode (mlxtend TransactionEncoder) → FP-Growth frequent items
3. Association rules (lift ≥ 1.0), filtered to single-item → single-item
4. Business filter (config basket.*):
       lift ≥ min_lift (1.2) AND confidence ≥ min_confidence (0.3)
5. Join with K-Means tiers (directional: antecedent → consequent):
       consequent SLOW            : BUNDLE        (discount the slow mover)
       consequent FAST            : CROSS-PROMOTE (ride the fast mover —
                                    includes the roadmap's FAST + FAST case)
       otherwise (MEDIUM)         : PROMOTE       (shelf/promo adjacency)
6. weekly_impact ₹ = extra consequent units/week × unit margin
       extra units = antecedent weekly transactions × confidence × (lift − 1)
7. Persist actionable rules into bundle_recommendations
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from dataclasses import dataclass, field

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from ..config import get


@dataclass
class BasketReport:
    n_baskets: int = 0
    n_frequent: int = 0
    n_rules_raw: int = 0
    n_rules_kept: int = 0
    actions: dict = field(default_factory=dict)
    total_weekly_impact: float = 0.0


def load_baskets(conn: sqlite3.Connection) -> list[list[str]]:
    rows = conn.execute(
        """
        SELECT li.order_id, p.name
        FROM order_line_items li JOIN products p USING (product_id)
        JOIN customer_orders o USING (order_id)
        WHERE o.status != 'CANCELLED' AND li.product_id IS NOT NULL
        """
    ).fetchall()
    baskets: dict[int, set] = {}
    for r in rows:
        baskets.setdefault(r["order_id"], set()).add(r["name"])
    return [sorted(b) for b in baskets.values() if len(b) >= 2]


def _unit_margin(row) -> float:
    if row["regime"] == "MRP":
        return float((row["mrp"] or 0.0) - row["wholesale_cost"])
    return float(row["wholesale_cost"]) * float(row["base_margin_pct"]) / 100.0


def weekly_txn_counts(conn, weeks: float) -> dict:
    """One grouped query → {product_name: weekly transaction count}.
    (Replaces the old per-rule COUNT(*) round-trip — N+1 pattern.)"""
    rows = conn.execute(
        """
        SELECT p.name, COUNT(DISTINCT li.order_id) AS n
        FROM order_line_items li JOIN products p USING (product_id)
        GROUP BY p.name
        """
    ).fetchall()
    return {r["name"]: r["n"] / max(weeks, 1e-9) for r in rows}


def run_basket_analysis(conn: sqlite3.Connection,
                        save: bool = True) -> tuple[BasketReport, pd.DataFrame]:
    report = BasketReport()

    baskets = load_baskets(conn)
    report.n_baskets = len(baskets)
    if not baskets:
        return report, pd.DataFrame()

    weeks = max(len(baskets) / 450.0, 26.0)      # ~3000 orders ≈ 26 weeks

    te = TransactionEncoder()
    X = pd.DataFrame(te.fit_transform(baskets), columns=te.columns_)

    freq = fpgrowth(X, min_support=float(get("basket.min_support", 0.03)),
                    use_colnames=True)
    report.n_frequent = len(freq)
    if freq.empty:
        return report, pd.DataFrame()

    rules = association_rules(freq, metric="lift", min_threshold=1.0,
                              num_itemsets=len(freq))
    report.n_rules_raw = len(rules)

    # single-item → single-item only
    rules = rules[(rules["antecedents"].str.len() == 1) &
                  (rules["consequents"].str.len() == 1)].copy()

    min_lift = float(get("basket.min_lift", 1.2))
    min_conf = float(get("basket.min_confidence", 0.3))
    rules = rules[(rules["lift"] >= min_lift) &
                  (rules["confidence"] >= min_conf)].copy()

    # ---- tier join + actions -------------------------------------------
    tiers = dict(conn.execute(
        "SELECT p.name, c.tier FROM ml_cluster_assignment c "
        "JOIN products p USING (product_id)"))
    products = {r["name"]: r for r in conn.execute(
        "SELECT name, regime, mrp, wholesale_cost, base_margin_pct "
        "FROM products").fetchall()}
    name_to_id = dict(conn.execute("SELECT name, product_id FROM products"))

    rules["antecedent"] = rules["antecedents"].apply(lambda s: next(iter(s)))
    rules["consequent"] = rules["consequents"].apply(lambda s: next(iter(s)))

    weekly = weekly_txn_counts(conn, weeks)     # single grouped query

    kept = []
    for _, r in rules.iterrows():
        ta = tiers.get(r["antecedent"])
        tc = tiers.get(r["consequent"])
        if not ta or not tc:
            continue
        if tc == "SLOW":
            action = "BUNDLE"            # discount the slow mover
        elif tc == "FAST":
            action = "CROSS-PROMOTE"     # includes roadmap FAST+FAST case
        else:
            action = "PROMOTE"           # MEDIUM consequent

        margin = _unit_margin(products[r["consequent"]])
        extra_units = (weekly.get(r["antecedent"], 0.0)
                       * r["confidence"] * (r["lift"] - 1.0))
        impact = round(extra_units * margin, 2)
        kept.append({
            "antecedent_id": name_to_id[r["antecedent"]],
            "consequent_id": name_to_id[r["consequent"]],
            "support": round(float(r["support"]), 4),
            "confidence": round(float(r["confidence"]), 4),
            "lift": round(float(r["lift"]), 4),
            "action": action,
            "weekly_impact": impact,
        })

    kept_df = pd.DataFrame(kept)
    report.n_rules_kept = len(kept_df)
    if not kept_df.empty:
        report.actions = kept_df["action"].value_counts().to_dict()
        report.total_weekly_impact = float(kept_df["weekly_impact"].sum())

    if save and not kept_df.empty:
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("DELETE FROM bundle_recommendations")
        conn.executemany(
            "INSERT INTO bundle_recommendations "
            "(antecedent_id, consequent_id, support, confidence, lift, "
            " action, weekly_impact, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(int(k["antecedent_id"]), int(k["consequent_id"]),
              k["support"], k["confidence"], k["lift"], k["action"],
              k["weekly_impact"], now) for k in kept])
        conn.commit()

    return report, kept_df
