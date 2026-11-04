"""
ArthaMart — Reorder Planning (Phase 5, Step 5.3)
===================================================
Per product:

    priority = stockout_misses × reorder_priority_weight + Σ(7-day forecast)

Products at or below their ROP get a RECOMMENDED purchase order sized by
EOQ plus a procurement alert (CRITICAL when stock ≤ safety stock).

Demand inputs come from the ML forecaster when available
(`ml_forecast_output`), else fall back to the last 28 days of sales —
so this module also works before `train` has run.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3

import numpy as np
import pandas as pd

from ..config import get
from .eoq import annual_demand_from_forecast, holding_cost_per_unit, eoq
from .rop import reorder_point, safety_stock, stock_status


# ---------------------------------------------------------------------------
# demand inputs
# ---------------------------------------------------------------------------

def demand_inputs(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per product: forecast7, avg_daily, sigma, misses, stock, cost."""
    hist = pd.read_sql_query(
        """
        SELECT product_id, SUM(stockout) AS misses
        FROM daily_sales_history GROUP BY product_id
        """, conn)
    stock = pd.read_sql_query(
        "SELECT product_id, stock_on_hand FROM inventory_current", conn)
    prods = pd.read_sql_query(
        "SELECT product_id, name, wholesale_cost FROM products "
        "WHERE is_active = 1", conn)

    fc = pd.read_sql_query(
        "SELECT product_id, SUM(predicted_qty) AS forecast7, "
        "       AVG(rmse) AS sigma "
        "FROM ml_forecast_output GROUP BY product_id", conn)

    df = prods.merge(stock, on="product_id", how="left") \
              .merge(hist, on="product_id", how="left") \
              .merge(fc, on="product_id", how="left")
    df["stock_on_hand"] = df["stock_on_hand"].fillna(0)
    df["misses"] = df["misses"].fillna(0)

    # fallback for products without forecast rows (vectorised, no iterrows)
    if df["forecast7"].isna().any():
        fb = pd.read_sql_query(
            """
            SELECT product_id,
                   AVG(qty_sold) * 7 AS fb_forecast7,
                   AVG(qty_sold * qty_sold)
                     - AVG(qty_sold) * AVG(qty_sold) AS fb_var
            FROM daily_sales_history
            WHERE date >= (SELECT date(MAX(date), '-27 day')
                           FROM daily_sales_history)
            GROUP BY product_id
            """, conn).set_index("product_id")
        fb_sigma = pd.Series(np.sqrt(fb["fb_var"].clip(lower=0)),
                             index=fb.index)
        mask = df["forecast7"].isna()
        pids = df.loc[mask, "product_id"]
        df.loc[mask, "forecast7"] = pids.map(fb["fb_forecast7"]).fillna(0.0)
        df.loc[mask, "sigma"] = pids.map(fb_sigma).fillna(0.0)
    df["forecast7"] = df["forecast7"].fillna(0.0)
    df["sigma"] = df["sigma"].fillna(0.0)
    df["avg_daily"] = df["forecast7"] / 7.0
    return df


# ---------------------------------------------------------------------------
# full plan
# ---------------------------------------------------------------------------

def compute_plan(conn: sqlite3.Connection) -> pd.DataFrame:
    """EOQ / safety stock / ROP / priority / status for every product."""
    df = demand_inputs(conn)
    lead = float(get("inventory.lead_time_days", 3))
    weight = float(get("inventory.reorder_priority_weight", 2))

    df["safety_stock"] = [safety_stock(s, lead) for s in df["sigma"]]
    df["rop"] = [reorder_point(a, s, lead)
                 for a, s in zip(df["avg_daily"], df["sigma"])]
    df["eoq"] = [eoq(annual_demand_from_forecast(f),
                     holding_per_unit=holding_cost_per_unit(c))
                 for f, c in zip(df["forecast7"], df["wholesale_cost"])]
    df["priority"] = df["misses"] * weight + df["forecast7"]
    df["status"] = [stock_status(st, r, s) for st, r, s in
                    zip(df["stock_on_hand"], df["rop"], df["safety_stock"])]
    return df.sort_values("priority", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# reorder cycle: alerts + purchase orders
# ---------------------------------------------------------------------------

def run_reorder_cycle(conn: sqlite3.Connection, save: bool = True) -> dict:
    """Generate procurement alerts + RECOMMENDED purchase orders."""
    plan = compute_plan(conn)
    # no forecast demand → nothing to sell → never raise a purchase order
    plan = plan[plan["forecast7"] > 0]
    due = plan[plan["status"].isin(["REORDER", "CRITICAL"])].copy()
    due["qty"] = [max(int(round(q)), 1) for q in due["eoq"]]

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if save:
        # idempotent: clear the previous auto-generated cycle
        conn.execute("DELETE FROM purchase_orders WHERE status = 'RECOMMENDED'")
        conn.execute("DELETE FROM procurement_alerts "
                     "WHERE alert_type IN ('REORDER', 'CRITICAL')")
        conn.executemany(
            "INSERT INTO purchase_orders "
            "(product_id, qty, eoq, rop, priority, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, 'RECOMMENDED', ?)",
            [(int(r.product_id), int(r.qty), round(float(r.eoq), 1),
              round(float(r.rop), 1), round(float(r.priority), 1), now)
             for r in due.itertuples()])
        conn.executemany(
            "INSERT INTO procurement_alerts "
            "(product_id, alert_type, message, created_at) VALUES (?, ?, ?, ?)",
            [(int(r.product_id), r.status,
              f"{r.name}: stock {int(r.stock_on_hand)} ≤ "
              f"{'safety stock' if r.status == 'CRITICAL' else 'ROP'} "
              f"{r.rop:.1f} — order {int(r.qty)} units", now)
             for r in due.itertuples()])
        conn.commit()

    return {
        "products_planned": len(plan),
        "orders_recommended": len(due),
        "critical": int((due["status"] == "CRITICAL").sum()),
        "reorder": int((due["status"] == "REORDER").sum()),
        "plan": plan,
        "due": due,
    }
