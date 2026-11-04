"""
ArthaMart — Policy Simulation Runner (Phase 9, Step 9.3)
==========================================================
Naive vs System policy over the SAME 180-day demand signal.

Fair-comparison rules:
    • identical demand draw per product per day (historical qty_sold,
      stockout days imputed with the product median)
    • identical wholesale costs, lead times, holding cost, starting stock
    • identical price-elasticity response: demand × (price/base)^−1.2
    • ONLY the decisions differ: reorder timing/qty, pricing, markdowns

Daily P&L components (per schema):
    revenue · cogs (+₹150 ordering cost per PO) · holding_cost ·
    markdown_loss (discounts given + expired-stock write-offs) ·
    stockout_loss (lost margin on unmet demand) · net_profit
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, get
from ..inventory.reorder import compute_plan
from ..pricing.engine import estimate_days_to_expiry
from .naive_policy import NaivePolicy
from .system_policy import SystemPolicy

SIM_DIR = ROOT / "data" / "simulation"


@dataclass
class ProductCtx:
    product_id: int
    name: str
    regime: str
    tier: str
    mrp: float
    base_margin_pct: float
    shelf_life: int | None
    initial_age: int
    initial_stock: float
    cost_series: np.ndarray          # per-day wholesale cost
    base_price: float                # elasticity reference price
    static_bulk_price: float


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def build_demand_signal(conn) -> dict[int, np.ndarray]:
    """Per-product daily demand, stockout days imputed by product median."""
    df = pd.read_sql_query(
        "SELECT product_id, qty_sold, stockout FROM daily_sales_history "
        "ORDER BY product_id, date", conn)
    signal = {}
    for pid, g in df.groupby("product_id"):
        y = g["qty_sold"].astype(float).values.copy()
        so = g["stockout"].values.astype(bool)
        if so.any():
            med = float(np.median(y[~so])) if (~so).any() and (y[~so] > 0).any() \
                else float(np.mean(y))
            y[so] = max(med, 1.0)
        signal[int(pid)] = y
    return signal


def build_context(conn) -> dict[int, ProductCtx]:
    dates = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM daily_sales_history ORDER BY date")]
    n_days = len(dates)
    date_idx = {d: i for i, d in enumerate(dates)}

    cost_map: dict[int, np.ndarray] = {}
    for r in conn.execute("SELECT product_id, date, cost FROM "
                          "wholesale_price_history"):
        arr = cost_map.setdefault(int(r["product_id"]), np.full(n_days, np.nan))
        arr[date_idx[r["date"]]] = float(r["cost"])

    ctx = {}
    for r in conn.execute(
        "SELECT p.product_id, p.name, p.regime, p.mrp, p.wholesale_cost, "
        "       p.base_margin_pct, p.shelf_life_days, "
        "       COALESCE(i.stock_on_hand, 0) AS stock, "
        "       COALESCE(c.tier, 'MEDIUM') AS tier "
        "FROM products p "
        "LEFT JOIN inventory_current i USING (product_id) "
        "LEFT JOIN ml_cluster_assignment c USING (product_id) "
        "WHERE p.is_active = 1 ORDER BY p.product_id"):
        pid = int(r["product_id"])
        series = cost_map.get(pid)
        if series is None or np.isnan(series).all():
            series = np.full(n_days, float(r["wholesale_cost"]))
        else:
            series = pd.Series(series).ffill().bfill().values
            series = np.where(np.isnan(series), float(r["wholesale_cost"]),
                              series)
        margin = float(r["base_margin_pct"] or 0.0)
        static_bulk = float(series[0]) * (1 + margin / 100.0)
        base_price = float(r["mrp"]) if r["regime"] == "MRP" else static_bulk

        shelf = r["shelf_life_days"]
        dte = estimate_days_to_expiry(conn, pid, shelf) if shelf else None
        initial_age = max(int(shelf) - int(dte), 0) if (shelf and dte is not None) else 0

        ctx[pid] = ProductCtx(
            product_id=pid, name=r["name"], regime=r["regime"],
            tier=r["tier"], mrp=float(r["mrp"] or 0.0),
            base_margin_pct=margin, shelf_life=int(shelf) if shelf else None,
            initial_age=initial_age, initial_stock=float(r["stock"]),
            cost_series=series, base_price=base_price,
            static_bulk_price=static_bulk)
    return ctx


def build_system_inputs(conn) -> tuple[dict, dict]:
    """(rop/eoq plans per product, bundle uplift per consequent).

    Intelligence on top of raw EOQ:
    • perishables are capped at min(½ shelf life, 30 days) of demand cover —
      ordering more than you can sell before expiry converts margin into
      write-offs;
    • long-life goods keep the plain EOQ.
    """
    shelf = {r["product_id"]: r["shelf_life_days"] for r in conn.execute(
        "SELECT product_id, shelf_life_days FROM products")}
    cover_cap = float(get("simulation.cover_cap_days", 30))
    plans = {}
    for row in compute_plan(conn).itertuples():
        pid = int(row.product_id)
        eoq = float(row.eoq)
        sl = shelf.get(pid)
        if sl and row.avg_daily > 0:
            cover_days = min(float(sl) * 0.5, cover_cap)
            cap = float(row.avg_daily) * cover_days
            eoq = max(min(eoq, cap), 1.0)
        plans[pid] = {"rop": float(row.rop), "eoq": eoq}
    uplift: dict[int, tuple[float, float]] = {}
    bundle_disc = float(get("basket.bundle_discount_pct", 10)) / 100.0
    b_up = float(get("basket.bundle_uplift_pct", 8)) / 100.0
    c_up = float(get("basket.cross_promote_uplift_pct", 5)) / 100.0
    for r in conn.execute("SELECT consequent_id, action FROM "
                          "bundle_recommendations"):
        cur = uplift.get(int(r["consequent_id"]), (0.0, 0.0))
        if r["action"] == "BUNDLE":
            uplift[int(r["consequent_id"])] = (max(cur[0], b_up), bundle_disc)
        elif r["action"] == "CROSS-PROMOTE":
            uplift[int(r["consequent_id"])] = (max(cur[0], c_up), cur[1])
    return plans, uplift


# ---------------------------------------------------------------------------
# one policy run
# ---------------------------------------------------------------------------

def simulate_policy(conn, policy, signal, ctx) -> pd.DataFrame:
    elasticity = float(get("simulation.demand_elasticity", -1.2))
    lead = int(get("inventory.lead_time_days", 3))
    holding_pct = float(get("inventory.holding_cost_pct", 0.18))
    ordering_cost = float(get("inventory.ordering_cost", 150.0))
    dates = [d for (d,) in conn.execute(
        "SELECT DISTINCT date FROM daily_sales_history ORDER BY date")]
    n_days = len(dates)

    state = {pid: {"stock": c.initial_stock, "age": c.initial_age,
                   "pending": deque()} for pid, c in ctx.items()}
    rows = []
    for day in range(n_days):
        month = int(dates[day].split("-")[1])
        rev = cogs = hold = md = so_loss = 0.0
        for pid, c in ctx.items():
            st = state[pid]
            # arrivals (same lead time for both policies — supplier physics)
            while st["pending"] and st["pending"][0][0] == day:
                _, q = st["pending"].popleft()
                st["stock"] += q
                if c.shelf_life:
                    st["age"] = 0                    # fresh batch
            # expiry write-off
            dte = None
            if c.shelf_life:
                st["age"] += 1
                dte = max(c.shelf_life - st["age"], 0)
                if dte == 0 and st["stock"] > 0:
                    md += st["stock"] * float(c.cost_series[day])
                    st["stock"] = 0.0
                    st["age"] = 0
            cost = float(c.cost_series[day])
            # --- decision under test -------------------------------------
            if policy.name == "system":
                price, is_md = policy.price(
                    c.regime, c.base_price, c.mrp, dte, cost,
                    c.base_margin_pct, c.static_bulk_price,
                    tier=c.tier, month=month,
                    stock=st["stock"],
                    avg_daily=float(np.mean(signal[pid][-28:])))
                up_frac, up_disc = policy.demand_uplift(pid)
            else:
                price, is_md = policy.price(
                    c.regime, c.base_price, c.mrp, dte, cost,
                    c.base_margin_pct, c.static_bulk_price)
                up_frac = up_disc = 0.0
            # --- demand response -----------------------------------------
            base_demand = float(signal[pid][day])
            if base_demand > 0:
                factor = (price / c.base_price) ** elasticity if price > 0 else 1.0
                demand = base_demand * factor * (1.0 + up_frac)
            else:
                demand = 0.0
            sales = min(st["stock"], demand)
            stockout_units = demand - sales
            # --- P&L ------------------------------------------------------
            rev += sales * price
            cogs += sales * cost
            # markdown loss = selling below TODAY'S normal price for the
            # regime (MRP, or bulk cost + base margin). Market-driven bulk
            # price movement with cost is NOT a markdown.
            normal_price = (c.mrp if c.regime == "MRP"
                            else cost * (1 + c.base_margin_pct / 100.0))
            if price < normal_price:
                # counterfactual: discount is a "loss" only on units that
                # would have sold at the normal price anyway
                md += min(base_demand, sales) * (normal_price - price)
            if up_disc and up_frac:
                up_units = min(base_demand * up_frac, sales)
                md += up_units * price * up_disc
            so_loss += stockout_units * max(c.base_price - cost, 0.0)
            st["stock"] -= sales
            # --- reorder decision (post-sales stock) ----------------------
            qty = (policy.reorder(pid, st["stock"]) if policy.name == "system"
                   else policy.reorder(st["stock"]))
            if qty > 0:
                st["pending"].append((day + lead, float(qty)))
                cogs += ordering_cost
            hold += st["stock"] * cost * holding_pct / 365.0

        rows.append({
            "day": day, "revenue": round(rev, 2), "cogs": round(cogs, 2),
            "holding_cost": round(hold, 2), "markdown_loss": round(md, 2),
            "stockout_loss": round(so_loss, 2),
            "net_profit": round(rev - cogs - hold - md - so_loss, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# full run + persistence
# ---------------------------------------------------------------------------

def run_simulation(conn, save: bool = True) -> dict:
    """Run both policies on the identical signal; persist + return report."""
    signal = build_demand_signal(conn)
    ctx = build_context(conn)
    plans, uplift = build_system_inputs(conn)

    naive = NaivePolicy()
    system = SystemPolicy(plans, uplift)

    df_naive = simulate_policy(conn, naive, signal, ctx)
    df_system = simulate_policy(conn, system, signal, ctx)

    def totals(df):
        return {k: float(df[k].sum()) for k in
                ("revenue", "cogs", "holding_cost", "markdown_loss",
                 "stockout_loss", "net_profit")}

    tn, ts = totals(df_naive), totals(df_system)
    delta = ts["net_profit"] - tn["net_profit"]
    delta_pct = (delta / abs(tn["net_profit"]) * 100.0
                 if tn["net_profit"] else float("nan"))

    if save:
        SIM_DIR.mkdir(parents=True, exist_ok=True)
        df_naive.to_csv(SIM_DIR / "naive_daily.csv", index=False)
        df_system.to_csv(SIM_DIR / "system_daily.csv", index=False)
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("DELETE FROM simulation_results "
                     "WHERE policy IN ('naive', 'system')")
        for policy_name, df in (("naive", df_naive), ("system", df_system)):
            conn.executemany(
                "INSERT INTO simulation_results (policy, day, revenue, cogs, "
                "holding_cost, markdown_loss, stockout_loss, net_profit, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(policy_name, int(r.day), r.revenue, r.cogs,
                  r.holding_cost, r.markdown_loss, r.stockout_loss,
                  r.net_profit, now) for r in df.itertuples()])
        conn.commit()

    return {"naive": df_naive, "system": df_system,
            "naive_totals": tn, "system_totals": ts,
            "delta_inr": delta, "delta_pct": delta_pct,
            "n_days": len(df_naive)}
