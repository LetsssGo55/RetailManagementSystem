"""
ArthaMart — Dashboard data layer (Phase 10)
=============================================
Cached loaders shared by all five tabs. Every function opens its own
SQLite connection (Streamlit reruns must never share live connections).
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pandas as pd
import streamlit as st

from ..config import DB_PATH, ensure_data_dir


@st.cache_resource
def _db_ok() -> bool:
    return DB_PATH.exists()


@contextmanager
def conn():
    """Yield a SQLite connection and GUARANTEE it is closed.

    (Previously a bare `sqlite3.Connection` was returned and callers used
    `with conn() as c:` — the connection's own context manager only commits
    the transaction, it does NOT close the socket/file handle, leaking one
    handle per dashboard query.)
    """
    ensure_data_dir()
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    try:
        yield c
    finally:
        c.close()


def db_exists() -> bool:
    return _db_ok()


# ---------------------------------------------------------------------------
# tab 1 — decision cards
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_cards() -> list:
    from ..decisions import engine as dec_engine
    with conn() as c:
        return dec_engine.generate_all(c)


# ---------------------------------------------------------------------------
# tab 2 — orders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def load_orders(limit: int = 100) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(
            "SELECT order_id, customer_id, order_date, source, status, "
            "       fulfilment_pct, total_value "
            "FROM customer_orders ORDER BY order_id DESC LIMIT ?",
            c, params=(limit,))


@st.cache_data(ttl=30)
def load_order_lines(order_id: int) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(
            "SELECT l.line_id, l.status, l.raw_text, p.name AS product, "
            "       l.ordered_qty, l.available_qty, l.unit_price, l.line_total "
            "FROM order_line_items l LEFT JOIN products p USING (product_id) "
            "WHERE l.order_id = ? ORDER BY l.line_id", c,
            params=(order_id,))


def load_queue() -> list:
    from ..orders import availability, queue
    with conn() as c:
        for q in queue.build_queue(c):
            if q.status == "DRAFT":
                try:
                    availability.check_order(c, q.order_id)
                except ValueError:
                    pass
        return queue.build_queue(c)


def load_pending_lines() -> list:
    from ..orders import intake
    with conn() as c:
        return intake.pending_lines(c)


# ---------------------------------------------------------------------------
# tab 4 — model performance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def load_forecast_metrics() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(
            "SELECT f.product_id, p.name, AVG(f.rmse) AS rmse, "
            "       AVG(f.mae) AS mae, AVG(f.mape) AS mape, "
            "       SUM(f.predicted_qty) AS next7 "
            "FROM ml_forecast_output f JOIN products p USING (product_id) "
            "GROUP BY f.product_id ORDER BY f.product_id", c)


@st.cache_data(ttl=120)
def load_forecast_series(product_id: int) -> pd.DataFrame:
    with conn() as c:
        hist = pd.read_sql_query(
            "SELECT date, qty_sold FROM daily_sales_history "
            "WHERE product_id = ? ORDER BY date", c, params=(product_id,))
        fc = pd.read_sql_query(
            "SELECT forecast_date AS date, predicted_qty "
            "FROM ml_forecast_output WHERE product_id = ? "
            "ORDER BY forecast_date", c, params=(product_id,))
    return hist, fc


@st.cache_data(ttl=300)
def load_clusters():
    """Read persisted cluster labels (no model retraining at view time).

    Silhouette (mean of stored per-sample values) and ARI vs the
    generation ground truth are recomputed from stored labels + SQL
    features — both are cheap and fit-free.
    """
    from sklearn.metrics import adjusted_rand_score

    from ..ml import clustering

    with conn() as c:
        df = pd.read_sql_query(
            "SELECT a.product_id, a.tier, a.silhouette, p.name, p.category "
            "FROM ml_cluster_assignment a JOIN products p USING (product_id)",
            c)
        if df.empty:
            return clustering.ClusterReport(), df
        feats = clustering.product_features(c)

    merged = df.merge(feats, on="product_id", how="left")
    report = clustering.ClusterReport()
    report.tier_sizes = df["tier"].value_counts().to_dict()
    report.silhouette_overall = float(df["silhouette"].mean())
    truth = [clustering.ground_truth_tier(cat) for cat in merged["category"]]
    report.ari = float(adjusted_rand_score(truth, merged["tier"]))

    # keep only what the page consumes (PCA input + hover labels)
    out = merged[["product_id", "name", "tier", *clustering.FEATURES]]
    return report, out


@st.cache_data(ttl=300)
def load_basket_rules() -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query(
            "SELECT b.antecedent_id, pa.name AS antecedent, "
            "       b.consequent_id, pc.name AS consequent, b.support, "
            "       b.confidence, b.lift, b.action, b.weekly_impact "
            "FROM bundle_recommendations b "
            "JOIN products pa ON pa.product_id = b.antecedent_id "
            "JOIN products pc ON pc.product_id = b.consequent_id "
            "ORDER BY b.weekly_impact DESC", c)


@st.cache_data(ttl=120)
def load_segments() -> pd.DataFrame:
    from ..personalisation import engine
    with conn() as c:
        return engine.compute_segments(c)


# ---------------------------------------------------------------------------
# tab 5 — simulation
# ---------------------------------------------------------------------------

@st.cache_data(ttl=120)
def load_simulation() -> pd.DataFrame | None:
    with conn() as c:
        df = pd.read_sql_query(
            "SELECT policy, day, revenue, cogs, holding_cost, "
            "       markdown_loss, stockout_loss, net_profit "
            "FROM simulation_results ORDER BY policy, day", c)
    return df if not df.empty else None


@st.cache_data(ttl=300)
def load_inventory_plan() -> pd.DataFrame:
    from ..inventory.reorder import compute_plan
    with conn() as c:
        return compute_plan(c)
