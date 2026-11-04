"""
ArthaMart — Pricing Dispatcher (Phase 6, Step 6.3)
=====================================================
Routes each product to its regime pricer:

    MRP  → expiry markdown ladder (mrp_pricing)
    BULK → market-linked cost-plus (bulk_pricing)

and persists every decision:
    price_history_log    one row per product per day (idempotent)
    markdown_alerts_log  one row per MRP markdown event

Days-to-expiry is not yet tracked per batch; until batch intake lands
(Phase 7 procurement), `estimate_days_to_expiry()` derives a deterministic
pseudo-age so the whole ladder is exercisable end-to-end.
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from dataclasses import dataclass

from ..config import get
from ..ml.features import FESTIVAL_MONTHS
from .bulk_pricing import demand_adjustment, price_bulk, seasonal_adjustment
from .mrp_pricing import price_mrp


@dataclass
class PriceResult:
    product_id: int
    name: str
    regime: str
    base_price: float          # MRP or latest wholesale cost
    price: float
    discount_pct: float        # MRP only
    margin_pct: float          # BULK only
    floor_applied: bool
    reason: str


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def estimate_days_to_expiry(conn: sqlite3.Connection, product_id: int,
                            shelf_life: int | None = None) -> int | None:
    """Deterministic pseudo days-to-expiry for MRP products.
    Returns None for products without a shelf life (bulk goods)."""
    if shelf_life is None:
        row = conn.execute(
            "SELECT shelf_life_days FROM products WHERE product_id = ?",
            (product_id,)).fetchone()
        shelf_life = row["shelf_life_days"] if row else None
    if not shelf_life:
        return None
    age = (product_id * 37) % int(shelf_life)       # deterministic pseudo-age
    return max(int(shelf_life) - age, 0)


def latest_wholesale_cost(conn: sqlite3.Connection,
                          product_id: int) -> float | None:
    row = conn.execute(
        "SELECT cost FROM wholesale_price_history WHERE product_id = ? "
        "ORDER BY date DESC LIMIT 1", (product_id,)).fetchone()
    return float(row["cost"]) if row else None


def _forecast7(conn: sqlite3.Connection, product_id: int) -> float | None:
    row = conn.execute(
        "SELECT SUM(predicted_qty) FROM ml_forecast_output WHERE product_id = ?",
        (product_id,)).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _baseline7(conn: sqlite3.Connection, product_id: int) -> float | None:
    row = conn.execute(
        "SELECT AVG(qty_sold) * 7 FROM daily_sales_history "
        "WHERE product_id = ? AND date >= "
        "(SELECT date(MAX(date), '-27 day') FROM daily_sales_history)",
        (product_id,)).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _tier(conn: sqlite3.Connection, product_id: int) -> str | None:
    row = conn.execute(
        "SELECT tier FROM ml_cluster_assignment WHERE product_id = ?",
        (product_id,)).fetchone()
    return row["tier"] if row else None


def _prev_price(conn: sqlite3.Connection, product_id: int,
                as_of: str) -> float | None:
    row = conn.execute(
        "SELECT price FROM price_history_log "
        "WHERE product_id = ? AND price_date < ? "
        "ORDER BY price_date DESC LIMIT 1", (product_id, as_of)).fetchone()
    return float(row["price"]) if row else None


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def price_product(conn: sqlite3.Connection, product_id: int,
                  as_of: str | None = None,
                  days_to_expiry: int | None = None) -> PriceResult:
    """Price one product according to its regime."""
    row = conn.execute(
        "SELECT product_id, name, regime, mrp, wholesale_cost, "
        "       base_margin_pct, shelf_life_days "
        "FROM products WHERE product_id = ?", (product_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown product_id={product_id}")

    as_of = as_of or _dt.date.today().isoformat()

    if row["regime"] == "MRP":
        dte = (days_to_expiry if days_to_expiry is not None
               else estimate_days_to_expiry(conn, product_id,
                                            row["shelf_life_days"]))
        res = price_mrp(float(row["mrp"]), float(row["wholesale_cost"]),
                        dte if dte is not None else 999,
                        tier=_tier(conn, product_id))
        return PriceResult(product_id, row["name"], "MRP",
                           float(row["mrp"]), res.price, res.discount_pct,
                           0.0, res.floor_applied, res.reason)

    # ---- BULK --------------------------------------------------------
    cost = latest_wholesale_cost(conn, product_id)
    if cost is None:
        cost = float(row["wholesale_cost"])
    base_margin = float(row["base_margin_pct"]
                        or get("pricing.bulk_base_margin_pct", 18))
    month = int(as_of.split("-")[1])
    res = price_bulk(
        cost, base_margin,
        demand_adj=demand_adjustment(_forecast7(conn, product_id),
                                     _baseline7(conn, product_id)),
        seasonal_adj=seasonal_adjustment(month in FESTIVAL_MONTHS),
        prev_price=_prev_price(conn, product_id, as_of),
    )
    return PriceResult(product_id, row["name"], "BULK", cost, res.price,
                       0.0, res.margin_pct, res.floor_applied, res.reason)


def price_all(conn: sqlite3.Connection, as_of: str | None = None,
              save: bool = True) -> list[PriceResult]:
    """Price every active product and persist the logs (idempotent)."""
    as_of = as_of or _dt.date.today().isoformat()
    pids = [r[0] for r in conn.execute(
        "SELECT product_id FROM products WHERE is_active = 1 "
        "ORDER BY product_id")]
    results = [price_product(conn, pid, as_of) for pid in pids]

    if save:
        now = _dt.datetime.now().strftime("%H:%M:%S")
        conn.execute("DELETE FROM price_history_log WHERE price_date = ?",
                     (as_of,))
        conn.execute("DELETE FROM markdown_alerts_log "
                     "WHERE created_at LIKE ?", (as_of + "%",))
        conn.executemany(
            "INSERT INTO price_history_log "
            "(product_id, price_date, regime, price, reason) "
            "VALUES (?, ?, ?, ?, ?)",
            [(r.product_id, as_of, r.regime, r.price, r.reason)
             for r in results])
        conn.executemany(
            "INSERT INTO markdown_alerts_log "
            "(product_id, old_price, new_price, discount_pct, reason, "
            " created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [(r.product_id, r.base_price, r.price, r.discount_pct,
              r.reason, f"{as_of} {now}")
             for r in results
             if r.regime == "MRP" and r.discount_pct > 0])
        conn.commit()
    return results
