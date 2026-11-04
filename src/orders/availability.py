"""
ArthaMart — Stock Availability (Phase 3, Step 3.1)
====================================================
For every resolvable line item:

    READY          stock ≥ ordered qty
    PARTIAL        0 < stock < ordered qty
    NOT_AVAILABLE  stock = 0

Fulfilment score per order = Σ(available qty) / Σ(ordered qty) × 100,
computed over all lines that carry a product_id.
"""
from __future__ import annotations

import sqlite3

from .models import LineStatus, OrderStatus


def stock_on_hand(conn: sqlite3.Connection, product_id: int) -> int:
    row = conn.execute(
        "SELECT stock_on_hand FROM inventory_current WHERE product_id = ?",
        (product_id,)).fetchone()
    return int(row["stock_on_hand"]) if row else 0


def classify(ordered_qty: float, stock: int) -> str:
    """Roadmap classification for a single line."""
    if stock <= 0:
        return LineStatus.NOT_AVAILABLE.value
    if stock >= ordered_qty:
        return LineStatus.READY.value
    return LineStatus.PARTIAL.value


def check_order(conn: sqlite3.Connection, order_id: int) -> dict:
    """
    Classify every resolvable line of an order against current stock.

    Writes `available_qty` + READY/PARTIAL/NOT_AVAILABLE back to each line
    and updates `customer_orders.fulfilment_pct`.

    Returns a summary dict:
        {order_id, fulfilment_pct, lines: n, ready, partial, unavailable,
         unresolved, updated: bool}
    """
    order = conn.execute(
        "SELECT status FROM customer_orders WHERE order_id = ?",
        (order_id,)).fetchone()
    if order is None:
        raise ValueError(f"Unknown order_id={order_id}")
    if order["status"] == OrderStatus.FULFILLED.value:
        raise ValueError(f"Order {order_id} already fulfilled.")

    lines = conn.execute(
        "SELECT line_id, product_id, ordered_qty, status "
        "FROM order_line_items WHERE order_id = ? ORDER BY line_id",
        (order_id,)).fetchall()

    counts = {"READY": 0, "PARTIAL": 0, "NOT_AVAILABLE": 0}
    ordered_sum = 0.0
    available_sum = 0.0
    unresolved = 0

    for ln in lines:
        status = ln["status"]
        if ln["product_id"] is None or status in ("REVIEW", "UNMATCHED"):
            unresolved += 1
            continue
        ordered = float(ln["ordered_qty"])
        stock = stock_on_hand(conn, ln["product_id"])
        cls = classify(ordered, stock)
        available = min(float(stock), ordered)
        conn.execute(
            "UPDATE order_line_items SET available_qty = ?, status = ? "
            "WHERE line_id = ?", (available, cls, ln["line_id"]))
        counts[cls] += 1
        ordered_sum += ordered
        available_sum += available

    score = 100.0 * available_sum / ordered_sum if ordered_sum else 0.0
    conn.execute(
        "UPDATE customer_orders SET fulfilment_pct = ? WHERE order_id = ?",
        (round(score, 2), order_id))
    conn.commit()

    return {
        "order_id": order_id,
        "fulfilment_pct": score,
        "lines": len(lines),
        "ready": counts["READY"],
        "partial": counts["PARTIAL"],
        "unavailable": counts["NOT_AVAILABLE"],
        "unresolved": unresolved,
        "updated": True,
    }
