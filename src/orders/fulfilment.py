"""
ArthaMart — Order Confirmation & Fulfilment (Phase 3)
=======================================================
    confirm_order()   block-until-resolved gate + availability scoring
    fulfil_order()    deduct stock, mark lines & order final
    get_order()       full order view (header + lines) for CLI / dashboard
"""
from __future__ import annotations

import datetime as _dt
import sqlite3

from .availability import check_order
from .intake import OrderError
from .models import LineStatus, OrderStatus
from ..inventory import management as inventory_management


def confirm_order(conn: sqlite3.Connection, order_id: int) -> dict:
    """
    Gate: all lines must be resolved (no REVIEW/UNMATCHED left).
    Runs the availability classification and moves the order to CONFIRMED.
    """
    order = conn.execute(
        "SELECT status FROM customer_orders WHERE order_id = ?",
        (order_id,)).fetchone()
    if order is None:
        raise OrderError(f"Unknown order_id={order_id}")
    if order["status"] in (OrderStatus.FULFILLED.value,
                           OrderStatus.CANCELLED.value):
        raise OrderError(f"Order {order_id} is {order['status']} — "
                         "cannot confirm.")

    bad = conn.execute(
        "SELECT COUNT(*) FROM order_line_items "
        "WHERE order_id = ? AND status IN ('REVIEW', 'UNMATCHED')",
        (order_id,)).fetchone()[0]
    if bad:
        raise OrderError(
            f"Order {order_id} has {bad} unresolved line(s) — "
            "resolve them in the review queue first.")

    summary = check_order(conn, order_id)
    conn.execute("UPDATE customer_orders SET status = ? WHERE order_id = ?",
                 (OrderStatus.CONFIRMED.value, order_id))
    conn.commit()
    summary["status"] = OrderStatus.CONFIRMED.value
    return summary


def fulfil_order(conn: sqlite3.Connection, order_id: int, actor: dict | None = None) -> dict:
    """
    Deduct stock for the available qty of each line and close the order.
    Fully-fulfilled → FULFILLED, otherwise PARTIALLY_FULFILLED is expressed
    via fulfilment_pct < 100 with status FULFILLED (schema has one status
    column; the percentage carries the nuance).
    """
    order = conn.execute(
        "SELECT status, fulfilment_pct FROM customer_orders WHERE order_id = ?",
        (order_id,)).fetchone()
    if order is None:
        raise OrderError(f"Unknown order_id={order_id}")
    if order["status"] == OrderStatus.FULFILLED.value:
        raise OrderError(f"Order {order_id} already fulfilled.")
    if order["status"] == OrderStatus.CANCELLED.value:
        raise OrderError(f"Order {order_id} is cancelled.")
    if order["status"] not in (OrderStatus.CONFIRMED.value,
                               OrderStatus.DRAFT.value):
        raise OrderError(f"Order {order_id} in state {order['status']} — "
                         "confirm it first.")

    # availability must have run (CONFIRMED guarantees it; DRAFT → run now)
    if order["status"] == OrderStatus.DRAFT.value:
        bad = conn.execute(
            "SELECT COUNT(*) FROM order_line_items "
            "WHERE order_id = ? AND status IN ('REVIEW', 'UNMATCHED')",
            (order_id,)).fetchone()[0]
        if bad:
            raise OrderError(f"Order {order_id} has {bad} unresolved line(s).")
        check_order(conn, order_id)

    inventory_management.ensure_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    current = conn.execute(
        "SELECT status FROM customer_orders WHERE order_id = ?", (order_id,)
    ).fetchone()
    if current is None or current["status"] == OrderStatus.FULFILLED.value:
        conn.rollback()
        raise OrderError(f"Order {order_id} has already been fulfilled.")
    if current["status"] == OrderStatus.CANCELLED.value:
        conn.rollback()
        raise OrderError(f"Order {order_id} is cancelled.")

    lines = conn.execute(
        "SELECT line_id, product_id, ordered_qty, available_qty "
        "FROM order_line_items WHERE order_id = ? AND product_id IS NOT NULL",
        (order_id,)).fetchall()

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shipped, ordered_total, shipped_total, low_stock = 0, 0, 0, []
    for ln in lines:
        ordered_total += 1
        avail = float(ln["available_qty"] or 0.0)
        if avail <= 0:
            continue
        before_row = conn.execute(
            "SELECT stock_on_hand FROM inventory_current WHERE product_id = ?",
            (ln["product_id"],),
        ).fetchone()
        stock_before = float(before_row["stock_on_hand"]) if before_row else 0.0
        # guarded deduction — never below zero even under concurrency
        cur = conn.execute(
            "UPDATE inventory_current "
            "SET stock_on_hand = stock_on_hand - ?, last_updated = ? "
            "WHERE product_id = ? AND stock_on_hand >= ?",
            (avail, now, ln["product_id"], avail))
        if cur.rowcount == 0:
            # stock moved under us — ship whatever remains (partial)
            left = conn.execute(
                "SELECT stock_on_hand FROM inventory_current "
                "WHERE product_id = ?", (ln["product_id"],)).fetchone()
            left = float(left["stock_on_hand"]) if left else 0.0
            if left <= 0:
                conn.execute(
                    "UPDATE order_line_items SET available_qty = 0, status = ? "
                    "WHERE line_id = ?",
                    (LineStatus.NOT_AVAILABLE.value, ln["line_id"]))
                continue
            conn.execute(
                "UPDATE inventory_current SET stock_on_hand = 0, "
                "last_updated = ? WHERE product_id = ?",
                (now, ln["product_id"]))
            conn.execute(
                "UPDATE order_line_items SET available_qty = ?, status = ? "
                "WHERE line_id = ?",
                (left, "PARTIAL", ln["line_id"]))
            avail = left
        shipped += 1
        if conn.execute("SELECT status FROM order_line_items WHERE line_id = ?",
                        (ln["line_id"],)).fetchone()[0] != "PARTIAL":
            conn.execute(
                "UPDATE order_line_items SET status = ? WHERE line_id = ?",
                ("FULFILLED", ln["line_id"]))
        new_stock = conn.execute(
            "SELECT stock_on_hand, reorder_level FROM inventory_current "
            "WHERE product_id = ?", (ln["product_id"],)).fetchone()
        actual_after = float(new_stock["stock_on_hand"]) if new_stock else 0.0
        actual_before = stock_before if cur.rowcount else max(stock_before, avail)
        inventory_management.record_stock_movement(
            conn, int(ln["product_id"]), -avail, actual_before, actual_after,
            movement_type="ORDER_FULFILMENT", actor=actor,
            reference_type="ORDER", reference_id=int(order_id),
            note=f"Customer order #{order_id}, line #{ln['line_id']}",
        )
        if new_stock and new_stock["stock_on_hand"] <= new_stock["reorder_level"]:
            low_stock.append(ln["product_id"])

    pct = float(order["fulfilment_pct"] or 0.0)
    conn.execute("UPDATE customer_orders SET status = ?, fulfilment_pct = ? "
                 "WHERE order_id = ?",
                 (OrderStatus.FULFILLED.value, pct, order_id))
    conn.commit()

    return {
        "order_id": order_id,
        "status": OrderStatus.FULFILLED.value,
        "lines_shipped": shipped,
        "lines_orderable": ordered_total,
        "fulfilment_pct": pct,
        "low_stock_products": low_stock,
    }


def cancel_order(conn: sqlite3.Connection, order_id: int) -> None:
    conn.execute(
        "UPDATE customer_orders SET status = ? WHERE order_id = ? "
        "AND status NOT IN ('FULFILLED')",
        (OrderStatus.CANCELLED.value, order_id))
    conn.commit()


# ---------------------------------------------------------------------------
# read helpers (CLI + dashboard)
# ---------------------------------------------------------------------------

def get_order(conn: sqlite3.Connection, order_id: int) -> dict:
    header = conn.execute(
        "SELECT * FROM customer_orders WHERE order_id = ?",
        (order_id,)).fetchone()
    if header is None:
        raise OrderError(f"Unknown order_id={order_id}")
    lines = conn.execute(
        "SELECT l.*, p.name AS product_name, p.unit "
        "FROM order_line_items l LEFT JOIN products p USING (product_id) "
        "WHERE l.order_id = ? ORDER BY l.line_id",
        (order_id,)).fetchall()
    return {"header": dict(header), "lines": [dict(x) for x in lines]}


def list_orders(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        "SELECT order_id, customer_id, order_date, source, status, "
        "       fulfilment_pct, total_value "
        "FROM customer_orders ORDER BY order_id DESC LIMIT ?",
        (limit,)).fetchall()
    return [dict(r) for r in rows]
