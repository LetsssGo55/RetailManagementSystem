"""
ArthaMart — Order Intake (Phase 3)
===================================
Turns an OCR SlipResult (Phase 2) into a persisted draft order:

    customer_orders  (1 row)  +  order_line_items (n rows)

Line items carry the matcher status:
    MATCHED   → auto-accepted (product_id set)
    REVIEW    → product_id set, but flagged for a human confirm
    UNMATCHED → product_id NULL, manual entry needed

Also exposes the review-queue operations:
    pending_lines()    – lines awaiting a human decision
    resolve_line()     – attach/override the product (+ optional qty)
    drop_line()        – remove an unwanted line
"""
from __future__ import annotations

import datetime as _dt
import sqlite3

from ..ocr.matcher import (MATCHED, REVIEW, CatalogProduct, load_catalog,
                           top_candidates)
from ..ocr.pipeline import SlipResult
from .models import LineStatus, OrderStatus


class OrderError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# provisional pricing (Phase 6 replaces this with the dual-regime engine)
# ---------------------------------------------------------------------------

def unit_price(conn: sqlite3.Connection, product_id: int) -> float:
    """MRP regime → MRP.  BULK regime → cost × (1 + base margin)."""
    row = conn.execute(
        "SELECT regime, mrp, wholesale_cost, base_margin_pct "
        "FROM products WHERE product_id = ?", (product_id,)).fetchone()
    if row is None:
        raise OrderError(f"Unknown product_id={product_id}")
    if row["regime"] == "MRP":
        return float(row["mrp"] or 0.0)
    return round(float(row["wholesale_cost"]) *
                 (1 + float(row["base_margin_pct"]) / 100.0), 2)


# ---------------------------------------------------------------------------
# order creation
# ---------------------------------------------------------------------------

def create_order_from_slip(conn: sqlite3.Connection, slip: SlipResult,
                           customer_id: int | None = None,
                           source: str = "OCR",
                           order_date: str | None = None) -> int:
    """Persist a draft order from an OCR result. Returns order_id."""
    if not slip.matches:
        raise OrderError("Slip produced no line items — nothing to order.")

    if customer_id is None:
        row = conn.execute(
            "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 1"
        ).fetchone()
        if row is None:
            raise OrderError("No customers in database — run setup first.")
        customer_id = int(row["customer_id"])

    date = order_date or _dt.date.today().isoformat()
    cur = conn.execute(
        "INSERT INTO customer_orders "
        "(customer_id, order_date, source, fulfilment_pct, status, total_value) "
        "VALUES (?, ?, ?, 0.0, ?, 0.0)",
        (customer_id, date, source, OrderStatus.DRAFT.value),
    )
    order_id = int(cur.lastrowid)

    total = 0.0
    for m in slip.matches:
        price = unit_price(conn, m.product_id) if m.product_id else 0.0
        line_total = round(price * float(m.quantity), 2)
        total += line_total if m.status == MATCHED else 0.0
        conn.execute(
            "INSERT INTO order_line_items "
            "(order_id, product_id, raw_text, ordered_qty, available_qty, "
            " status, unit_price, line_total) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?, ?)",
            (order_id, m.product_id, m.query_raw, float(m.quantity),
             m.status, price, line_total),
        )

    conn.execute("UPDATE customer_orders SET total_value = ? WHERE order_id = ?",
                 (round(total, 2), order_id))
    conn.commit()
    return order_id


# ---------------------------------------------------------------------------
# review queue
# ---------------------------------------------------------------------------

def pending_lines(conn: sqlite3.Connection,
                  order_id: int | None = None) -> list[dict]:
    """Lines needing a human decision, each with top-3 catalog candidates."""
    sql = ("SELECT l.line_id, l.order_id, l.raw_text, l.ordered_qty, l.status, "
           "       l.product_id "
           "FROM order_line_items l JOIN customer_orders o USING (order_id) "
           "WHERE l.status IN ('REVIEW', 'UNMATCHED') "
           "  AND o.status NOT IN ('FULFILLED', 'CANCELLED')")
    params: tuple = ()
    if order_id is not None:
        sql += " AND l.order_id = ?"
        params = (order_id,)
    sql += " ORDER BY l.order_id, l.line_id"

    catalog = load_catalog(conn)
    out = []
    for r in conn.execute(sql, params).fetchall():
        # candidate text: use stored product name for REVIEW lines
        name = r["raw_text"] or ""
        if r["status"] == LineStatus.REVIEW.value and r["product_id"]:
            cand = [c for c in catalog if c.product_id == r["product_id"]]
            if cand:
                name = cand[0].name
        cands = top_candidates(name, catalog, k=3)
        out.append({
            "line_id": r["line_id"],
            "order_id": r["order_id"],
            "raw_text": r["raw_text"],
            "ordered_qty": r["ordered_qty"],
            "status": r["status"],
            "suggested_product_id": r["product_id"],
            "candidates": [(c.product_id, c.name, round(s, 1))
                           for c, s in cands],
        })
    return out


def resolve_line(conn: sqlite3.Connection, line_id: int, product_id: int,
                 ordered_qty: float | None = None) -> None:
    """Attach a product to a REVIEW/UNMATCHED line (human resolution)."""
    row = conn.execute("SELECT status FROM order_line_items WHERE line_id = ?",
                       (line_id,)).fetchone()
    if row is None:
        raise OrderError(f"Unknown line_id={line_id}")
    if row["status"] not in (LineStatus.REVIEW.value,
                             LineStatus.UNMATCHED.value):
        raise OrderError(f"Line {line_id} is {row['status']} — only "
                         "REVIEW/UNMATCHED lines can be resolved.")

    prod = conn.execute("SELECT product_id FROM products WHERE product_id = ?",
                        (product_id,)).fetchone()
    if prod is None:
        raise OrderError(f"Unknown product_id={product_id}")

    price = unit_price(conn, product_id)
    if ordered_qty is None:
        conn.execute(
            "UPDATE order_line_items SET product_id = ?, status = ?, "
            "unit_price = ?, line_total = ROUND(? * ordered_qty, 2) "
            "WHERE line_id = ?",
            (product_id, LineStatus.CONFIRMED.value, price, price, line_id),
        )
    else:
        conn.execute(
            "UPDATE order_line_items SET ordered_qty = ?, product_id = ?, "
            "status = ?, unit_price = ?, line_total = ROUND(? * ?, 2) "
            "WHERE line_id = ?",
            (float(ordered_qty), product_id, LineStatus.CONFIRMED.value,
             price, price, float(ordered_qty), line_id),
        )
    conn.commit()


def drop_line(conn: sqlite3.Connection, line_id: int) -> None:
    """Remove an unwanted line from a draft order."""
    conn.execute("DELETE FROM order_line_items WHERE line_id = ?", (line_id,))
    conn.commit()
