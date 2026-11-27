"""
ArthaMart — Manual Stock Management & Audit Ledger (Phase 13)
===============================================================

Manual inventory control plus an *immutable* audit trail. Every stock
movement — whether added by hand, removed by hand, set to an exact level,
or deducted because a customer order was fulfilled — appends one row to
`stock_ledger` recording:

    who       → user_email / user_name
    what      → product_id + signed delta
    before    → qty_before
    after     → qty_after
    when      → created_at
    why       → reason
    context   → reference_type ('manual'|'order'|'scan') + reference_id

Public operations
-----------------
    add_stock(...)        increase stock_on_hand
    remove_stock(...)     decrease stock_on_hand (never below zero)
    set_stock(...)        set stock_on_hand to an exact level
    deduct_for_order(...) deduct fulfilled lines of an order
    restore_for_order(...) give stock back for a cancelled order
    get_ledger(...)       read the audit trail
    get_stock_snapshot()  current stock per product

All functions are synchronous and run on the caller's connection, matching
the rest of the ArthaMart data layer (`src/data/database.py`).
"""
from __future__ import annotations

import datetime as _dt
import sqlite3

CHANGE_TYPES = ("ADD", "REMOVE", "SET", "ORDER_DEDUCT", "ORDER_RESTORE")
REFERENCE_TYPES = ("manual", "order", "scan")


class StockError(RuntimeError):
    """Raised for invalid stock operations (unknown product, insufficient
    stock, bad change type, …)."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _resolve_user(
    conn: sqlite3.Connection, user_email: str, user_name: str | None
) -> tuple[str, str]:
    """Return (email, display_name). Falls back to the app_users record when
    only an email is supplied, and to the email itself as a last resort."""
    email = (user_email or "").strip().lower()
    name = (user_name or "").strip() or email
    try:
        row = conn.execute(
            "SELECT name FROM app_users WHERE lower(email) = ? LIMIT 1",
            (email,),
        ).fetchone()
        if row and row["name"]:
            name = row["name"]
    except sqlite3.OperationalError:
        # app_users table may not exist yet in very old databases.
        pass
    return email, name


def _validate_change_type(change_type: str) -> str:
    ct = (change_type or "").upper()
    if ct not in CHANGE_TYPES:
        raise StockError(
            f"Invalid change_type {change_type!r}; "
            f"expected one of {CHANGE_TYPES}"
        )
    return ct


def _ledger_row(
    conn: sqlite3.Connection,
    product_id: int,
    user_email: str,
    user_name: str,
    change_type: str,
    delta: int,
    qty_before: int,
    qty_after: int,
    reason: str | None,
    reference_type: str,
    reference_id: int | None,
) -> int:
    """Append one immutable row to stock_ledger. Returns the new ledger_id."""
    cur = conn.execute(
        """
        INSERT INTO stock_ledger
        (product_id, user_email, user_name, change_type, delta, qty_before,
         qty_after, reason, reference_type, reference_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(product_id),
            user_email,
            user_name,
            change_type,
            int(delta),
            int(qty_before),
            int(qty_after),
            reason,
            reference_type,
            reference_id,
            _now(),
        ),
    )
    return int(cur.lastrowid)


def _stock_row(conn: sqlite3.Connection, product_id: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT stock_on_hand FROM inventory_current WHERE product_id = ?",
        (product_id,),
    ).fetchone()
    if row is None:
        raise StockError(f"Unknown product_id={product_id}")
    return row


# ---------------------------------------------------------------------------
# public mutations
# ---------------------------------------------------------------------------
def add_stock(
    conn: sqlite3.Connection,
    product_id: int,
    delta: int,
    user_email: str,
    user_name: str | None = None,
    reason: str | None = None,
    reference_type: str = "manual",
    reference_id: int | None = None,
) -> dict:
    """Increase `stock_on_hand` by `delta` and record the movement."""
    delta = int(delta)
    if delta <= 0:
        raise StockError("add_stock requires a positive delta")
    reference_type = reference_type or "manual"
    if reference_type not in REFERENCE_TYPES:
        raise StockError(
            f"Invalid reference_type {reference_type!r}; "
            f"expected one of {REFERENCE_TYPES}"
        )

    email, name = _resolve_user(conn, user_email, user_name)
    row = _stock_row(conn, product_id)
    before = int(row["stock_on_hand"])
    after = before + delta

    conn.execute(
        "UPDATE inventory_current SET stock_on_hand = ?, last_updated = ? "
        "WHERE product_id = ?",
        (after, _now(), int(product_id)),
    )
    _ledger_row(
        conn, product_id, email, name, "ADD", delta, before, after, reason,
        reference_type, reference_id,
    )
    conn.commit()
    return {
        "product_id": int(product_id),
        "change_type": "ADD",
        "delta": delta,
        "qty_before": before,
        "qty_after": after,
        "user_email": email,
        "user_name": name,
        "reason": reason,
        "created_at": _now(),
    }


def remove_stock(
    conn: sqlite3.Connection,
    product_id: int,
    delta: int,
    user_email: str,
    user_name: str | None = None,
    reason: str | None = None,
    reference_type: str = "manual",
    reference_id: int | None = None,
    change_type: str = "REMOVE",
) -> dict:
    """Decrease `stock_on_hand` by `delta`, never below zero.

    Returns the *actual* amount removed (capped at what is on hand) so
    callers can tell whether the removal was partial.
    `change_type` labels the ledger entry (default REMOVE; ORDER_DEDUCT when
    the removal is driven by a fulfilled customer order).
    """
    delta = int(delta)
    if delta <= 0:
        raise StockError("remove_stock requires a positive delta")
    change_type = _validate_change_type(change_type)
    reference_type = reference_type or "manual"
    if reference_type not in REFERENCE_TYPES:
        raise StockError(
            f"Invalid reference_type {reference_type!r}; "
            f"expected one of {REFERENCE_TYPES}"
        )

    email, name = _resolve_user(conn, user_email, user_name)
    row = _stock_row(conn, product_id)
    before = int(row["stock_on_hand"])
    actual = min(delta, before)
    after = before - actual

    conn.execute(
        "UPDATE inventory_current SET stock_on_hand = ?, last_updated = ? "
        "WHERE product_id = ?",
        (after, _now(), int(product_id)),
    )
    _ledger_row(
        conn, product_id, email, name, change_type, actual, before, after,
        reason, reference_type, reference_id,
    )
    conn.commit()
    return {
        "product_id": int(product_id),
        "change_type": change_type,
        "delta": actual,
        "requested_delta": delta,
        "partial": actual < delta,
        "qty_before": before,
        "qty_after": after,
        "user_email": email,
        "user_name": name,
        "reason": reason,
        "created_at": _now(),
    }


def set_stock(
    conn: sqlite3.Connection,
    product_id: int,
    qty: int,
    user_email: str,
    user_name: str | None = None,
    reason: str | None = None,
    reference_type: str = "manual",
    reference_id: int | None = None,
) -> dict:
    """Set `stock_on_hand` to an exact level (positive delta = ADD, negative
    delta = REMOVE) and record the movement as a SET entry."""
    qty = int(qty)
    if qty < 0:
        raise StockError("set_stock requires a non-negative quantity")
    reference_type = reference_type or "manual"
    if reference_type not in REFERENCE_TYPES:
        raise StockError(
            f"Invalid reference_type {reference_type!r}; "
            f"expected one of {REFERENCE_TYPES}"
        )

    email, name = _resolve_user(conn, user_email, user_name)
    row = _stock_row(conn, product_id)
    before = int(row["stock_on_hand"])
    delta = qty - before  # signed

    conn.execute(
        "UPDATE inventory_current SET stock_on_hand = ?, last_updated = ? "
        "WHERE product_id = ?",
        (qty, _now(), int(product_id)),
    )
    _ledger_row(
        conn, product_id, email, name, "SET", delta, before, qty, reason,
        reference_type, reference_id,
    )
    conn.commit()
    return {
        "product_id": int(product_id),
        "change_type": "SET",
        "delta": delta,
        "qty_before": before,
        "qty_after": qty,
        "user_email": email,
        "user_name": name,
        "reason": reason,
        "created_at": _now(),
    }


def stock_on_hand(conn: sqlite3.Connection, product_id: int) -> int:
    """Current on-hand qty for a product (0 if not tracked)."""
    row = conn.execute(
        "SELECT stock_on_hand FROM inventory_current WHERE product_id = ?",
        (int(product_id),),
    ).fetchone()
    return int(row["stock_on_hand"]) if row else 0


def get_product(conn: sqlite3.Connection, product_id: int) -> dict | None:
    """Fetch one product row as a dict, or None."""
    row = conn.execute(
        "SELECT * FROM products WHERE product_id = ?", (int(product_id),)
    ).fetchone()
    return dict(row) if row else None


def deduct_for_order(
    conn: sqlite3.Connection,
    order_id: int,
    user_email: str,
    user_name: str | None = None,
) -> dict:
    """Deduct the available qty of every billable line of an order from
    `inventory_current`, recording one ORDER_DEDUCT ledger row per product.

    Billable lines = lines that carry a product_id and have available_qty > 0
    (status READY / PARTIAL / FULFILLED). Lines already deducted are skipped
    safely because available_qty drives the amount.
    """
    email, name = _resolve_user(conn, user_email, user_name)

    order = conn.execute(
        "SELECT status FROM customer_orders WHERE order_id = ?", (order_id,)
    ).fetchone()
    if order is None:
        raise StockError(f"Unknown order_id={order_id}")

    lines = conn.execute(
        """
        SELECT line_id, product_id, available_qty, status
        FROM order_line_items
        WHERE order_id = ? AND product_id IS NOT NULL
          AND (available_qty IS NULL OR available_qty > 0)
        """,
        (order_id,),
    ).fetchall()

    deducted: list[dict] = []
    for ln in lines:
        avail = int(ln["available_qty"] or 0)
        if avail <= 0:
            continue
        res = remove_stock(
            conn, int(ln["product_id"]), avail, email, name,
            reason=f"Fulfilled order #{order_id}, line {ln['line_id']}",
            reference_type="order", reference_id=order_id,
            change_type="ORDER_DEDUCT",
        )
        conn.execute(
            "UPDATE order_line_items SET status = 'FULFILLED' "
            "WHERE line_id = ?", (ln["line_id"],))
        deducted.append(res)

    conn.commit()
    return {
        "order_id": int(order_id),
        "order_status": order["status"],
        "lines_deducted": len(deducted),
        "deductions": deducted,
    }


def restore_for_order(
    conn: sqlite3.Connection,
    order_id: int,
    user_email: str,
    user_name: str | None = None,
) -> dict:
    """Give stock back for a cancelled/unfulfilled order (ORDER_RESTORE)."""
    email, name = _resolve_user(conn, user_email, user_name)

    order = conn.execute(
        "SELECT status FROM customer_orders WHERE order_id = ?", (order_id,)
    ).fetchone()
    if order is None:
        raise StockError(f"Unknown order_id={order_id}")

    lines = conn.execute(
        """
        SELECT line_id, product_id, available_qty, status
        FROM order_line_items
        WHERE order_id = ? AND product_id IS NOT NULL
          AND status IN ('READY', 'PARTIAL', 'FULFILLED')
        """,
        (order_id,),
    ).fetchall()

    restored: list[dict] = []
    for ln in lines:
        avail = int(ln["available_qty"] or 0)
        if avail <= 0:
            continue
        row = _stock_row(conn, int(ln["product_id"]))
        before = int(row["stock_on_hand"])
        after = before + avail
        conn.execute(
            "UPDATE inventory_current SET stock_on_hand = ?, last_updated = ? "
            "WHERE product_id = ?",
            (after, _now(), int(ln["product_id"])),
        )
        _ledger_row(
            conn, int(ln["product_id"]), email, name, "ORDER_RESTORE", avail,
            before, after,
            f"Restored stock for cancelled order #{order_id}, "
            f"line {ln['line_id']}",
            "order", order_id,
        )
        conn.execute(
            "UPDATE order_line_items SET status = 'RESTORED' WHERE line_id = ?",
            (ln["line_id"],))
        restored.append({
            "line_id": int(ln["line_id"]),
            "product_id": int(ln["product_id"]),
            "delta": avail,
            "qty_before": before,
            "qty_after": after,
        })

    conn.commit()
    return {
        "order_id": int(order_id),
        "lines_restored": len(restored),
        "restorations": restored,
    }


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
def get_ledger(
    conn: sqlite3.Connection,
    product_id: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Read stock_ledger, newest first. Optionally filter to one product."""
    if product_id is not None:
        rows = conn.execute(
            """
            SELECT * FROM stock_ledger
            WHERE product_id = ?
            ORDER BY ledger_id DESC LIMIT ?
            """, (int(product_id), int(limit))).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM stock_ledger
            ORDER BY ledger_id DESC LIMIT ?
            """, (int(limit),)).fetchall()
    return [dict(r) for r in rows]


def get_ledger_for_product(
    conn: sqlite3.Connection, product_id: int, limit: int = 50
) -> list[dict]:
    """Convenience wrapper around get_ledger for a single product."""
    return get_ledger(conn, product_id=product_id, limit=limit)


def get_stock_snapshot(conn: sqlite3.Connection) -> list[dict]:
    """Current stock per product, enriched with category/name and the most
    recent ledger movement (if any)."""
    rows = conn.execute(
        """
        SELECT p.product_id, p.name, p.category, p.sku,
               i.stock_on_hand, i.reorder_level, i.last_updated,
               l.change_type AS last_change, l.delta AS last_delta,
               l.user_email AS last_user, l.created_at AS last_change_at
        FROM inventory_current i
        JOIN products p USING (product_id)
        LEFT JOIN stock_ledger l ON l.ledger_id = (
            SELECT ledger_id FROM stock_ledger s2
            WHERE s2.product_id = p.product_id
            ORDER BY s2.ledger_id DESC LIMIT 1)
        WHERE p.is_active = 1
        ORDER BY p.name
        """).fetchall()
    return [dict(r) for r in rows]


def ledger_summary(conn: sqlite3.Connection) -> dict:
    """High-level totals for the audit dashboard."""
    added = conn.execute(
        "SELECT COALESCE(SUM(delta),0) FROM stock_ledger "
        "WHERE change_type='ADD'").fetchone()[0]
    removed = conn.execute(
        "SELECT COALESCE(SUM(delta),0) FROM stock_ledger "
        "WHERE change_type IN ('REMOVE','ORDER_DEDUCT')").fetchone()[0]
    orders_deducted = conn.execute(
        "SELECT COUNT(*) FROM stock_ledger WHERE change_type='ORDER_DEDUCT'"
    ).fetchone()[0]
    users = conn.execute(
        "SELECT COUNT(DISTINCT user_email) FROM stock_ledger").fetchone()[0]
    return {
        "total_added": int(added),
        "total_removed": int(removed),
        "orders_deducted": int(orders_deducted),
        "distinct_users": int(users),
        "net_movement": int(added) - int(removed),
    }
