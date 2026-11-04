"""
ArthaMart — Prioritised Order Queue (Phase 3, Step 3.2)
=========================================================
Ranks pending orders for dispatch:

    1. fulfilment score  (desc)  — ship the most-complete orders first
    2. order value       (desc)  — break ties toward bigger revenue
    3. FIFO              (asc)   — oldest order wins remaining ties

Dispatch flags:
    DISPATCH         score = 100  → ship everything now
    PARTIAL          0 < score < 100 → ship what's available
    HOLD             score = 0 (or unresolved lines only)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .models import OrderStatus


@dataclass
class QueueEntry:
    rank: int
    order_id: int
    customer_id: int
    order_date: str
    fulfilment_pct: float
    total_value: float
    status: str
    lines_total: int
    dispatch_flag: str


_PENDING = ("'" + "','".join(
    [OrderStatus.DRAFT.value, OrderStatus.PENDING_REVIEW.value,
     OrderStatus.CONFIRMED.value]) + "'")


def dispatch_flag(score: float) -> str:
    if score >= 100.0:
        return "DISPATCH"
    if score > 0.0:
        return "PARTIAL"
    return "HOLD"


def build_queue(conn: sqlite3.Connection) -> list[QueueEntry]:
    """Return the prioritised fulfilment queue for all pending orders."""
    rows = conn.execute(
        f"""
        SELECT o.order_id, o.customer_id, o.order_date, o.fulfilment_pct,
               o.total_value, o.status, COUNT(l.line_id) AS lines_total
        FROM customer_orders o
        LEFT JOIN order_line_items l USING (order_id)
        WHERE o.status IN ({_PENDING})
        GROUP BY o.order_id
        ORDER BY o.fulfilment_pct DESC,
                 o.total_value DESC,
                 o.order_date ASC,
                 o.order_id ASC
        """
    ).fetchall()

    return [
        QueueEntry(
            rank=i,
            order_id=int(r["order_id"]),
            customer_id=int(r["customer_id"]),
            order_date=r["order_date"],
            fulfilment_pct=float(r["fulfilment_pct"] or 0.0),
            total_value=float(r["total_value"] or 0.0),
            status=r["status"],
            lines_total=int(r["lines_total"]),
            dispatch_flag=dispatch_flag(float(r["fulfilment_pct"] or 0.0)),
        )
        for i, r in enumerate(rows, start=1)
    ]
