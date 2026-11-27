"""
ArthaMart — Multi-buyer consolidated picking wave.

Open orders are scored against the same live stock snapshot, ranked with
fully-fulfillable orders first, then by standalone availability percentage,
then FIFO. Stock is allocated exactly once across that ranked wave; identical
products are consolidated into category/product baskets while preserving the
buyer and order allocation on every pick line.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict


PENDING_STATUSES = ("DRAFT", "PENDING_REVIEW", "CONFIRMED")
UNRESOLVED_STATUSES = {"REVIEW", "UNMATCHED"}


def build_picking_wave(conn: sqlite3.Connection) -> dict:
    """Build a read-only, globally allocated pick wave and cross-sell suggestions."""
    inventory = {
        int(row["product_id"]): {
            "stock": float(row["stock_on_hand"] or 0),
            "name": row["name"],
            "sku": row["sku"],
            "category": row["category"] or "Uncategorised",
            "unit": row["unit"] or "unit",
        }
        for row in conn.execute(
            """SELECT p.product_id, p.name, p.sku, p.category, p.unit,
                      COALESCE(i.stock_on_hand, 0) AS stock_on_hand
               FROM products p LEFT JOIN inventory_current i USING(product_id)
               WHERE p.is_active = 1"""
        )
    }
    placeholders = ",".join("?" for _ in PENDING_STATUSES)
    headers = conn.execute(
        f"""SELECT o.order_id, o.customer_id, o.order_date, o.created_at,
                   o.status, o.total_value,
                   COALESCE(c.name, 'Walk-in') AS buyer_name,
                   c.phone AS buyer_phone
            FROM customer_orders o
            LEFT JOIN customers c USING(customer_id)
            WHERE o.status IN ({placeholders})
            ORDER BY o.order_date, o.order_id""",
        PENDING_STATUSES,
    ).fetchall()

    order_work = []
    for header in headers:
        order_id = int(header["order_id"])
        lines = conn.execute(
            """SELECT l.line_id, l.product_id, l.raw_text, l.ordered_qty,
                      l.status, p.name AS product, p.category, p.unit
               FROM order_line_items l
               LEFT JOIN products p USING(product_id)
               WHERE l.order_id = ?
               ORDER BY l.line_id""",
            (order_id,),
        ).fetchall()

        demand_by_product: dict[int, float] = defaultdict(float)
        unresolved_qty = 0.0
        unresolved_lines = 0
        requested = 0.0
        for line in lines:
            qty = max(float(line["ordered_qty"] or 0), 0.0)
            requested += qty
            pid = line["product_id"]
            if pid is None or line["status"] in UNRESOLVED_STATUSES or int(pid) not in inventory:
                unresolved_qty += qty
                unresolved_lines += 1
            else:
                demand_by_product[int(pid)] += qty

        standalone_available = sum(
            min(inventory[pid]["stock"], qty)
            for pid, qty in demand_by_product.items()
        )
        standalone_pct = (100.0 * standalone_available / requested) if requested else 0.0
        fully_ready = (
            unresolved_lines == 0
            and requested > 0
            and all(inventory[pid]["stock"] >= qty
                    for pid, qty in demand_by_product.items())
        )
        order_work.append({
            "order_id": order_id,
            "customer_id": header["customer_id"],
            "buyer_name": header["buyer_name"],
            "buyer_phone": header["buyer_phone"],
            "order_date": header["order_date"],
            "created_at": header["created_at"],
            "status": header["status"],
            "total_value": float(header["total_value"] or 0),
            "lines": lines,
            "demand_by_product": dict(demand_by_product),
            "requested_units": requested,
            "unresolved_qty": unresolved_qty,
            "unresolved_lines": unresolved_lines,
            "standalone_fulfilment_pct": standalone_pct,
            "fully_ready": fully_ready,
        })

    # Full orders are picked first; then best live availability; FIFO breaks ties.
    order_work.sort(key=lambda order: (
        0 if order["fully_ready"] else 1,
        -order["standalone_fulfilment_pct"],
        order["order_date"] or "",
        order["order_id"],
    ))

    remaining = {pid: info["stock"] for pid, info in inventory.items()}
    product_wave: dict[int, dict] = {}
    order_summaries = []
    suggestions = []

    for rank, order in enumerate(order_work, start=1):
        allocated_units = 0.0
        short_units = float(order["unresolved_qty"])
        allocation_details = []
        ordered_ids = sorted(order["demand_by_product"])

        for pid, requested_qty in sorted(
            order["demand_by_product"].items(),
            key=lambda pair: (
                inventory[pair[0]]["category"].casefold(),
                inventory[pair[0]]["name"].casefold(),
                pair[0],
            ),
        ):
            info = inventory[pid]
            before_wave = remaining.get(pid, 0.0)
            picked = min(requested_qty, before_wave)
            remaining[pid] = max(before_wave - picked, 0.0)
            short = max(requested_qty - picked, 0.0)
            allocated_units += picked
            short_units += short

            if pid not in product_wave:
                product_wave[pid] = {
                    "product_id": pid,
                    "product": info["name"],
                    "sku": info["sku"],
                    "category": info["category"],
                    "unit": info["unit"],
                    "stock_at_wave_start": info["stock"],
                    "requested_qty": 0.0,
                    "pick_qty": 0.0,
                    "short_qty": 0.0,
                    "remaining_after_wave": info["stock"],
                    "buyers": [],
                }
            item = product_wave[pid]
            item["requested_qty"] += requested_qty
            item["pick_qty"] += picked
            item["short_qty"] += short
            item["remaining_after_wave"] = remaining[pid]
            item["buyers"].append({
                "order_id": order["order_id"],
                "customer_id": order["customer_id"],
                "buyer": order["buyer_name"],
                "requested_qty": requested_qty,
                "pick_qty": picked,
                "short_qty": short,
                "priority_rank": rank,
            })
            allocation_details.append({
                "product_id": pid,
                "product": info["name"],
                "requested_qty": requested_qty,
                "pick_qty": picked,
                "short_qty": short,
                "unit": info["unit"],
            })

        pct = (100.0 * allocated_units / order["requested_units"]
               if order["requested_units"] else 0.0)
        dispatch = (
            "READY" if short_units <= 0 and order["unresolved_lines"] == 0
            else "PARTIAL" if allocated_units > 0
            else "HOLD"
        )
        order_summaries.append({
            "priority_rank": rank,
            "order_id": order["order_id"],
            "customer_id": order["customer_id"],
            "buyer": order["buyer_name"],
            "buyer_phone": order["buyer_phone"],
            "order_date": order["order_date"],
            "status": order["status"],
            "standalone_fulfilment_pct": round(order["standalone_fulfilment_pct"], 2),
            "planned_fulfilment_pct": round(pct, 2),
            "requested_units": round(order["requested_units"], 3),
            "allocated_units": round(allocated_units, 3),
            "short_units": round(short_units, 3),
            "unresolved_lines": order["unresolved_lines"],
            "dispatch": dispatch,
            "allocations": allocation_details,
        })

        if ordered_ids:
            marks = ",".join("?" for _ in ordered_ids)
            rec_rows = conn.execute(
                f"""SELECT r.antecedent_id, a.name AS based_on,
                           r.consequent_id, p.name AS suggested_product,
                           r.support, r.confidence, r.lift, r.action,
                           COALESCE(i.stock_on_hand, 0) AS current_stock
                    FROM bundle_recommendations r
                    JOIN products a ON a.product_id = r.antecedent_id
                    JOIN products p ON p.product_id = r.consequent_id
                    LEFT JOIN inventory_current i ON i.product_id = p.product_id
                    WHERE r.antecedent_id IN ({marks})
                      AND r.consequent_id NOT IN ({marks})
                      AND p.is_active = 1
                    ORDER BY r.confidence DESC, r.lift DESC, r.weekly_impact DESC""",
                (*ordered_ids, *ordered_ids),
            ).fetchall()
            added = set()
            for rec in rec_rows:
                rid = int(rec["consequent_id"])
                after_wave = remaining.get(rid, float(rec["current_stock"] or 0))
                if rid in added or after_wave <= 0:
                    continue
                added.add(rid)
                suggestions.append({
                    "order_id": order["order_id"],
                    "customer_id": order["customer_id"],
                    "buyer": order["buyer_name"],
                    "based_on": rec["based_on"],
                    "suggested_product_id": rid,
                    "suggested_product": rec["suggested_product"],
                    "support": float(rec["support"] or 0),
                    "confidence": float(rec["confidence"] or 0),
                    "lift": float(rec["lift"] or 0),
                    "action": rec["action"],
                    "stock_after_wave": after_wave,
                })
                if len(added) >= 5:
                    break

    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in product_wave.values():
        item["requested_qty"] = round(item["requested_qty"], 3)
        item["pick_qty"] = round(item["pick_qty"], 3)
        item["short_qty"] = round(item["short_qty"], 3)
        item["remaining_after_wave"] = round(item["remaining_after_wave"], 3)
        grouped[item["category"]].append(item)

    category_groups = [
        {
            "category": category,
            "products": sorted(items, key=lambda item: item["product"].casefold()),
        }
        for category, items in sorted(grouped.items(), key=lambda pair: pair[0].casefold())
    ]
    return {
        "priority_rule": "Fully ready orders first, then live standalone fulfilment percentage, then FIFO. Stock is allocated once across the entire wave.",
        "open_orders": len(order_summaries),
        "priority_orders": order_summaries,
        "category_groups": category_groups,
        "suggestions": suggestions,
        "stock_allocation_is_planned": True,
    }
