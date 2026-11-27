"""Tests for buyer-aware consolidated picking waves and cross-sell suggestions."""
import pytest

from src.data import database
from src.orders.picking import build_picking_wave


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "picking-wave.db"
    database.build_database(path)
    c = database.get_connection(path)
    c.executemany(
        """INSERT INTO products
           (product_id, sku, name, category, regime, mrp, wholesale_cost,
            base_margin_pct, unit)
           VALUES (?, ?, ?, ?, 'MRP', ?, ?, 20, ?)""",
        [
            (1, "RICE-1", "Basmati Rice", "Grain", 100, 70, "kg"),
            (2, "TEA-1", "Assam Tea", "Beverages", 80, 55, "pack"),
            (3, "BIS-1", "Digestive Biscuits", "Snacks", 30, 20, "pack"),
        ],
    )
    c.executemany(
        "INSERT INTO inventory_current (product_id, stock_on_hand, reorder_level) VALUES (?, ?, 0)",
        [(1, 5), (2, 1), (3, 3)],
    )
    c.executemany(
        "INSERT INTO customers (customer_id, name, phone, segment) VALUES (?, ?, ?, 'NEW')",
        [(1, "Buyer One", "9000000001"), (2, "Buyer Two", "9000000002")],
    )
    c.executemany(
        """INSERT INTO customer_orders
           (order_id, customer_id, order_date, source, fulfilment_pct, status, total_value)
           VALUES (?, ?, ?, 'RECEIPT-OCR', 0, 'DRAFT', 100)""",
        [(101, 1, "2026-11-28"), (102, 2, "2026-11-28")],
    )
    c.executemany(
        """INSERT INTO order_line_items
           (order_id, product_id, raw_text, ordered_qty, status)
           VALUES (?, ?, ?, ?, 'MATCHED')""",
        [
            (101, 1, "Basmati Rice 2 kg", 2),
            (102, 1, "Basmati Rice 4 kg", 4),
            (102, 2, "Assam Tea 2 packs", 2),
        ],
    )
    c.execute(
        """INSERT INTO bundle_recommendations
           (antecedent_id, consequent_id, support, confidence, lift, action, weekly_impact)
           VALUES (1, 3, 0.25, 0.75, 1.8, 'CROSS-PROMOTE', 120)"""
    )
    c.commit()
    yield c
    c.close()


def test_wave_consolidates_same_product_but_keeps_buyer_allocations(conn):
    wave = build_picking_wave(conn)
    assert wave["open_orders"] == 2
    assert [o["order_id"] for o in wave["priority_orders"]] == [101, 102]

    rice = next(
        p for group in wave["category_groups"] for p in group["products"]
        if p["product_id"] == 1
    )
    assert rice["requested_qty"] == 6
    assert rice["pick_qty"] == 5
    assert rice["short_qty"] == 1
    assert rice["remaining_after_wave"] == 0
    assert rice["buyers"] == [
        {"order_id": 101, "customer_id": 1, "buyer": "Buyer One",
         "requested_qty": 2.0, "pick_qty": 2.0, "short_qty": 0.0,
         "priority_rank": 1},
        {"order_id": 102, "customer_id": 2, "buyer": "Buyer Two",
         "requested_qty": 4.0, "pick_qty": 3.0, "short_qty": 1.0,
         "priority_rank": 2},
    ]


def test_ready_order_is_prioritised_and_recommendations_use_existing_rules(conn):
    wave = build_picking_wave(conn)
    first, second = wave["priority_orders"]
    assert first["dispatch"] == "READY"
    assert first["planned_fulfilment_pct"] == 100
    assert second["dispatch"] == "PARTIAL"
    assert second["planned_fulfilment_pct"] < 100

    rec = next(x for x in wave["suggestions"] if x["order_id"] == 101)
    assert rec["buyer"] == "Buyer One"
    assert rec["based_on"] == "Basmati Rice"
    assert rec["suggested_product"] == "Digestive Biscuits"
    assert rec["confidence"] == pytest.approx(0.75)
    assert rec["stock_after_wave"] == 3


def test_wave_does_not_mutate_inventory_or_order_state(conn):
    wave = build_picking_wave(conn)
    assert wave["open_orders"] == 2
    before = conn.execute(
        "SELECT product_id, stock_on_hand FROM inventory_current ORDER BY product_id"
    ).fetchall()
    states = conn.execute(
        "SELECT order_id, status FROM customer_orders ORDER BY order_id"
    ).fetchall()
    build_picking_wave(conn)
    after = conn.execute(
        "SELECT product_id, stock_on_hand FROM inventory_current ORDER BY product_id"
    ).fetchall()
    states_after = conn.execute(
        "SELECT order_id, status FROM customer_orders ORDER BY order_id"
    ).fetchall()
    assert [tuple(r) for r in before] == [tuple(r) for r in after]
    assert [tuple(r) for r in states] == [tuple(r) for r in states_after]



def test_cross_sell_does_not_recommend_stock_reserved_by_later_buyer(conn):
    conn.execute(
        """INSERT INTO customer_orders
           (order_id, customer_id, order_date, source, fulfilment_pct, status, total_value)
           VALUES (103, 2, '2026-11-29', 'RECEIPT-OCR', 0, 'DRAFT', 100)"""
    )
    conn.execute(
        """INSERT INTO order_line_items
           (order_id, product_id, raw_text, ordered_qty, status)
           VALUES (103, 3, 'Digestive Biscuits 3 packs', 3, 'MATCHED')"""
    )
    conn.commit()

    wave = build_picking_wave(conn)
    # Biscuits are fully allocated to an existing order, so no buyer is offered
    # the same units as an additional cross-sell.
    assert all(
        rec["suggested_product_id"] != 3
        for rec in wave["suggestions"]
    )
