"""Tests for operational inventory adjustments and stock audit history."""
import pytest

from src.data import database
from src.inventory import management


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "inventory-management.db"
    database.build_database(path)
    c = database.get_connection(path)
    c.execute(
        """INSERT INTO products
           (product_id, sku, name, category, regime, mrp, wholesale_cost,
            base_margin_pct, unit)
           VALUES (1, 'SKU-001', 'Test Rice', 'Grocery', 'BULK', NULL, 40, 18, 'kg')"""
    )
    c.execute(
        "INSERT INTO inventory_current (product_id, stock_on_hand, reorder_level) VALUES (1, 12, 3)"
    )
    c.commit()
    yield c
    c.close()


def test_manual_add_and_remove_are_logged_with_actor(conn):
    actor = {"name": "Store Manager", "email": "manager@example.com"}
    added = management.adjust_stock(
        conn, 1, 5, action="add", actor=actor, note="Received supplier stock"
    )
    assert added["stock_before"] == 12
    assert added["stock_after"] == 17

    removed = management.adjust_stock(
        conn, 1, 4, action="remove", actor=actor, note="Damaged goods"
    )
    assert removed["stock_before"] == 17
    assert removed["stock_after"] == 13

    rows = management.movement_history(conn)
    assert len(rows) == 2
    assert rows[0]["movement_type"] == "MANUAL_REMOVE"
    assert rows[0]["change_qty"] == -4
    assert rows[0]["actor_email"] == "manager@example.com"
    assert rows[1]["movement_type"] == "MANUAL_ADD"
    assert rows[1]["change_qty"] == 5


def test_remove_cannot_make_stock_negative(conn):
    with pytest.raises(ValueError, match="Insufficient stock"):
        management.adjust_stock(conn, 1, 13, action="remove")
    assert conn.execute(
        "SELECT stock_on_hand FROM inventory_current WHERE product_id = 1"
    ).fetchone()[0] == 12
    assert conn.execute("SELECT COUNT(*) FROM inventory_movements").fetchone()[0] == 0


def test_create_product_registers_stock_and_barcode(conn):
    pid = management.create_product(
        conn, sku="SKU-NEW", name="New Tea", category="Beverages",
        regime="MRP", mrp=120, wholesale_cost=80, barcode="8901234567890",
        opening_stock=9, reorder_level=2,
        actor={"name": "Owner", "email": "owner@example.com"},
    )
    found = management.lookup_barcode(conn, "8901234567890")
    assert found and found["product_id"] == pid
    assert found["stock"] == 9
    assert management.lookup_barcode(conn, "SKU-NEW")["product_id"] == pid
    assert management.lookup_barcode(conn, "sku-new")["product_id"] == pid
    movement = conn.execute(
        "SELECT movement_type, change_qty, actor_email FROM inventory_movements WHERE product_id = ?",
        (pid,),
    ).fetchone()
    assert tuple(movement) == ("OPENING_BALANCE", 9, "owner@example.com")


def test_barcode_not_found_returns_none(conn):
    assert management.lookup_barcode(conn, "0000000000000") is None



def test_barcode_order_deducts_stock_once_and_logs_confirming_user(conn):
    from src.inventory.management import ensure_schema, movement_history
    from src.orders import fulfilment, intake
    from src.orders.intake import OrderError

    conn.execute(
        "INSERT INTO customers (name, phone, segment) VALUES ('Barcode Buyer', '9000000000', 'NEW')"
    )
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO product_barcodes (barcode, product_id) VALUES ('8901112223334', 1)"
    )
    conn.commit()

    order_id = intake.create_order_from_barcodes(
        conn, [{"barcode": "8901112223334", "quantity": 2}]
    )
    preview = fulfilment.confirm_order(conn, order_id)
    assert preview["fulfilment_pct"] == 100.0

    actor = {"name": "Counter Operator", "email": "counter@example.com"}
    result = fulfilment.fulfil_order(conn, order_id, actor=actor)
    assert result["status"] == "FULFILLED"
    assert conn.execute(
        "SELECT stock_on_hand FROM inventory_current WHERE product_id = 1"
    ).fetchone()[0] == 10

    movement = movement_history(conn, limit=10)[0]
    assert movement["movement_type"] == "ORDER_FULFILMENT"
    assert movement["change_qty"] == -2
    assert movement["stock_before"] == 12
    assert movement["stock_after"] == 10
    assert movement["reference_type"] == "ORDER"
    assert movement["reference_id"] == order_id
    assert movement["actor_email"] == "counter@example.com"
    assert movement["actor_name"] == "Counter Operator"
    assert movement["customer_name"] == "Barcode Buyer"
    assert movement["customer_id"] == 1

    with pytest.raises(OrderError, match="already fulfilled"):
        fulfilment.fulfil_order(conn, order_id, actor=actor)
    assert conn.execute(
        "SELECT COUNT(*) FROM inventory_movements WHERE movement_type = 'ORDER_FULFILMENT'"
    ).fetchone()[0] == 1
