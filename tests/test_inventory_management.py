"""Phase 13 — manual stock management + audit ledger tests."""
import os
import sqlite3
import tempfile

import pytest

from src.data import database
from src.inventory import management


@pytest.fixture
def conn():
    # Use a temporary file database so we don't wipe the production DB
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        database.build_database(db_path=path)
        database.reset_database(db_path=path)          # clean slate for every test
        database.load_config_into_db(db_path=path)
        c = database.get_connection(db_path=path)
        # minimal seed: two products + stock + one customer + one confirmed order
        c.executescript("""
            INSERT INTO products (sku,name,category,regime,mrp,wholesale_cost,is_active)
            VALUES ('SKU-1','Basmati Rice','Grocery','MRP',110.0,70.0,1),
                   ('SKU-2','Parle-G Biscuit','Snacks','MRP',10.0,6.0,1);
            INSERT INTO inventory_current (product_id,stock_on_hand,reorder_level)
            VALUES (1,50,5),(2,100,10);
            INSERT INTO customers (customer_id,name) VALUES (1,'Walk-in');
            INSERT INTO customer_orders (order_id,customer_id,order_date,source,status,total_value)
            VALUES (101,1,'2025-01-01','manual','CONFIRMED',0);
            INSERT INTO order_line_items (order_id,product_id,raw_text,ordered_qty,
                 available_qty,status,unit_price,line_total)
            VALUES (101,1,'Basmati Rice',5,5,'READY',110.0,550.0),
                   (101,2,'Parle-G Biscuit',10,10,'READY',10.0,100.0);
            """)
        c.commit()
        yield c
    finally:
        c.close()
        os.unlink(path)


# ---------------------------------------------------------------------------
# mutations
# ---------------------------------------------------------------------------

def test_add_stock_increases_and_ledgers(conn):
    r = management.add_stock(conn, 1, 20, "manager@store.com",
                             "Manager", "weekly delivery")
    assert r["change_type"] == "ADD"
    assert r["delta"] == 20
    assert r["qty_before"] == 50
    assert r["qty_after"] == 70
    assert r["user_email"] == "manager@store.com"
    assert r["user_name"] == "Manager"
    # stock table reflects it
    stock = conn.execute("SELECT stock_on_hand FROM inventory_current "
                         "WHERE product_id=1").fetchone()["stock_on_hand"]
    assert stock == 70
    # one ledger row
    rows = management.get_ledger(conn)
    assert len(rows) == 1
    assert rows[0]["change_type"] == "ADD"
    assert rows[0]["reference_type"] == "manual"


def test_remove_stock_floors_at_zero(conn):
    r = management.remove_stock(conn, 2, 30, "cashier@store.com",
                                "Cashier", "shrinkage")
    assert r["delta"] == 30
    assert r["qty_after"] == 70
    # removing more than on hand → partial, capped at available
    r2 = management.remove_stock(conn, 2, 999, "cashier@store.com", None)
    assert r2["partial"] is True
    assert r2["delta"] == 70
    assert r2["qty_after"] == 0
    stock = conn.execute("SELECT stock_on_hand FROM inventory_current "
                         "WHERE product_id=2").fetchone()["stock_on_hand"]
    assert stock == 0


def test_remove_stock_rejects_nonpositive(conn):
    with pytest.raises(management.StockError):
        management.remove_stock(conn, 1, 0, "u@x.com")
    with pytest.raises(management.StockError):
        management.remove_stock(conn, 1, -5, "u@x.com")


def test_set_stock_records_signed_delta(conn):
    r = management.set_stock(conn, 1, 5, "manager@store.com", None, "stocktake")
    assert r["change_type"] == "SET"
    assert r["delta"] == -45  # 5 - 50
    assert r["qty_after"] == 5
    rows = management.get_ledger(conn)
    assert rows[0]["change_type"] == "SET"


def test_unknown_product_raises(conn):
    with pytest.raises(management.StockError):
        management.add_stock(conn, 9999, 5, "u@x.com")


# ---------------------------------------------------------------------------
# order deduction & restore
# ---------------------------------------------------------------------------

def test_deduct_for_order_writes_order_deduct_ledger(conn):
    d = management.deduct_for_order(conn, 101, "cashier@store.com", "Cashier")
    assert d["lines_deducted"] == 2
    pids = {x["product_id"] for x in d["deductions"]}
    assert pids == {1, 2}
    # ledger rows are ORDER_DEDUCT tagged to the order
    rows = management.get_ledger(conn)
    assert len(rows) == 2
    assert all(r["change_type"] == "ORDER_DEDUCT" for r in rows)
    assert all(r["reference_type"] == "order" for r in rows)
    assert all(r["reference_id"] == 101 for r in rows)
    # stock deducted
    s1 = conn.execute("SELECT stock_on_hand FROM inventory_current WHERE product_id=1").fetchone()[0]
    s2 = conn.execute("SELECT stock_on_hand FROM inventory_current WHERE product_id=2").fetchone()[0]
    assert s1 == 45 and s2 == 90


def test_restore_for_order_gives_stock_back(conn):
    management.deduct_for_order(conn, 101, "cashier@store.com", "Cashier")
    res = management.restore_for_order(conn, 101, "cashier@store.com", "Cashier")
    assert res["lines_restored"] == 2
    s1 = conn.execute("SELECT stock_on_hand FROM inventory_current WHERE product_id=1").fetchone()[0]
    assert s1 == 50  # back to original
    rows = management.get_ledger(conn)
    types = [r["change_type"] for r in rows]
    assert "ORDER_RESTORE" in types


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------

def test_get_ledger_filters_by_product(conn):
    management.add_stock(conn, 1, 10, "u@x.com")
    management.add_stock(conn, 2, 5, "u@x.com")
    p1 = management.get_ledger(conn, product_id=1)
    assert len(p1) == 1
    assert p1[0]["product_id"] == 1
    all_ = management.get_ledger(conn)
    assert len(all_) == 2


def test_ledger_summary(conn):
    management.add_stock(conn, 1, 10, "a@x.com")
    management.remove_stock(conn, 2, 3, "b@x.com")
    management.deduct_for_order(conn, 101, "a@x.com", None)
    s = management.ledger_summary(conn)
    assert s["total_added"] == 10
    assert s["total_removed"] == 18  # remove 3 + order deduct (5 + 10)
    assert s["orders_deducted"] == 2  # one ORDER_DEDUCT ledger row per line
    assert s["distinct_users"] == 2


def test_stock_snapshot_enriched(conn):
    snap = management.get_stock_snapshot(conn)
    assert len(snap) == 2
    names = {r["name"] for r in snap}
    assert names == {"Basmati Rice", "Parle-G Biscuit"}
    assert snap[0]["stock_on_hand"] in (50, 100)
