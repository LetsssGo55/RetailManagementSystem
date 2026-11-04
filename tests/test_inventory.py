"""Phase 5 — EOQ, ROP/safety stock, reorder planning tests."""
import math

import pytest

from src import config
from src.data import database
from src.inventory import eoq, reorder, rop


# ---------------------------------------------------------------------------
# EOQ
# ---------------------------------------------------------------------------

def test_eoq_known_value():
    # classic textbook case: D=1000/yr, S=100, H=2 → √100000
    assert eoq.eoq(1000, ordering_cost=100, holding_per_unit=2) \
        == pytest.approx(math.sqrt(100_000))


def test_eoq_zero_demand():
    assert eoq.eoq(0, ordering_cost=100, holding_per_unit=2) == 0.0
    assert eoq.eoq(-5) == 0.0


def test_annual_demand_from_forecast():
    assert eoq.annual_demand_from_forecast(100) == 5200.0
    assert eoq.annual_demand_from_forecast(-3) == 0.0


def test_holding_cost():
    assert eoq.holding_cost_per_unit(500, holding_pct=0.18) == pytest.approx(90)
    # config default is 0.18 (18%)
    assert eoq.holding_cost_per_unit(100) == pytest.approx(18)


def test_recommended_order_qty_rounds_up():
    q = eoq.recommended_order_qty(1000, unit_cost=100)   # H = 18 ₹/unit/yr
    assert q == math.ceil(math.sqrt(2 * 1000 * 150 / 18))
    assert eoq.recommended_order_qty(0, unit_cost=100) == 0


# ---------------------------------------------------------------------------
# ROP / safety stock
# ---------------------------------------------------------------------------

def test_safety_stock_known_value():
    ss = rop.safety_stock(10, lead_time=3, z=1.65)
    assert ss == pytest.approx(1.65 * 10 * math.sqrt(3))


def test_reorder_point_known_value():
    r = rop.reorder_point(20, sigma_demand=10, lead_time=3, z=1.65)
    assert r == pytest.approx(20 * 3 + 1.65 * 10 * math.sqrt(3))


def test_stock_status_transitions():
    ss = rop.safety_stock(10, lead_time=3, z=1.65)      # ≈ 28.6
    r = rop.reorder_point(20, 10, lead_time=3, z=1.65)  # ≈ 88.6
    assert rop.stock_status(10, r, ss) == "CRITICAL"    # ≤ safety stock
    assert rop.stock_status(ss, r, ss) == "CRITICAL"    # boundary
    assert rop.stock_status(50, r, ss) == "REORDER"     # between ss and rop
    assert rop.stock_status(r, r, ss) == "REORDER"      # boundary
    assert rop.stock_status(r + 1, r, ss) == "OK"


# ---------------------------------------------------------------------------
# reorder planning — synthetic temp DB
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    db_path = tmp_path / "inv.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    c.executemany(
        "INSERT INTO products (product_id, sku, name, category, regime, mrp, "
        "wholesale_cost, base_margin_pct, unit) "
        "VALUES (?, ?, ?, 'TEST', 'BULK', NULL, ?, 20, 'pcs')",
        [(1, "A", "Hot Seller", 50.0), (2, "B", "Calm Seller", 50.0)])
    c.execute("INSERT INTO inventory_current (product_id, stock_on_hand, "
              "reorder_level) VALUES (1, 2, 5)")
    c.execute("INSERT INTO inventory_current (product_id, stock_on_hand, "
              "reorder_level) VALUES (2, 500, 5)")
    # 30 days history: product 1 sells ~10/day with 5 stockout days;
    # product 2 sells 1/day, no stockouts
    for d in range(30):
        date = f"2026-07-{d + 1:02d}"
        qty1 = 8 + (d % 5)                              # 8..12
        so1 = 1 if d < 5 else 0                         # 5 stockout days
        c.execute("INSERT INTO daily_sales_history "
                  "(date, product_id, qty_sold, revenue, stockout) "
                  "VALUES (?, 1, ?, ?, ?)",
                  (date, 0 if so1 else qty1, 0.0, so1))
        c.execute("INSERT INTO daily_sales_history "
                  "(date, product_id, qty_sold, revenue, stockout) "
                  "VALUES (?, 2, 1, 50.0, 0)", (date,))
    c.commit()
    yield c
    c.close()


def test_plan_priority_ranks_stockout_product_first(conn):
    plan = reorder.compute_plan(conn)
    assert len(plan) == 2
    top = plan.iloc[0]
    assert top["product_id"] == 1
    # priority = misses × 2 + forecast7
    expected = 5 * float(config.get("inventory.reorder_priority_weight", 2)) \
        + top["forecast7"]
    assert top["priority"] == pytest.approx(expected)


def test_plan_flags_and_eoq(conn):
    plan = reorder.compute_plan(conn)
    p1 = plan[plan["product_id"] == 1].iloc[0]
    p2 = plan[plan["product_id"] == 2].iloc[0]
    assert p1["status"] in ("REORDER", "CRITICAL")     # stock 2 ≤ ROP
    assert p2["status"] == "OK"                        # stock 500 ≫ ROP
    assert p1["eoq"] > 0 and p2["eoq"] > 0
    # EOQ cross-check with the direct formula (fallback demand path)
    S = float(config.get("inventory.ordering_cost", 150.0))
    H = p1["wholesale_cost"] * float(config.get("inventory.holding_cost_pct", 0.18))
    ann = p1["forecast7"] * 52
    assert p1["eoq"] == pytest.approx(math.sqrt(2 * ann * S / H))


def test_reorder_cycle_creates_pos_and_alerts(conn):
    s = reorder.run_reorder_cycle(conn, save=True)
    assert s["orders_recommended"] == 1                  # only product 1 due
    assert s["products_planned"] == 2

    pos = conn.execute("SELECT * FROM purchase_orders").fetchall()
    assert len(pos) == 1
    assert pos[0]["product_id"] == 1
    assert pos[0]["qty"] == max(int(round(s["due"].iloc[0]["eoq"])), 1)
    assert pos[0]["status"] == "RECOMMENDED"

    alerts = conn.execute("SELECT * FROM procurement_alerts").fetchall()
    assert len(alerts) == 1
    assert alerts[0]["product_id"] == 1


def test_reorder_cycle_idempotent(conn):
    reorder.run_reorder_cycle(conn, save=True)
    reorder.run_reorder_cycle(conn, save=True)
    assert conn.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM procurement_alerts").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# integration on the real DB (forecast rows exist after `train`)
# ---------------------------------------------------------------------------

def test_real_plan_integration():
    if not config.DB_PATH.exists():
        pytest.skip("run setup first")
    c = database.get_connection()
    try:
        plan = reorder.compute_plan(c)
        assert len(plan) >= 60
        assert (plan["eoq"] >= 0).all()
        assert set(plan["status"]) <= {"OK", "REORDER", "CRITICAL"}
        # priority sorted descending
        assert plan["priority"].is_monotonic_decreasing

        s = reorder.run_reorder_cycle(c, save=True)
        assert s["orders_recommended"] >= 0
        n_po = c.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
        assert n_po == s["orders_recommended"]
    finally:
        c.close()
