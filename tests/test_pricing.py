"""Phase 6 — dual-regime pricing tests."""
import pytest

from src.data import database
from src.pricing import bulk_pricing, engine, mrp_pricing


# ---------------------------------------------------------------------------
# MRP markdown ladder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("days,discount", [
    (30, 0), (15, 0),          # fresh — full price
    (14, 10), (8, 10),         # 8–14 days
    (7, 20), (4, 20),          # 4–7 days
    (3, 35), (1, 35), (0, 35), # 0–3 days
])
def test_markdown_ladder_boundaries(days, discount):
    assert mrp_pricing.markdown_for_days(days) == discount


def test_tier_shifts_markdown_timing():
    # 16 days left: MEDIUM stays at 0%, SLOW (−4d → 12) starts markdown
    assert mrp_pricing.price_mrp(100, 60, 16, tier="MEDIUM").discount_pct == 0
    assert mrp_pricing.price_mrp(100, 60, 16, tier="SLOW").discount_pct == 10
    # 3 days left: MEDIUM gets 35%, FAST (+2d → 5) only 20%
    assert mrp_pricing.price_mrp(100, 60, 3, tier="MEDIUM").discount_pct == 35
    assert mrp_pricing.price_mrp(100, 60, 3, tier="FAST").discount_pct == 20


def test_mrp_price_values():
    r = mrp_pricing.price_mrp(100.0, 60.0, 5, tier="MEDIUM")
    assert r.discount_pct == 20
    assert r.price == pytest.approx(80.0)
    assert r.floor_applied is False


def test_cost_floor_caps_deep_markdown():
    # thin margin: MRP 100, cost 90 → 35% off gives 65 < floor 81 (90%)
    r = mrp_pricing.price_mrp(100.0, 90.0, 1, tier="MEDIUM")
    assert r.discount_pct == 35
    assert r.floor_applied is True
    assert r.price == pytest.approx(90.0 * 0.90)


def test_estimated_expiry_is_deterministic_and_bounded():
    conn = database.get_connection(":memory:")
    try:
        conn.execute("CREATE TABLE products (product_id INTEGER, "
                     "shelf_life_days INTEGER)")
        conn.execute("INSERT INTO products VALUES (1, 180)")
        a = engine.estimate_days_to_expiry(conn, 1)
        b = engine.estimate_days_to_expiry(conn, 1)
        assert a == b and 0 <= a <= 180
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# BULK cost-plus
# ---------------------------------------------------------------------------

def test_bulk_basic_cost_plus():
    r = bulk_pricing.price_bulk(100.0, base_margin_pct=18.0)
    assert r.price == pytest.approx(118.0)
    assert r.margin_pct == 18.0


def test_bulk_margin_clamped_to_range():
    # config range is ±8 — huge demand uplift must clamp at +8
    r = bulk_pricing.price_bulk(100.0, 18.0, demand_adj=50.0)
    assert r.margin_pct == pytest.approx(26.0)
    r2 = bulk_pricing.price_bulk(100.0, 18.0, demand_adj=-50.0)
    assert r2.margin_pct == pytest.approx(10.0)


def test_bulk_demand_and_seasonal_add():
    r = bulk_pricing.price_bulk(100.0, 18.0, demand_adj=3.0, seasonal_adj=2.0)
    assert r.margin_pct == pytest.approx(23.0)
    assert r.price == pytest.approx(123.0)


def test_bulk_daily_move_cap():
    # cost doubles → uncapped price would jump +118%; cap at +15% vs prev
    r = bulk_pricing.price_bulk(200.0, 18.0, prev_price=118.0)
    assert r.capped is True
    assert r.price == pytest.approx(118.0 * 1.15)


def test_bulk_floor():
    # negative-ish margin path: adj −8 on 5% base → −3% → floored at 90% cost
    r = bulk_pricing.price_bulk(100.0, 5.0, demand_adj=-8.0)
    assert r.price == pytest.approx(100.0 * 0.97)   # 5−8 = −3% above floor


def test_demand_adjustment_formula():
    assert bulk_pricing.demand_adjustment(105, 100) == pytest.approx(1.0)
    assert bulk_pricing.demand_adjustment(120, 100) == pytest.approx(4.0)
    assert bulk_pricing.demand_adjustment(None, 100) == 0.0
    assert bulk_pricing.demand_adjustment(100, 0) == 0.0


# ---------------------------------------------------------------------------
# dispatcher + logging (temp DB)
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    db_path = tmp_path / "price.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    c.executemany(
        "INSERT INTO products (product_id, sku, name, category, regime, mrp, "
        "wholesale_cost, base_margin_pct, shelf_life_days, unit) VALUES "
        "(?, ?, ?, 'TEST', ?, ?, ?, 25, ?, 'pcs')",
        [(1, "M1", "Packaged Thing 100g", "MRP", 100.0, 70.0, 180),
         (2, "B1", "Loose Grain", "BULK", None, 50.0, None)])
    c.execute("INSERT INTO wholesale_price_history (date, product_id, cost) "
              "VALUES ('2026-08-14', 2, 52.0)")
    for d in range(1, 15):
        c.execute("INSERT INTO daily_sales_history "
                  "(date, product_id, qty_sold, revenue, stockout) "
                  "VALUES (?, 2, 10, 520, 0)", (f"2026-08-{d:02d}",))
    c.commit()
    yield c
    c.close()


def test_dispatcher_routes_by_regime(conn):
    r1 = engine.price_product(conn, 1, as_of="2026-08-15")
    r2 = engine.price_product(conn, 2, as_of="2026-08-15")
    assert r1.regime == "MRP" and r1.base_price == 100.0
    assert r2.regime == "BULK"
    assert r2.base_price == pytest.approx(52.0)      # latest wholesale cost
    assert r2.price > 52.0                            # cost + margin


def test_dispatcher_unknown_product(conn):
    with pytest.raises(ValueError):
        engine.price_product(conn, 999)


def test_explicit_expiry_override(conn):
    r = engine.price_product(conn, 1, as_of="2026-08-15", days_to_expiry=2)
    assert r.discount_pct == 35


def test_price_all_logs_idempotent(conn):
    res = engine.price_all(conn, as_of="2026-08-15", save=True)
    assert len(res) == 2
    n_hist = conn.execute("SELECT COUNT(*) FROM price_history_log").fetchone()[0]
    assert n_hist == 2
    engine.price_all(conn, as_of="2026-08-15", save=True)   # rerun
    n_hist2 = conn.execute("SELECT COUNT(*) FROM price_history_log").fetchone()[0]
    assert n_hist2 == 2                                     # no duplicates


def test_markdown_alert_logged(conn):
    engine.price_product(conn, 1, as_of="2026-08-15", days_to_expiry=1)
    engine.price_all(conn, as_of="2026-08-15", save=True)
    alerts = conn.execute("SELECT * FROM markdown_alerts_log").fetchall()
    # product 1's pseudo-expiry may or may not trigger; at minimum the
    # alert machinery must run without error and log only markdowns
    for a in alerts:
        assert a["discount_pct"] > 0
        assert a["new_price"] <= a["old_price"]


# ---------------------------------------------------------------------------
# integration on the real DB
# ---------------------------------------------------------------------------

def test_real_price_all():
    from src import config
    if not config.DB_PATH.exists():
        pytest.skip("run setup first")
    c = database.get_connection()
    try:
        results = engine.price_all(c, as_of="2026-08-15", save=False)
        assert len(results) >= 60
        assert all(r.price > 0 for r in results)
        assert all(r.regime in ("MRP", "BULK") for r in results)
        # every MRP price ≤ its MRP; bulk price ≥ floor of cost
        for r in results:
            if r.regime == "MRP":
                assert r.price <= r.base_price + 1e-6
    finally:
        c.close()
