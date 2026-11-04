"""Personalisation engine tests — segmentation + recommendations."""
import pytest

from src.data import database
from src.personalisation import engine as pers


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    # pin segmentation thresholds regardless of the shipped config values
    from src import config
    cfg = config.load_config()
    monkeypatch.setitem(cfg["personalisation"], "vip_threshold_spend", 5000)
    monkeypatch.setitem(cfg["personalisation"], "at_risk_days_inactive", 21)
    db_path = tmp_path / "pers.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    c.executemany(
        "INSERT INTO products (product_id, sku, name, category, regime, "
        "mrp, wholesale_cost, base_margin_pct, unit) VALUES "
        "(?, ?, ?, 'TEST', 'MRP', 100, 70, 25, 'pcs')",
        [(i, f"S{i}", f"Product {i}") for i in range(1, 6)])
    c.executemany("INSERT INTO customers (customer_id, name, phone) VALUES "
                  "(?, ?, '+91')",
                  [(1, "VIP Buyer"), (2, "Sleepy Buyer"),
                   (3, "New Buyer"), (4, "Regular Buyer")])
    # basket rule 1 → 2 (lift 2.0)
    c.execute("INSERT INTO bundle_recommendations (antecedent_id, "
              "consequent_id, support, confidence, lift, action, "
              "weekly_impact) VALUES (1, 2, .03, .5, 2.0, 'CROSS-PROMOTE', 9)")
    # tiers for cold-start bestsellers
    c.executemany("INSERT INTO ml_cluster_assignment (product_id, cluster, "
                  "tier, velocity) VALUES (?, 0, 'FAST', ?)",
                  [(3, 15.0), (4, 12.0), (5, 10.0)])

    def order(oid, cust, date, pids_total):
        c.execute("INSERT INTO customer_orders (order_id, customer_id, "
                  "order_date, source, status, total_value) VALUES "
                  "(?, ?, ?, 'TEST', 'FULFILLED', 0)", (oid, cust, date))
        for li, (pid, total) in enumerate(pids_total, 1):
            c.execute("INSERT INTO order_line_items (order_id, product_id, "
                      "raw_text, ordered_qty, available_qty, status, "
                      "unit_price, line_total) VALUES (?, ?, 'x', 1, 1, "
                      "'FULFILLED', ?, ?)", (oid, pid, total, total))

    # VIP: 3 orders, ₹6,000, recent (buys 1 & 5 → expects cross-sell rec 2)
    order(1, 1, "2026-08-10", [(1, 2000.0)])
    order(2, 1, "2026-08-12", [(5, 2000.0)])
    order(3, 1, "2026-08-14", [(1, 2000.0)])
    # AT_RISK: spent but last order 40 days before the newest order
    order(4, 2, "2026-07-05", [(3, 500.0)])
    # NEW: single order
    order(5, 3, "2026-08-13", [(4, 100.0)])
    # REGULAR: 4 small recent orders
    for i in range(4):
        order(10 + i, 4, f"2026-08-1{i}", [(5, 200.0)])
    c.commit()
    yield c
    c.close()


def test_segments(conn):
    df = pers.compute_segments(conn)
    seg = dict(zip(df["customer_id"], df["segment"]))
    assert seg[1] == "VIP"          # ₹6,000 ≥ 5,000
    assert seg[2] == "AT_RISK"      # 40 days inactive
    assert seg[3] == "NEW"          # one order
    assert seg[4] == "REGULAR"      # four orders


def test_segment_rules_unit(monkeypatch):
    from src import config
    cfg = config.load_config()
    monkeypatch.setitem(cfg["personalisation"], "vip_threshold_spend", 5000)
    monkeypatch.setitem(cfg["personalisation"], "at_risk_days_inactive", 21)
    assert pers.segment_for(6000, 1, 1) == "VIP"
    assert pers.segment_for(100, 2, 30) == "AT_RISK"
    assert pers.segment_for(100, 3, 2) == "REGULAR"
    assert pers.segment_for(100, 1, 2) == "NEW"


def test_recommend_cross_sell_excludes_purchased(conn):
    recs = pers.recommend(conn, 1)
    ids = [r["product_id"] for r in recs]
    assert 1 not in ids                       # already bought
    assert ids[0] == 2                        # lift rule wins
    assert "often add" in recs[0]["reason"].lower() or "lift" in \
        recs[0]["reason"].lower()


def test_recommend_top_n_and_cold_start(conn):
    recs = pers.recommend(conn, 3, top_n=3)   # 1 purchase < min_history 2
    assert len(recs) <= 3
    assert all(r["product_id"] in (3, 4, 5) for r in recs)  # FAST tier


def test_build_profiles_persists(conn):
    df = pers.build_profiles(conn, save=True)
    n = conn.execute("SELECT COUNT(*) FROM personalisation_profiles") \
        .fetchone()[0]
    assert n == len(df) == 4
    row = conn.execute("SELECT segment, rec_products FROM "
                       "personalisation_profiles WHERE customer_id = 1") \
        .fetchone()
    assert row["segment"] == "VIP"
    assert row["rec_products"].startswith("2")
    # customers summary refreshed
    cust = conn.execute("SELECT segment, total_spend, visit_count FROM "
                        "customers WHERE customer_id = 1").fetchone()
    assert cust["segment"] == "VIP"
    assert cust["total_spend"] == pytest.approx(6000.0)
    assert cust["visit_count"] == 3
