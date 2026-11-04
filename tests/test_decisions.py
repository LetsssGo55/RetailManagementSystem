"""Phase 8 — decision card tests."""
import pytest

from src import config
from src.data import database
from src.decisions import engine as dec
from src.decisions.cards import (BUNDLE, HOLD, INVEST_REORDER,
                                 URGENT_MARKDOWN, ICONS, TITLES)


@pytest.fixture()
def conn(tmp_path):
    """Temp DB engineered so every card type fires."""
    db_path = tmp_path / "cards.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    c.executemany(
        "INSERT INTO products (product_id, sku, name, category, regime, mrp, "
        "wholesale_cost, base_margin_pct, shelf_life_days, unit) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, 25, ?, 'pcs')",
        # pid 4, shelf 10 → pseudo days-to-expiry = 10 − (4×37 % 10) = 2
        [(4, "M1", "Near Expiry Snack", "Snacks", "MRP", 100.0, 60.0, 10),
         (5, "F1", "Fast Seller", "Snacks", "MRP", 50.0, 30.0, 180),
         (6, "S1", "Slow Seller", "Spice", "BULK", None, 40.0, 180)])
    c.execute("INSERT INTO inventory_current VALUES (4, 40, 10, 'now')")
    c.execute("INSERT INTO inventory_current VALUES (5, 5, 20, 'now')")    # low!
    c.execute("INSERT INTO inventory_current VALUES (6, 300, 10, 'now')")  # high
    c.executemany(
        "INSERT INTO ml_cluster_assignment (product_id, cluster, tier, "
        "velocity) VALUES (?, ?, ?, 1)",
        [(4, 1, "MEDIUM"), (5, 2, "FAST"), (6, 0, "SLOW")])
    for d in range(1, 31):
        date = f"2026-07-{d:02d}"
        c.execute("INSERT INTO daily_sales_history VALUES (?, 5, 20, 1000, 0)",
                  (date,))
        c.execute("INSERT INTO daily_sales_history VALUES (?, 6, 1, 40, 0)",
                  (date,))
    c.execute(
        "INSERT INTO bundle_recommendations (antecedent_id, consequent_id, "
        "support, confidence, lift, action, weekly_impact) "
        "VALUES (5, 6, 0.03, 0.35, 1.9, 'BUNDLE', 500.0)")
    c.commit()
    yield c
    c.close()


# ---------------------------------------------------------------------------
# individual generators
# ---------------------------------------------------------------------------

def test_markdown_card_fires_for_near_expiry(conn):
    cards = dec.markdown_cards(conn)
    assert len(cards) == 1
    c = cards[0]
    assert c.card_type == URGENT_MARKDOWN
    assert c.product_name == "Near Expiry Snack"
    assert c.detail["discount_pct"] == 35          # 2 days left → 35% rung
    assert c.impact_inr == pytest.approx(40 * 65.0)  # 40 units × ₹65
    assert "expiry" in c.reason.lower()


def test_reorder_card_only_for_fast_movers(conn):
    cards = dec.reorder_cards(conn)
    assert len(cards) == 1
    c = cards[0]
    assert c.card_type == INVEST_REORDER
    assert c.product_name == "Fast Seller"
    assert c.detail["eoq"] > 0 and c.detail["rop"] > 0
    assert c.impact_inr > 0


def test_hold_card_for_overstocked_slow_mover(conn):
    cards = dec.hold_cards(conn)
    assert len(cards) == 1
    c = cards[0]
    assert c.card_type == HOLD
    assert c.product_name == "Slow Seller"
    assert c.detail["stock"] == 300
    assert "cover" in c.reason.lower()
    assert c.impact_inr > 0


def test_bundle_card(conn):
    cards = dec.bundle_cards(conn)
    assert len(cards) == 1
    c = cards[0]
    assert c.card_type == BUNDLE
    assert "Fast Seller" in c.product_name
    assert "Slow Seller" in c.product_name
    assert c.impact_inr == pytest.approx(500.0)
    assert "lift" in c.reason.lower()


# ---------------------------------------------------------------------------
# engine: aggregation, sorting, quality rules
# ---------------------------------------------------------------------------

def test_generate_all_types_present_and_sorted(conn):
    cards = dec.generate_all(conn)
    types = {c.card_type for c in cards}
    assert types == {URGENT_MARKDOWN, INVEST_REORDER, HOLD, BUNDLE}
    impacts = [c.impact_inr for c in cards]
    assert impacts == sorted(impacts, reverse=True)


def test_every_card_has_full_reasoning(conn):
    """Synopsis rule: no bare numbers — icon, title, action, reason, ₹."""
    for c in dec.generate_all(conn):
        assert c.icon and c.icon == ICONS[c.card_type]
        assert c.title == TITLES[c.card_type]
        assert c.action.strip()
        assert c.reason.strip()
        assert len(c.reason) > 10
        assert c.impact_inr >= 0
        rendered = c.render()
        assert "₹" in rendered


def test_unknown_card_type_rendering(conn):
    cards = dec.generate_all(conn)
    assert all(c.render().count("\n") == 3 for c in cards)


# ---------------------------------------------------------------------------
# integration on the real DB
# ---------------------------------------------------------------------------

def test_real_decision_cards():
    if not config.DB_PATH.exists():
        pytest.skip("run setup first")
    c = database.get_connection()
    try:
        cards = dec.generate_all(c)
        assert len(cards) >= 4
        types = {x.card_type for x in cards}
        # current real data produces all four types
        assert types == {URGENT_MARKDOWN, INVEST_REORDER, HOLD, BUNDLE}
        impacts = [x.impact_inr for x in cards]
        assert impacts == sorted(impacts, reverse=True)
    finally:
        c.close()
