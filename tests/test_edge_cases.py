"""Phase 11 — QA edge cases: threshold boundaries, zero demand, stock
guards, duplicate lines, idempotent setup."""
import pytest

from src import config
from src.data import database
from src.inventory import eoq, reorder, rop
from src.ocr import matcher
from src.ocr.matcher import MATCHED, REVIEW, UNMATCHED, CatalogProduct
from src.orders import fulfilment, intake
from src.ocr.matcher import MatchResult
from src.ocr.parser import ParseResult
from src.ocr.pipeline import SlipResult


def _catalog():
    return [CatalogProduct(1, "Alpha Biscuit 100g", "MRP", "pcs",
                           "alpha biscuit 100g"),
            CatalogProduct(2, "Rice Basmati (loose)", "BULK", "kg",
                           "rice basmati loose")]


# ---------------------------------------------------------------------------
# fuzzy-match threshold boundaries (72% default)
# ---------------------------------------------------------------------------

def test_match_threshold_boundaries(monkeypatch):
    """Scores exactly at the boundary: 72 → MATCHED, 71.9 → REVIEW,
    54.9 → UNMATCHED (review floor 55)."""
    cat = _catalog()

    def fake_score(q, c):
        return fake_score.value
    monkeypatch.setattr(matcher, "_score", fake_score)

    fake_score.value = 72.0
    assert matcher.match_name("whatever", cat).status == MATCHED
    fake_score.value = 71.9
    assert matcher.match_name("whatever", cat).status == REVIEW
    fake_score.value = 55.0
    assert matcher.match_name("whatever", cat).status == REVIEW
    fake_score.value = 54.9
    assert matcher.match_name("whatever", cat).status == UNMATCHED
    fake_score.value = 0.0
    assert matcher.match_name("whatever", cat).status == UNMATCHED


def test_empty_query_and_empty_catalog():
    assert matcher.match_name("", _catalog()).status == UNMATCHED
    assert matcher.match_name("Alpha Biscuit", []).status == UNMATCHED


# ---------------------------------------------------------------------------
# zero / degenerate demand
# ---------------------------------------------------------------------------

def test_zero_demand_no_order():
    assert eoq.eoq(0) == 0.0
    assert eoq.recommended_order_qty(0, unit_cost=100) == 0


def test_zero_demand_product_gets_no_po(tmp_path):
    db_path = tmp_path / "zero.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    try:
        c.execute("INSERT INTO products (product_id, sku, name, category, "
                  "regime, mrp, wholesale_cost, base_margin_pct, unit) "
                  "VALUES (1, 'Z', 'Dead Product', 'TEST', 'BULK', NULL, "
                  "50.0, 20, 'pcs')")
        c.execute("INSERT INTO inventory_current VALUES (1, 0, 5, 'now')")
        for d in range(1, 29):                       # 28 days of zero sales
            c.execute("INSERT INTO daily_sales_history VALUES "
                      "(?, 1, 0, 0, 0)", (f"2026-07-{d:02d}",))
        c.commit()
        s = reorder.run_reorder_cycle(c, save=True)
        assert s["orders_recommended"] == 0          # no demand → no PO
    finally:
        c.close()


def test_safety_stock_zero_sigma():
    assert rop.safety_stock(0, lead_time=3, z=1.65) == 0.0


# ---------------------------------------------------------------------------
# stock can never go negative
# ---------------------------------------------------------------------------

def _make_match(raw, qty, pid, status=MATCHED):
    return MatchResult(query_raw=raw, quantity=qty, product_id=pid,
                       matched_name=None, score=99.0, status=status)


def _make_slip(matches):
    return SlipResult(image_path=None, raw_text="", parse=ParseResult(),
                      matches=matches, matched_count=len(matches))


def test_concurrent_orders_cannot_drive_stock_negative(tmp_path):
    db_path = tmp_path / "neg.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    try:
        c.execute("INSERT INTO customers (customer_id, name, phone, segment) "
                  "VALUES (1, 'Buyer', '+91', 'NEW')")
        c.execute("INSERT INTO products (product_id, sku, name, category, "
                  "regime, mrp, wholesale_cost, base_margin_pct, unit) "
                  "VALUES (1, 'A', 'Alpha', 'TEST', 'MRP', 100, 70, 25, 'pcs')")
        c.execute("INSERT INTO inventory_current VALUES (1, 10, 2, 'now')")
        c.commit()

        # two orders of 8 units each — only 10 in stock total
        o1 = intake.create_order_from_slip(c, _make_slip(
            [_make_match("Alpha 8", 8, 1)]), customer_id=1)
        o2 = intake.create_order_from_slip(c, _make_slip(
            [_make_match("Alpha 8", 8, 1)]), customer_id=1)
        fulfilment.confirm_order(c, o1)
        fulfilment.confirm_order(c, o2)
        f1 = fulfilment.fulfil_order(c, o1)
        f2 = fulfilment.fulfil_order(c, o2)

        stock = c.execute("SELECT stock_on_hand FROM inventory_current "
                          "WHERE product_id = 1").fetchone()[0]
        assert stock >= 0                            # never negative
        assert f1["lines_shipped"] == 1              # first ships 8
        assert f2["lines_shipped"] == 1              # second ships rest (2)
        assert stock == 0                            # 10 − 8 − 2
    finally:
        c.close()


# ---------------------------------------------------------------------------
# duplicate product lines in one order
# ---------------------------------------------------------------------------

def test_duplicate_product_lines(tmp_path):
    db_path = tmp_path / "dup.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    try:
        c.execute("INSERT INTO customers (customer_id, name, phone, segment) "
                  "VALUES (1, 'Buyer', '+91', 'NEW')")
        c.execute("INSERT INTO products (product_id, sku, name, category, "
                  "regime, mrp, wholesale_cost, base_margin_pct, unit) "
                  "VALUES (1, 'A', 'Alpha', 'TEST', 'MRP', 100, 70, 25, 'pcs')")
        c.execute("INSERT INTO inventory_current VALUES (1, 100, 2, 'now')")
        c.commit()

        oid = intake.create_order_from_slip(c, _make_slip([
            _make_match("Alpha 2", 2, 1),
            _make_match("Alpha 3", 3, 1),           # same product again
        ]), customer_id=1)
        fulfilment.confirm_order(c, oid)
        f = fulfilment.fulfil_order(c, oid)
        assert f["lines_shipped"] == 2
        stock = c.execute("SELECT stock_on_hand FROM inventory_current "
                          "WHERE product_id = 1").fetchone()[0]
        assert stock == 100 - 5                      # both lines deducted
    finally:
        c.close()


# ---------------------------------------------------------------------------
# idempotent setup (acceptance criterion #1 foundation)
# ---------------------------------------------------------------------------

def test_reset_database_clears_rows(tmp_path):
    db_path = tmp_path / "reset.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    c.execute("INSERT INTO products (sku, name, category, regime, "
              "wholesale_cost) VALUES ('X', 'Temp', 'TEST', 'BULK', 10)")
    c.commit()
    c.close()
    database.reset_database(db_path)
    c = database.get_connection(db_path)
    try:
        assert c.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0
        # schema still intact
        cols = [r[1] for r in c.execute("PRAGMA table_info(products)")]
        assert "product_id" in cols
    finally:
        c.close()


def test_synthetic_generator_deterministic(tmp_path):
    """Same seed → same products (setup reproducibility)."""
    from src.data.generate_synthetic import generate_products
    a = generate_products()
    b = generate_products()
    assert [p["name"] for p in a] == [p["name"] for p in b]
    assert len(a) == int(config.get("data.n_products", 65))
