"""Phase 3 — order intake, availability, queue, fulfilment tests."""
import pytest

from src.data import database
from src.ocr.matcher import MATCHED, REVIEW, UNMATCHED, MatchResult
from src.ocr.parser import ParseResult
from src.ocr.pipeline import SlipResult
from src.orders import availability, fulfilment, intake, queue
from src.orders.intake import OrderError
from src.orders.models import LineStatus, OrderStatus


# ---------------------------------------------------------------------------
# fixtures — fresh temp DB per test
# ---------------------------------------------------------------------------

@pytest.fixture()
def conn(tmp_path):
    db_path = tmp_path / "test.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    c.execute("INSERT INTO customers (name, phone, segment) "
              "VALUES ('Test Customer', '+910000000000', 'NEW')")
    c.executemany(
        "INSERT INTO products (product_id, sku, name, category, regime, mrp, "
        "wholesale_cost, base_margin_pct, shelf_life_days, unit) "
        "VALUES (?, ?, ?, 'TEST', ?, ?, ?, 20, 180, 'pcs')",
        [(1, "SKU1", "Alpha Biscuit 100g", "MRP", 100.0, 70.0),
         (2, "SKU2", "Rice Loose", "BULK", 0.0, 50.0),
         (3, "SKU3", "Beta Tea 250g", "MRP", 10.0, 6.0)])
    # margin for product 2 = 20% → unit price 60.0
    c.execute("UPDATE products SET base_margin_pct = 20 WHERE product_id = 2")
    c.executemany(
        "INSERT INTO inventory_current (product_id, stock_on_hand, reorder_level) "
        "VALUES (?, ?, ?)",
        [(1, 5, 2), (2, 0, 5), (3, 100, 10)])
    c.commit()
    yield c
    c.close()


def make_match(raw, qty, product_id, status):
    return MatchResult(query_raw=raw, quantity=qty, product_id=product_id,
                       matched_name=None, score=99.0, status=status)


def make_slip(matches):
    return SlipResult(image_path=None, raw_text="", parse=ParseResult(),
                      matches=matches,
                      matched_count=sum(m.status == MATCHED for m in matches),
                      review_count=sum(m.status == REVIEW for m in matches),
                      unmatched_count=sum(m.status == UNMATCHED for m in matches))


# ---------------------------------------------------------------------------
# intake
# ---------------------------------------------------------------------------

def test_create_draft_order(conn):
    slip = make_slip([
        make_match("Alpha Biscuit 100g 2", 2, 1, MATCHED),
        make_match("Beta Tea 250g 3", 3, 3, MATCHED),
    ])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    header = conn.execute("SELECT * FROM customer_orders WHERE order_id = ?",
                          (oid,)).fetchone()
    assert header["status"] == OrderStatus.DRAFT.value
    # 2×100 + 3×10
    assert header["total_value"] == pytest.approx(230.0)
    lines = conn.execute("SELECT * FROM order_line_items WHERE order_id = ?",
                         (oid,)).fetchall()
    assert len(lines) == 2
    assert {l["status"] for l in lines} == {LineStatus.MATCHED.value}


def test_unmatched_line_kept_without_product(conn):
    slip = make_slip([make_match("Zebra Flux Capacitor 9", 9, None, UNMATCHED)])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    ln = conn.execute("SELECT * FROM order_line_items WHERE order_id = ?",
                      (oid,)).fetchone()
    assert ln["product_id"] is None
    assert ln["status"] == LineStatus.UNMATCHED.value


def test_empty_slip_rejected(conn):
    with pytest.raises(OrderError):
        intake.create_order_from_slip(conn, make_slip([]), customer_id=1)


def test_review_queue_and_resolution(conn):
    slip = make_slip([
        make_match("Alpha Biscut 100g", 2, 1, REVIEW),
        make_match("Rise Loose", 5, None, UNMATCHED),
    ])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)

    pend = intake.pending_lines(conn, oid)
    assert len(pend) == 2
    assert all(p["candidates"] for p in pend)          # suggestions present
    # top candidate for 'Rise Loose' should be Rice Loose (id 2)
    rice = [p for p in pend if p["suggested_product_id"] is None][0]
    assert rice["candidates"][0][0] == 2

    intake.resolve_line(conn, rice["line_id"], 2)
    ln = conn.execute("SELECT * FROM order_line_items WHERE line_id = ?",
                      (rice["line_id"],)).fetchone()
    assert ln["status"] == LineStatus.CONFIRMED.value
    assert ln["product_id"] == 2
    assert ln["line_total"] == pytest.approx(5 * 60.0)   # bulk price 60
    assert intake.pending_lines(conn, oid)  # REVIEW line still pending

    with pytest.raises(OrderError):                       # already resolved
        intake.resolve_line(conn, rice["line_id"], 3)


# ---------------------------------------------------------------------------
# availability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ordered,stock,expected", [
    (5, 5, LineStatus.READY),          # exact fit
    (5, 50, LineStatus.READY),         # overstock
    (6, 5, LineStatus.PARTIAL),        # short
    (6, 0, LineStatus.NOT_AVAILABLE),  # zero stock
])
def test_classify_edge_cases(ordered, stock, expected):
    assert availability.classify(ordered, stock) == expected.value


def test_fulfilment_score_computation(conn):
    slip = make_slip([
        make_match("Alpha", 10, 1, MATCHED),   # stock 5  → 5/10
        make_match("Beta", 4, 3, MATCHED),     # stock 100 → 4/4
        make_match("Rice", 3, 2, MATCHED),     # stock 0  → 0/3
    ])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    s = availability.check_order(conn, oid)
    assert s["ready"] == 1 and s["partial"] == 1 and s["unavailable"] == 1
    assert s["fulfilment_pct"] == pytest.approx(100 * 9 / 17)

    ln = conn.execute(
        "SELECT status, available_qty FROM order_line_items "
        "WHERE order_id = ? AND product_id = 1", (oid,)).fetchone()
    assert ln["status"] == LineStatus.PARTIAL.value
    assert ln["available_qty"] == 5


def test_zero_available_order_scores_zero(conn):
    slip = make_slip([make_match("Rice", 3, 2, MATCHED)])   # stock 0
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    s = availability.check_order(conn, oid)
    assert s["fulfilment_pct"] == 0.0


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------

def _insert_order(conn, oid, score, value, date, status="CONFIRMED"):
    conn.execute(
        "INSERT INTO customer_orders (order_id, customer_id, order_date, "
        "source, fulfilment_pct, status, total_value) "
        "VALUES (?, 1, ?, 'TEST', ?, ?, ?)",
        (oid, date, score, status, value))


def test_queue_priority_rules(conn):
    # same score+value → FIFO by date; higher score first; higher value next
    _insert_order(conn, 101, 100.0, 500.0, "2026-08-02")
    _insert_order(conn, 102, 100.0, 500.0, "2026-08-01")   # older → wins tie
    _insert_order(conn, 103, 40.0, 900.0, "2026-08-01")    # partial, big value
    _insert_order(conn, 104, 40.0, 100.0, "2026-07-30")    # partial, small
    _insert_order(conn, 105, 100.0, 999.0, "2026-08-15")   # full + biggest
    _insert_order(conn, 106, 100.0, 500.0, "2026-08-01",
                  status="FULFILLED")                       # excluded

    q = queue.build_queue(conn)
    ids = [e.order_id for e in q]
    assert ids == [105, 102, 101, 103, 104]
    flags = {e.order_id: e.dispatch_flag for e in q}
    assert flags[105] == "DISPATCH"
    assert flags[103] == "PARTIAL"
    assert 106 not in flags                                 # fulfilled excluded


def test_queue_hold_flag(conn):
    _insert_order(conn, 201, 0.0, 300.0, "2026-08-01")
    q = queue.build_queue(conn)
    assert q[0].dispatch_flag == "HOLD"


# ---------------------------------------------------------------------------
# confirm + fulfil
# ---------------------------------------------------------------------------

def test_confirm_blocked_by_unresolved_lines(conn):
    slip = make_slip([
        make_match("Alpha", 2, 1, MATCHED),
        make_match("Mystery Item", 1, None, UNMATCHED),
    ])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    with pytest.raises(OrderError):
        fulfilment.confirm_order(conn, oid)


def test_full_lifecycle_confirm_fulfil(conn):
    slip = make_slip([
        make_match("Alpha", 3, 1, MATCHED),    # stock 5 → READY
        make_match("Beta", 2, 3, MATCHED),     # stock 100 → READY
    ])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    s = fulfilment.confirm_order(conn, oid)
    assert s["fulfilment_pct"] == 100.0
    assert s["status"] == OrderStatus.CONFIRMED.value

    f = fulfilment.fulfil_order(conn, oid)
    assert f["lines_shipped"] == 2
    assert f["fulfilment_pct"] == 100.0

    stock = {r["product_id"]: r["stock_on_hand"] for r in conn.execute(
        "SELECT product_id, stock_on_hand FROM inventory_current")}
    assert stock[1] == 5 - 3
    assert stock[3] == 100 - 2

    header = conn.execute("SELECT status FROM customer_orders WHERE order_id = ?",
                          (oid,)).fetchone()
    assert header["status"] == OrderStatus.FULFILLED.value

    with pytest.raises(OrderError):                     # double fulfil blocked
        fulfilment.fulfil_order(conn, oid)


def test_partial_fulfilment_and_low_stock(conn):
    slip = make_slip([
        make_match("Alpha", 6, 1, MATCHED),    # stock 5 < 6 → PARTIAL, ship 5
        make_match("Rice", 2, 2, MATCHED),     # stock 0 → nothing
    ])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    fulfilment.confirm_order(conn, oid)
    f = fulfilment.fulfil_order(conn, oid)

    assert f["lines_shipped"] == 1
    assert f["fulfilment_pct"] == pytest.approx(100 * 5 / 8)   # 62.5%
    assert 1 in f["low_stock_products"]          # 5-5 = 0 ≤ reorder level 2

    row = conn.execute(
        "SELECT stock_on_hand FROM inventory_current WHERE product_id = 1"
    ).fetchone()
    assert row["stock_on_hand"] == 0


def test_overorder_never_drives_stock_negative(conn):
    slip = make_slip([make_match("Beta", 500, 3, MATCHED)])   # stock 100
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    fulfilment.confirm_order(conn, oid)
    fulfilment.fulfil_order(conn, oid)
    row = conn.execute(
        "SELECT stock_on_hand FROM inventory_current WHERE product_id = 3"
    ).fetchone()
    assert row["stock_on_hand"] == 0               # floored at zero


def test_cancel_blocks_fulfil(conn):
    slip = make_slip([make_match("Alpha", 1, 1, MATCHED)])
    oid = intake.create_order_from_slip(conn, slip, customer_id=1)
    fulfilment.cancel_order(conn, oid)
    with pytest.raises(OrderError):
        fulfilment.fulfil_order(conn, oid)
