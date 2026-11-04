"""Phase 7 — invoice & procurement PDF generation tests."""
import pytest

from src.data import database
from src.invoices import generator


@pytest.fixture()
def conn(tmp_path):
    """Temp DB with a controllable order: one billable + one unavailable line."""
    db_path = tmp_path / "inv.db"
    database.build_database(db_path)
    c = database.get_connection(db_path)
    c.execute("INSERT INTO customers (customer_id, name, phone, segment) "
              "VALUES (1, 'Test Buyer', '+91111', 'NEW')")
    c.executemany(
        "INSERT INTO products (product_id, sku, name, category, regime, mrp, "
        "wholesale_cost, base_margin_pct, shelf_life_days, unit) VALUES "
        "(?, ?, ?, ?, ?, ?, ?, 25, 180, ?)",
        [(1, "A", "Alpha Biscuit 100g", "Snacks", "MRP", 100.0, 70.0, "pcs"),
         (2, "B", "Rice Loose", "Grain", "BULK", None, 50.0, "kg")])
    # stock: product 1 has plenty, product 2 has ZERO → unavailable
    c.execute("INSERT INTO inventory_current VALUES (1, 100, 10, 'now')")
    c.execute("INSERT INTO inventory_current VALUES (2, 0, 5, 'now')")
    c.execute(
        "INSERT INTO customer_orders (order_id, customer_id, order_date, "
        "source, fulfilment_pct, status, total_value) "
        "VALUES (7, 1, '2026-08-15', 'TEST', 50.0, 'FULFILLED', 250.0)")
    # line 1 billable (available 4), line 2 NOT available (available 0)
    c.executemany(
        "INSERT INTO order_line_items (order_id, product_id, raw_text, "
        "ordered_qty, available_qty, status, unit_price, line_total) "
        "VALUES (7, ?, ?, ?, ?, ?, ?, ?)",
        [(1, "Alpha", 5, 4, "FULFILLED", 100.0, 400.0),
         (2, "Rice", 10, 0, "NOT_AVAILABLE", 60.0, 600.0)])
    c.commit()
    yield c
    c.close()


def test_invoice_pdf_generated_and_readable(conn, tmp_path):
    import pdfplumber
    out = tmp_path / "pdfs"
    res = generator.generate_invoice_pdf(conn, 7, out_dir=out)
    path = res["pdf_path"]
    assert path.endswith("invoice_7.pdf")

    # file exists and is a valid PDF
    with open(path, "rb") as f:
        assert f.read(4) == b"%PDF"

    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text()
    assert "ArthaMart" in text
    assert "TAX INVOICE" in text and "INV-7" in text
    assert "Test Buyer" in text
    assert "Alpha Biscuit 100g" in text          # billable line present
    assert "Rice Loose" not in text               # unavailable line excluded
    assert "3718" in text or "3717" in text or "400" in text  # totals present


def test_invoice_total_uses_available_qty(conn, tmp_path):
    res = generator.generate_invoice_pdf(conn, 7, out_dir=tmp_path)
    # bills available_qty=4 at 100.0 → 400, rounded
    assert res["total_amount"] == pytest.approx(400.0)
    assert res["line_count"] == 1


def test_procurement_note_lists_unavailable(conn, tmp_path):
    import pdfplumber
    out = tmp_path / "proc"
    path = generator.generate_procurement_note_pdf(conn, 7, out_dir=out)
    assert path is not None and path.endswith("procurement_7.pdf")
    with pdfplumber.open(path) as pdf:
        text = pdf.pages[0].extract_text()
    assert "PROCUREMENT NOTE" in text
    assert "Rice Loose" in text                   # the unavailable item
    assert "Alpha Biscuit" not in text            # billable item excluded
    # supplier suggestion for Grain category
    assert "Grain" in text or "Mandi" in text or "Wholesale" in text


def test_no_procurement_note_when_all_fulfilled(conn, tmp_path):
    # order with only billable lines
    conn.execute(
        "INSERT INTO customer_orders (order_id, customer_id, order_date, "
        "source, fulfilment_pct, status, total_value) "
        "VALUES (8, 1, '2026-08-15', 'TEST', 100, 'FULFILLED', 100)")
    conn.execute(
        "INSERT INTO order_line_items (order_id, product_id, raw_text, "
        "ordered_qty, available_qty, status, unit_price, line_total) "
        "VALUES (8, 1, 'Alpha', 2, 2, 'FULFILLED', 100.0, 200.0)")
    conn.commit()
    assert generator.generate_procurement_note_pdf(conn, 8, out_dir=tmp_path) is None


def test_generate_for_order_records_row(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "INVOICE_DIR", tmp_path / "inv")
    monkeypatch.setattr(generator, "PROCUREMENT_DIR", tmp_path / "proc")
    res = generator.generate_for_order(conn, 7)
    assert res["invoice"].endswith("invoice_7.pdf")
    assert res["procurement_note"] is not None

    rows = conn.execute("SELECT * FROM invoices_generated").fetchall()
    assert len(rows) == 1
    assert rows[0]["order_id"] == 7
    assert rows[0]["total_amount"] == pytest.approx(400.0)


def test_generate_for_order_idempotent(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "INVOICE_DIR", tmp_path / "inv")
    monkeypatch.setattr(generator, "PROCUREMENT_DIR", tmp_path / "proc")
    generator.generate_for_order(conn, 7)
    generator.generate_for_order(conn, 7)          # regenerate
    n = conn.execute("SELECT COUNT(*) FROM invoices_generated").fetchone()[0]
    assert n == 1                                  # replaced, not duplicated


def test_unknown_order_raises(conn):
    with pytest.raises(ValueError):
        generator.generate_invoice_pdf(conn, 999)


def test_empty_order_invoice(conn, tmp_path):
    conn.execute(
        "INSERT INTO customer_orders (order_id, customer_id, order_date, "
        "source, fulfilment_pct, status, total_value) "
        "VALUES (9, 1, '2026-08-15', 'TEST', 0, 'CONFIRMED', 0)")
    conn.commit()
    res = generator.generate_invoice_pdf(conn, 9, out_dir=tmp_path)
    assert res["total_amount"] == 0.0
    assert res["line_count"] == 0


# ---------------------------------------------------------------------------
# integration on the real DB (uses a fulfilled OCR-DEMO order)
# ---------------------------------------------------------------------------

def test_real_invoice_generation(tmp_path):
    from src import config
    if not config.DB_PATH.exists():
        pytest.skip("run setup first")
    c = database.get_connection()
    try:
        row = c.execute(
            "SELECT order_id FROM customer_orders "
            "WHERE status = 'FULFILLED' AND source = 'OCR-DEMO' "
            "ORDER BY order_id DESC LIMIT 1").fetchone()
        if row is None:
            pytest.skip("no fulfilled demo order")
        res = generator.generate_for_order(c, row["order_id"],
                                           save_record=False)
        assert res["invoice"].endswith(".pdf")
        assert res["total_amount"] > 0
        with open(res["invoice"], "rb") as f:
            assert f.read(4) == b"%PDF"
    finally:
        c.close()
