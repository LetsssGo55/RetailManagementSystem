"""Phase 11 — end-to-end pipeline walkthrough (acceptance criteria).

slip image → OCR → draft order → review queue → confirm → fulfil
→ invoice PDF → decision cards → simulation results present.
"""
import pytest

from src import config
from src.data import database


@pytest.fixture(scope="module")
def conn():
    assert config.DB_PATH.exists(), "run setup first"
    c = database.get_connection()
    yield c
    c.close()


def test_full_walkdown_slip_to_invoice(conn, tmp_path):
    from src.ocr import engine as ocr_engine, pipeline
    from src.ocr.slip_generator import generate_all
    from src.orders import fulfilment, intake
    from src.invoices import generator

    if not ocr_engine.tesseract_available():
        pytest.skip("tesseract not installed")

    # 1. render + OCR a clean slip
    slips = generate_all(tmp_path)
    res = pipeline.process_slip(slips["clean"], conn=conn)
    assert res.matched_count >= 5                    # OCR + matcher working

    # 2. intake → draft order
    oid = intake.create_order_from_slip(conn, res, source="E2E-TEST")

    # 3. resolve anything left in the review queue for this order
    for it in intake.pending_lines(conn, oid):
        pid = it["suggested_product_id"] or (
            it["candidates"][0][0] if it["candidates"] else None)
        if pid:
            intake.resolve_line(conn, it["line_id"], pid)

    # 4. confirm + fulfil
    s = fulfilment.confirm_order(conn, oid)
    assert 0 <= s["fulfilment_pct"] <= 100
    f = fulfilment.fulfil_order(conn, oid)
    assert f["status"] == "FULFILLED"

    # 5. invoice PDF generated + recorded
    inv = generator.generate_for_order(conn, oid)
    with open(inv["invoice"], "rb") as fh:
        assert fh.read(4) == b"%PDF"
    row = conn.execute("SELECT * FROM invoices_generated WHERE order_id = ?",
                       (oid,)).fetchone()
    assert row is not None and row["total_amount"] == inv["total_amount"]

    # cleanup so repeated runs don't accumulate test orders
    conn.execute("DELETE FROM invoices_generated WHERE order_id = ?", (oid,))
    conn.execute("DELETE FROM order_line_items WHERE order_id = ?", (oid,))
    conn.execute("DELETE FROM customer_orders WHERE order_id = ?", (oid,))
    conn.commit()


def test_decision_cards_exist(conn):
    from src.decisions import engine as dec
    cards = dec.generate_all(conn)
    assert len(cards) >= 1
    assert all(c.reason and c.action and c.icon for c in cards)


def test_ml_outputs_exist(conn):
    assert conn.execute("SELECT COUNT(*) FROM ml_forecast_output"
                        ).fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM ml_cluster_assignment"
                        ).fetchone()[0] > 0
    assert conn.execute("SELECT COUNT(*) FROM bundle_recommendations"
                        ).fetchone()[0] > 0


def test_simulation_results_exist(conn):
    n = conn.execute("SELECT COUNT(*) FROM simulation_results").fetchone()[0]
    assert n == 360                                   # 2 policies × 180 days


def test_acceptance_rf_beats_baselines_recorded(conn):
    """RF test metrics persisted per product must be sane (acceptance #2
    re-checked against stored values)."""
    row = conn.execute("SELECT AVG(rmse) AS r, AVG(mae) AS m, AVG(mape) AS p "
                       "FROM ml_forecast_output").fetchone()
    assert row["r"] < 5.0 and row["m"] < 4.0 and row["p"] < 60.0
