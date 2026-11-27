"""Tests for the database layer and synthetic data generator."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import database
from src import config


@pytest.fixture(scope="module")
def conn():
    db_path = database.build_database()
    database.load_config_into_db()
    c = database.get_connection(db_path)
    yield c
    c.close()


def test_all_tables_created(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {
        "products", "inventory_current", "daily_sales_history",
        "wholesale_price_history", "customers", "customer_orders",
        "order_line_items", "invoices_generated", "purchase_orders",
        "procurement_alerts", "ml_forecast_output", "ml_cluster_assignment",
        "bundle_recommendations", "markdown_alerts_log", "price_history_log",
        "personalisation_profiles", "config_parameters", "simulation_results",
        "stock_ledger",
    }
    missing = expected - tables
    assert not missing, f"Missing tables: {missing}"


def test_config_loaded_into_db(conn):
    row = conn.execute(
        "SELECT value FROM config_parameters WHERE key = ?",
        ("ocr.similarity_threshold",)).fetchone()
    assert row is not None
    assert int(row[0]) == 72


def test_foreign_keys_enabled(conn):
    row = conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1
