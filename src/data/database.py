"""
ArthaMart — Database layer.
Defines the SQLite schema (16 normalised tables) and provides a connection helper.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
from ..config import load_config, get, ensure_data_dir, DB_PATH

SCHEMA = """
-- ─────────────────────────────────────────────────────────────
-- CATALOG & INVENTORY
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS products (
    product_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    sku             TEXT UNIQUE NOT NULL,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    regime          TEXT NOT NULL CHECK (regime IN ('MRP', 'BULK')),
    mrp             REAL,
    wholesale_cost  REAL NOT NULL,
    base_margin_pct REAL DEFAULT 18.0,
    shelf_life_days INTEGER,
    unit            TEXT DEFAULT 'pcs',
    is_active       INTEGER DEFAULT 1,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS inventory_current (
    product_id     INTEGER PRIMARY KEY REFERENCES products(product_id),
    stock_on_hand  INTEGER NOT NULL DEFAULT 0,
    reorder_level  INTEGER DEFAULT 0,
    last_updated   TEXT DEFAULT (datetime('now'))
);


CREATE TABLE IF NOT EXISTS product_barcodes (
    barcode     TEXT PRIMARY KEY,
    product_id  INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS inventory_movements (
    movement_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    movement_type TEXT NOT NULL,
    quantity      REAL NOT NULL CHECK (quantity > 0),
    change_qty    REAL NOT NULL CHECK (change_qty <> 0),
    stock_before  REAL NOT NULL,
    stock_after   REAL NOT NULL CHECK (stock_after >= 0),
    reference_type TEXT,
    reference_id  INTEGER,
    note          TEXT,
    actor_email   TEXT NOT NULL DEFAULT 'system',
    actor_name    TEXT NOT NULL DEFAULT 'System',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_inventory_movements_product
    ON inventory_movements(product_id, movement_id DESC);
CREATE INDEX IF NOT EXISTS idx_inventory_movements_created
    ON inventory_movements(created_at DESC);

CREATE TABLE IF NOT EXISTS daily_sales_history (
    date        TEXT NOT NULL,
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    qty_sold    INTEGER NOT NULL,
    revenue     REAL NOT NULL,
    stockout    INTEGER DEFAULT 0,
    PRIMARY KEY (date, product_id)
);

CREATE TABLE IF NOT EXISTS wholesale_price_history (
    date        TEXT NOT NULL,
    product_id  INTEGER NOT NULL REFERENCES products(product_id),
    cost        REAL NOT NULL,
    PRIMARY KEY (date, product_id)
);

-- ─────────────────────────────────────────────────────────────
-- ORDERS & INVOICING
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS customers (
    customer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT,
    phone         TEXT,
    segment       TEXT DEFAULT 'NEW',
    total_spend   REAL DEFAULT 0,
    visit_count   INTEGER DEFAULT 0,
    last_visit    TEXT,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS customer_orders (
    order_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id     INTEGER REFERENCES customers(customer_id),
    order_date      TEXT NOT NULL,
    source          TEXT DEFAULT 'slip',
    fulfilment_pct  REAL DEFAULT 0,
    status          TEXT DEFAULT 'PENDING',
    total_value     REAL DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS order_line_items (
    line_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES customer_orders(order_id),
    product_id    INTEGER REFERENCES products(product_id),
    raw_text      TEXT,
    ordered_qty   INTEGER NOT NULL,
    available_qty INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'N/A',
    unit_price    REAL DEFAULT 0,
    line_total    REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS invoices_generated (
    invoice_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES customer_orders(order_id),
    pdf_path      TEXT,
    invoice_date  TEXT NOT NULL,
    total_amount  REAL DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS purchase_orders (
    po_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    qty           INTEGER NOT NULL,
    eoq           REAL,
    rop           REAL,
    priority      REAL,
    status        TEXT DEFAULT 'SUGGESTED',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS procurement_alerts (
    alert_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    alert_type    TEXT,
    message       TEXT,
    created_at    TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────
-- ML & DECISION LOGS
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_forecast_output (
    product_id      INTEGER NOT NULL REFERENCES products(product_id),
    forecast_date   TEXT NOT NULL,
    predicted_qty   REAL NOT NULL,
    model           TEXT,
    rmse            REAL,
    mae             REAL,
    mape            REAL,
    created_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (product_id, forecast_date)
);

CREATE TABLE IF NOT EXISTS ml_cluster_assignment (
    product_id    INTEGER PRIMARY KEY REFERENCES products(product_id),
    cluster       INTEGER,
    tier          TEXT,
    velocity      REAL,
    silhouette    REAL,
    assigned_at   TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS bundle_recommendations (
    rec_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    antecedent_id   INTEGER NOT NULL REFERENCES products(product_id),
    consequent_id   INTEGER NOT NULL REFERENCES products(product_id),
    support         REAL,
    confidence      REAL,
    lift            REAL,
    action          TEXT,
    weekly_impact   REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS markdown_alerts_log (
    alert_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    old_price     REAL,
    new_price     REAL,
    discount_pct  REAL,
    reason        TEXT,
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS price_history_log (
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    price_date    TEXT NOT NULL,
    regime        TEXT,
    price         REAL NOT NULL,
    reason        TEXT,
    PRIMARY KEY (product_id, price_date)
);

-- ─────────────────────────────────────────────────────────────
-- PERSONALISATION
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS personalisation_profiles (
    customer_id     INTEGER PRIMARY KEY REFERENCES customers(customer_id),
    segment         TEXT,
    preferred_cat   TEXT,
    rec_products    TEXT,
    last_updated    TEXT DEFAULT (datetime('now'))
);

-- ─────────────────────────────────────────────────────────────
-- CONFIG / META
-- ─────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS config_parameters (
    key    TEXT PRIMARY KEY,
    value  TEXT,
    description TEXT
);

CREATE TABLE IF NOT EXISTS simulation_results (
    sim_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    policy        TEXT NOT NULL,
    day           INTEGER NOT NULL,
    revenue       REAL DEFAULT 0,
    cogs          REAL DEFAULT 0,
    holding_cost  REAL DEFAULT 0,
    markdown_loss REAL DEFAULT 0,
    stockout_loss REAL DEFAULT 0,
    net_profit    REAL DEFAULT 0,
    created_at    TEXT DEFAULT (datetime('now')),
    UNIQUE (policy, day)
);
"""


def get_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Return a SQLite connection with foreign keys enabled."""
    ensure_data_dir()
    path = Path(db_path) if db_path else DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


CORE_TABLES = [
    "inventory_movements", "product_barcodes",
    "order_line_items", "invoices_generated", "purchase_orders",
    "procurement_alerts", "ml_forecast_output", "ml_cluster_assignment",
    "bundle_recommendations", "markdown_alerts_log", "price_history_log",
    "personalisation_profiles", "simulation_results", "customer_orders",
    "daily_sales_history", "wholesale_price_history", "inventory_current",
    "customers", "products",
]


def reset_database(db_path: Path | str | None = None) -> None:
    """Wipe all data rows (schema kept) so `setup` is idempotent."""
    conn = get_connection(db_path)
    try:
        for t in CORE_TABLES:
            conn.execute(f"DELETE FROM {_assert_ident(t)}")
        conn.execute("DELETE FROM sqlite_sequence")
        conn.commit()
    finally:
        conn.close()


def build_database(db_path: Path | str | None = None) -> Path:
    """Create the database file and all tables. Returns the path."""
    ensure_data_dir()
    path = Path(db_path) if db_path else DB_PATH
    conn = get_connection(path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return path


def load_config_into_db(db_path: Path | str | None = None):
    """Persist key business rules into config_parameters table."""
    cfg = load_config()
    flat = _flatten(cfg)
    conn = get_connection(db_path)
    try:
        for k, v in flat.items():
            conn.execute(
                "INSERT OR REPLACE INTO config_parameters (key, value) VALUES (?, ?)",
                (k, str(v)),
            )
        conn.commit()
    finally:
        conn.close()


def _flatten(d: dict, parent: str = "") -> dict:
    """Flatten nested dict into dot-notation keys."""
    out = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, list):
            out[key] = str(v)
        else:
            out[key] = v
    return out


_IDENT = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _assert_ident(name: str) -> str:
    """Reject anything that is not a plain SQL identifier — the value is
    interpolated into SQL, so it must never carry quotes/spaces/semicolons."""
    if not _IDENT.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def table_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM {_assert_ident(table)}").fetchone()
    return row[0]
