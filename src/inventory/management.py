"""
ArthaMart — Operational inventory management and immutable stock ledger.

All manual stock writes are atomic and create an attributable movement record.
Product catalogue rows are retained so historic orders and inventory movements
keep referential integrity.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import sqlite3


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Upgrade an existing SQLite file without committing caller transactions."""
    # Do not use sqlite3.Connection.executescript here: it implicitly commits
    # an open transaction and would break the savepoint around stock updates.
    conn.execute("""
    CREATE TABLE IF NOT EXISTS product_barcodes (
        barcode TEXT PRIMARY KEY,
        product_id INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS inventory_movements (
        movement_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL REFERENCES products(product_id),
        movement_type TEXT NOT NULL,
        quantity REAL NOT NULL CHECK (quantity > 0),
        change_qty REAL NOT NULL CHECK (change_qty <> 0),
        stock_before REAL NOT NULL,
        stock_after REAL NOT NULL CHECK (stock_after >= 0),
        reference_type TEXT,
        reference_id INTEGER,
        note TEXT,
        actor_email TEXT NOT NULL DEFAULT 'system',
        actor_name TEXT NOT NULL DEFAULT 'System',
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_inventory_movements_product
           ON inventory_movements(product_id, movement_id DESC)"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_inventory_movements_created
           ON inventory_movements(created_at DESC)"""
    )

@contextmanager
def _atomic(conn: sqlite3.Connection):
    """Use a savepoint so these writes are safe both standalone and nested."""
    conn.execute("SAVEPOINT inventory_management_write")
    try:
        yield
        conn.execute("RELEASE SAVEPOINT inventory_management_write")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT inventory_management_write")
        conn.execute("RELEASE SAVEPOINT inventory_management_write")
        raise


def _actor(actor: dict | None) -> tuple[str, str]:
    actor = actor or {}
    email = str(actor.get("email") or "system").strip()[:254]
    name = str(actor.get("name") or ("System" if email == "system" else email)).strip()[:120]
    return email or "system", name or "System"


def record_stock_movement(
    conn: sqlite3.Connection,
    product_id: int,
    change_qty: float,
    stock_before: float,
    stock_after: float,
    *,
    movement_type: str,
    actor: dict | None = None,
    reference_type: str | None = None,
    reference_id: int | None = None,
    note: str = "",
) -> int:
    """Insert one ledger row inside the caller's transaction; does not commit."""
    ensure_schema(conn)
    change = float(change_qty)
    before, after = float(stock_before), float(stock_after)
    if not math.isfinite(change) or change == 0:
        raise ValueError("Stock movement must be a finite, non-zero quantity.")
    if not math.isfinite(before) or not math.isfinite(after) or before < 0 or after < 0:
        raise ValueError("Stock balances must be finite and non-negative.")
    email, name = _actor(actor)
    cur = conn.execute(
        """INSERT INTO inventory_movements
           (product_id, movement_type, quantity, change_qty, stock_before,
            stock_after, reference_type, reference_id, note, actor_email, actor_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (int(product_id), str(movement_type).upper(), abs(change), change,
         before, after, reference_type, reference_id, str(note or "")[:500],
         email, name),
    )
    return int(cur.lastrowid)


def adjust_stock(
    conn: sqlite3.Connection,
    product_id: int,
    quantity: int,
    *,
    action: str,
    actor: dict | None = None,
    note: str = "",
) -> dict:
    """Add or remove a whole-number quantity with a guarded atomic update."""
    ensure_schema(conn)
    try:
        qty = float(quantity)
    except (TypeError, ValueError) as exc:
        raise ValueError("Quantity must be a positive whole number.") from exc
    if not math.isfinite(qty) or qty <= 0 or not qty.is_integer():
        raise ValueError("Quantity must be a positive whole number.")
    qty = int(qty)
    action = (action or "").strip().lower()
    if action not in ("add", "remove"):
        raise ValueError("Action must be 'add' or 'remove'.")
    delta = qty if action == "add" else -qty

    with _atomic(conn):
        product = conn.execute(
            "SELECT product_id, name FROM products WHERE product_id = ? AND is_active = 1",
            (int(product_id),),
        ).fetchone()
        if product is None:
            raise ValueError("Product not found or inactive.")
        conn.execute(
            """INSERT OR IGNORE INTO inventory_current
               (product_id, stock_on_hand, reorder_level) VALUES (?, 0, 0)""",
            (int(product_id),),
        )
        row = conn.execute(
            "SELECT stock_on_hand FROM inventory_current WHERE product_id = ?",
            (int(product_id),),
        ).fetchone()
        before = float(row["stock_on_hand"] or 0)
        after = before + delta
        if after < 0:
            raise ValueError(
                f"Insufficient stock for {product['name']}: "
                f"available {before:g}, requested {qty}."
            )
        conn.execute(
            """UPDATE inventory_current
               SET stock_on_hand = ?, last_updated = datetime('now')
               WHERE product_id = ?""",
            (after, int(product_id)),
        )
        movement_id = record_stock_movement(
            conn, int(product_id), delta, before, after,
            movement_type="MANUAL_ADD" if action == "add" else "MANUAL_REMOVE",
            actor=actor, note=note or f"Manual {action} of {qty} units",
        )
    return {
        "product_id": int(product_id), "product": product["name"],
        "stock_before": before, "stock_after": after,
        "change_qty": delta, "movement_id": movement_id,
    }


def create_product(
    conn: sqlite3.Connection,
    *,
    sku: str,
    name: str,
    category: str,
    regime: str,
    wholesale_cost: float,
    mrp: float | None = None,
    base_margin_pct: float = 18.0,
    shelf_life_days: int | None = None,
    unit: str = "pcs",
    barcode: str | None = None,
    opening_stock: int = 0,
    reorder_level: int = 0,
    actor: dict | None = None,
) -> int:
    """Create a catalogue item and its stock row; optionally register a barcode."""
    ensure_schema(conn)
    sku, name = str(sku or "").strip().upper(), str(name or "").strip()
    category, unit = str(category or "").strip(), str(unit or "pcs").strip()
    regime = str(regime or "").strip().upper()
    barcode = str(barcode or "").strip()
    if not sku or len(sku) > 80:
        raise ValueError("SKU is required and must be at most 80 characters.")
    if not name or len(name) > 160:
        raise ValueError("Product name is required and must be at most 160 characters.")
    if not category or len(category) > 80:
        raise ValueError("Category is required and must be at most 80 characters.")
    if not unit or len(unit) > 30:
        raise ValueError("Unit is required and must be at most 30 characters.")
    if regime not in ("MRP", "BULK"):
        raise ValueError("Regime must be MRP or BULK.")
    try:
        cost, margin = float(wholesale_cost), float(base_margin_pct)
        stock, rop = float(opening_stock), float(reorder_level)
        price = None if mrp in (None, "") else float(mrp)
    except (TypeError, ValueError) as exc:
        raise ValueError("Price, stock and margin fields must be numeric.") from exc
    if not math.isfinite(cost) or cost <= 0:
        raise ValueError("Wholesale cost must be greater than zero.")
    if not math.isfinite(margin) or margin < 0 or margin > 200:
        raise ValueError("Base margin must be between 0 and 200 percent.")
    if regime == "MRP" and (price is None or not math.isfinite(price) or price <= 0):
        raise ValueError("MRP products require an MRP greater than zero.")
    if price is not None and (not math.isfinite(price) or price <= 0):
        raise ValueError("MRP must be greater than zero when supplied.")
    if not stock.is_integer() or stock < 0 or not rop.is_integer() or rop < 0:
        raise ValueError("Opening stock and reorder level must be non-negative whole numbers.")
    if shelf_life_days not in (None, ""):
        shelf_life_days = int(shelf_life_days)
        if shelf_life_days <= 0:
            raise ValueError("Shelf life must be a positive number of days.")
    else:
        shelf_life_days = None
    if barcode and (len(barcode) > 128 or any(ch.isspace() for ch in barcode)):
        raise ValueError("Barcode must be at most 128 characters without spaces.")

    with _atomic(conn):
        if conn.execute("SELECT 1 FROM products WHERE sku = ?", (sku,)).fetchone():
            raise ValueError(f"SKU {sku!r} already exists.")
        if barcode and conn.execute(
            "SELECT 1 FROM product_barcodes WHERE barcode = ?", (barcode,)
        ).fetchone():
            raise ValueError("That barcode is already assigned to another product.")
        cur = conn.execute(
            """INSERT INTO products
               (sku, name, category, regime, mrp, wholesale_cost,
                base_margin_pct, shelf_life_days, unit, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (sku, name, category, regime, price, cost, margin,
             shelf_life_days, unit),
        )
        pid = int(cur.lastrowid)
        conn.execute(
            """INSERT INTO inventory_current
               (product_id, stock_on_hand, reorder_level)
               VALUES (?, ?, ?)""",
            (pid, int(stock), int(rop)),
        )
        if barcode:
            conn.execute(
                "INSERT INTO product_barcodes (barcode, product_id) VALUES (?, ?)",
                (barcode, pid),
            )
        if stock > 0:
            record_stock_movement(
                conn, pid, stock, 0, stock, movement_type="OPENING_BALANCE",
                actor=actor, note="Initial stock when product was created",
            )
    return pid


def lookup_barcode(conn: sqlite3.Connection, barcode: str) -> dict | None:
    """Resolve a scanned barcode (or SKU) to the active catalogue and stock."""
    ensure_schema(conn)
    code = str(barcode or "").strip()
    if not code or len(code) > 128:
        return None
    row = conn.execute(
        """SELECT p.product_id, p.sku, p.name, p.category, p.unit,
                  COALESCE(i.stock_on_hand, 0) AS stock,
                  p.regime, p.mrp, p.wholesale_cost
           FROM products p
           LEFT JOIN inventory_current i USING(product_id)
           LEFT JOIN product_barcodes b USING(product_id)
           WHERE p.is_active = 1 AND (b.barcode = ? OR p.sku = ?)
           ORDER BY CASE WHEN b.barcode = ? THEN 0 ELSE 1 END
           LIMIT 1""",
        (code, code.upper(), code),
    ).fetchone()
    return dict(row) if row else None


def list_inventory(conn: sqlite3.Connection) -> list[dict]:
    """List active catalogue products with operational stock and reorder data."""
    ensure_schema(conn)
    return [dict(row) for row in conn.execute(
        """SELECT p.product_id, p.sku, p.name, p.category, p.regime, p.unit,
                  p.mrp, p.wholesale_cost, p.base_margin_pct,
                  COALESCE(i.stock_on_hand, 0) AS stock_on_hand,
                  COALESCE(i.reorder_level, 0) AS reorder_level,
                  i.last_updated,
                  (SELECT b.barcode FROM product_barcodes b
                   WHERE b.product_id = p.product_id ORDER BY b.barcode LIMIT 1) barcode
           FROM products p
           LEFT JOIN inventory_current i USING(product_id)
           WHERE p.is_active = 1
           ORDER BY p.name"""
    ).fetchall()]


def movement_history(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    """Return the newest stock movements, including actor and order references."""
    ensure_schema(conn)
    limit = max(1, min(int(limit), 500))
    return [dict(row) for row in conn.execute(
        """SELECT m.movement_id, m.product_id, p.name AS product, p.sku,
                  m.movement_type, m.quantity, m.change_qty,
                  m.stock_before, m.stock_after, m.reference_type,
                  m.reference_id, m.note, m.actor_email, m.actor_name, m.created_at
           FROM inventory_movements m
           JOIN products p USING(product_id)
           ORDER BY m.movement_id DESC LIMIT ?""",
        (limit,),
    ).fetchall()]
