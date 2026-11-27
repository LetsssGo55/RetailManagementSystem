"""
ArthaMart — Receipt Scanner (Phase 13)
=======================================

Scan a receipt (image *or* PDF) → extract text → match every line against the
product catalogue by **SKU/barcode** first, then by fuzzy name → look up
current stock → show availability → confirm the order to deduct quantity from
the database (with a full audit ledger entry per product, tagged to the user).

Image slips go through the existing Phase-2 pipeline (preprocess → Tesseract →
fuzzy match). PDFs are handled by a pure-Python text extractor (``pypdf`` /
``PyPDF2`` / ``pdfplumber`` — one must be installed; see ``requirements.txt``)
followed by the same slip parser + matcher.

A dedicated barcode/SKU pass means a scanned barcode number lands on the exact
product even when the OCR'd product name is garbled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

from ..data import database
from ..inventory import management
from . import engine, matcher, parser, pipeline

MATCHED = matcher.MATCHED
REVIEW = matcher.REVIEW
UNMATCHED = matcher.UNMATCHED


class ScanError(RuntimeError):
    """Raised when a receipt cannot be scanned (no OCR, no PDF lib, …)."""


@dataclass
class ScanItem:
    raw_text: str
    qty: float
    product_id: int | None
    matched_name: str | None
    score: float
    match_by: str | None          # 'barcode' | 'name' | None
    status: str                   # MATCHED / REVIEW / UNMATCHED
    stock_on_hand: int
    available_qty: float
    classification: str           # READY / PARTIAL / NOT_AVAILABLE

    def as_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "qty": self.qty,
            "product_id": self.product_id,
            "matched_name": self.matched_name,
            "score": round(self.score, 1),
            "match_by": self.match_by,
            "status": self.status,
            "stock_on_hand": self.stock_on_hand,
            "available_qty": self.available_qty,
            "classification": self.classification,
            "in_stock": self.available_qty > 0,
        }


# ---------------------------------------------------------------------------
# text extraction
# ---------------------------------------------------------------------------
def _extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF using the first available pure-Python lib.

    Tries pypdf → PyPDF2 → pdfplumber. Raises ScanError with an actionable
    message when none is installed.
    """
    last_err: str | None = None

    # --- pypdf -----------------------------------------------------------
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        text = "\n".join(texts)
        if text.strip():
            return text
    except Exception as exc:  # pragma: no cover - dependency availability
        last_err = f"pypdf: {exc}"

    # --- PyPDF2 ----------------------------------------------------------
    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                texts.append("")
        text = "\n".join(texts)
        if text.strip():
            return text
    except Exception as exc:
        last_err = (f"{last_err}; PyPDF2: {exc}"
                    if last_err else f"PyPDF2: {exc}")

    # --- pdfplumber ------------------------------------------------------
    try:
        import pdfplumber  # type: ignore

        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    parts.append("")
        text = "\n".join(parts)
        if text.strip():
            return text
    except Exception as exc:
        last_err = (f"{last_err}; pdfplumber: {exc}"
                    if last_err else f"pdfplumber: {exc}")

    raise ScanError(
        "Cannot extract text from PDF — no PDF library is installed. "
        "Install one of:  pip install pypdf   (or PyPDF2 / pdfplumber)"
    )


def _extract_text(image_src) -> str:
    """Extract text from an image slip (OCR) or a PDF slip."""
    src = Path(image_src) if not hasattr(image_src, "shape") else None
    if src is not None and src.suffix.lower() == ".pdf":
        return _extract_text_from_pdf(src)
    # image → OCR pipeline (engine.ocr_text raises a clear error if tesseract
    # is missing, which we surface to the caller)
    return engine.ocr_text(image_src)


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def _build_catalog(conn: sqlite3.Connection):
    """Load active products plus a sku → product lookup (barcode matching)."""
    rows = conn.execute(
        "SELECT product_id, name, sku, regime, unit FROM products "
        "WHERE is_active = 1 ORDER BY product_id"
    ).fetchall()
    catalog = [
        matcher.CatalogProduct(int(r[0]), r[1], r[2], r[3],
                               matcher.normalize_name(r[1]))
        for r in rows
    ]
    sku_index = {}
    for r in rows:
        sku = (r[2] or "").strip().upper()
        if sku:
            sku_index[sku] = int(r[0])
    return catalog, sku_index


def _product_row(conn, product_id: int):
    return conn.execute(
        "SELECT product_id, name FROM products WHERE product_id = ?",
        (product_id,)).fetchone()


def match_lines_with_stock(parsed_items, conn: sqlite3.Connection):
    """Match parsed slip lines against the catalogue and attach stock."""
    catalog, sku_index = _build_catalog(conn)

    items = []
    for it in parsed_items:
        raw = it.raw
        qty = float(it.quantity or 0.0)

        # 1) barcode / SKU match (exact, case-insensitive)
        pid: int | None = None
        mname: str | None = None
        score = 0.0
        match_by: str | None = None
        status = UNMATCHED

        token = raw.strip().upper()
        if token and token in sku_index:
            pid = sku_index[token]
            prow = _product_row(conn, pid)
            mname = prow["name"] if prow else None
            score = 100.0
            match_by = "barcode"
            status = MATCHED
        else:
            # 2) fuzzy name match
            m = matcher.match_name(raw, catalog)
            pid = m.product_id
            mname = m.matched_name
            score = m.score
            match_by = "name"
            status = m.status

        # stock lookup
        stock = management.stock_on_hand(conn, pid) if pid else 0
        available = min(qty, stock) if pid else 0.0
        if pid is None or stock <= 0:
            classification = "NOT_AVAILABLE"
        elif stock >= qty:
            classification = "READY"
        else:
            classification = "PARTIAL"

        items.append(ScanItem(
            raw_text=raw, qty=qty, product_id=pid,
            matched_name=mname, score=score, match_by=match_by, status=status,
            stock_on_hand=stock, available_qty=available,
            classification=classification,
        ))
    return items


# ---------------------------------------------------------------------------
# public scan
# ---------------------------------------------------------------------------
def scan_file(path, conn: sqlite3.Connection | None = None) -> dict:
    """Scan one receipt file.

    Returns {"items": [ScanItem dicts], "summary": {...}}.

    Items that matched a product carry `product_id`, `matched_name` and their
    current `stock_on_hand` / `available_qty` / `classification`, ready to be
    shown to the user before they confirm.
    """
    close = conn is None
    if conn is None:
        conn = database.get_connection()
    try:
        p = Path(path)
        if not p.exists():
            raise ScanError(f"File not found: {p}")

        # text extraction
        raw_text = None
        pipeline_matches = {}
        if p.suffix.lower() == ".pdf":
            raw_text = _extract_text_from_pdf(p)
        else:
            # reuse the full image pipeline (preprocess + OCR + fuzzy match)
            res = pipeline.process_slip(p, conn=conn)
            raw_text = res.raw_text
            pipeline_matches = {m.query_raw: m for m in res.matches}

        parse = parser.parse_slip_text(raw_text)
        items = match_lines_with_stock(parse.items, conn)

        # for image scans, prefer the pipeline's own fuzzy match when it beat
        # our standalone matcher (keeps preprocessing/deskew benefits)
        if p.suffix.lower() != ".pdf":
            for it in items:
                pm = pipeline_matches.get(it.raw_text)
                if pm is not None and pm.product_id is not None:
                    it.product_id = pm.product_id
                    it.matched_name = pm.matched_name
                    it.score = pm.score
                    it.status = pm.status
                    it.match_by = "name"
                    stock = management.stock_on_hand(conn, pm.product_id)
                    it.stock_on_hand = stock
                    it.available_qty = min(it.qty, stock) if stock > 0 else 0.0
                    it.classification = (
                        "READY" if stock >= it.qty else
                        "PARTIAL" if stock > 0 else "NOT_AVAILABLE")

        matched = [it for it in items if it.product_id is not None]
        return {
            "file": str(p),
            "raw_text": raw_text,
            "items": [it.as_dict() for it in items],
            "summary": {
                "lines": len(items),
                "matched": len(matched),
                "review": sum(i.status == REVIEW for i in items),
                "unmatched": sum(i.status == UNMATCHED for i in items),
                "in_stock": sum(1 for i in matched
                                if i.classification == "READY"),
                "partial": sum(1 for i in matched
                               if i.classification == "PARTIAL"),
                "unavailable": sum(1 for i in matched
                                   if i.classification == "NOT_AVAILABLE"),
            },
        }
    finally:
        if close:
            conn.close()


def confirm_scanned_order(
    conn: sqlite3.Connection,
    items: list[dict],
    user_email: str,
    user_name: str | None = None,
    order_date: str | None = None,
) -> dict:
    """Confirm a scanned order: persist it, deduct available stock per
    product via :func:`management.remove_stock` (change_type ORDER_DEDUCT)
    and record every movement in the audit ledger tagged to `user_email`.

    `items` is the list of ScanItem dicts (from scan_file). Only items with a
    product_id and available_qty > 0 are deducted.

    Returns {"order_id", "lines_deducted", "deductions", "total_value"}.
    """
    import datetime as _dt

    email, name = management._resolve_user(conn, user_email, user_name)

    order_id = None
    total = 0.0
    deducted = []

    # resolve a default customer (walk-in) if none implied
    cust = conn.execute(
        "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 1"
    ).fetchone()
    customer_id = int(cust["customer_id"]) if cust else None
    if customer_id is None:
        raise ScanError("No customers in database — run `python main.py setup`.")

    date = order_date or _dt.date.today().isoformat()
    cur = conn.execute(
        "INSERT INTO customer_orders "
        "(customer_id, order_date, source, fulfilment_pct, status, total_value) "
        "VALUES (?, ?, ?, 0.0, 'CONFIRMED', 0.0)",
        (customer_id, date, "SCAN"),
    )
    order_id = int(cur.lastrowid)

    ordered_total = 0.0
    available_total = 0.0
    for it in items:
        pid = it.get("product_id")
        qty = float(it.get("qty", 0) or 0)
        if not pid or qty <= 0:
            continue
        prod = management.get_product(conn, pid)
        if prod is None:
            continue
        stock = management.stock_on_hand(conn, pid)
        available = min(qty, stock) if stock > 0 else 0
        unit_price = float(prod["mrp"] or prod["wholesale_cost"] * 1.18 or 0)
        line_total = round(unit_price * available, 2)
        total += line_total
        ordered_total += qty
        available_total += available

        status = "NOT_AVAILABLE" if available <= 0 else (
            "READY" if available >= qty else "PARTIAL")
        conn.execute(
            "INSERT INTO order_line_items "
            "(order_id, product_id, raw_text, ordered_qty, available_qty, "
            "status, unit_price, line_total) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (order_id, pid, it.get("raw_text"), qty, available, status,
             unit_price, line_total))

        if available > 0:
            res = management.remove_stock(
                conn, pid, int(available), email, name,
                reason=(f"Scanned order #{order_id}: "
                        f"{it.get('matched_name') or prod['name']}"),
                reference_type="scan", reference_id=order_id,
                change_type="ORDER_DEDUCT",
            )
            conn.execute(
                "UPDATE order_line_items SET status = 'FULFILLED' "
                "WHERE order_id = ? AND product_id = ? AND available_qty > 0 "
                "AND status != 'FULFILLED'",
                (order_id, pid))
            deducted.append({
                "line_raw": it.get("raw_text"),
                "product_id": pid,
                "product_name": prod["name"],
                "ordered_qty": qty,
                "available_qty": available,
                "deducted": int(available),
                "qty_before": res["qty_before"],
                "qty_after": res["qty_after"],
                "unit_price": unit_price,
                "line_total": line_total,
                "user_email": email,
                "user_name": name,
            })

    fulfilment_pct = (round(100.0 * available_total / max(ordered_total, 1), 2)
                      if ordered_total else 0.0)
    conn.execute(
        "UPDATE customer_orders SET total_value = ?, fulfilment_pct = ? "
        "WHERE order_id = ?",
        (round(total, 2), fulfilment_pct, order_id))
    conn.commit()

    return {
        "order_id": order_id,
        "status": "CONFIRMED",
        "total_value": round(total, 2),
        "lines_deducted": len(deducted),
        "deductions": deducted,
        "user_email": email,
        "user_name": name,
    }
