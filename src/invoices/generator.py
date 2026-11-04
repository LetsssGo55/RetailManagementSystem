"""
ArthaMart — Invoice & Procurement Generation (Phase 7)
========================================================
ReportLab PDFs:

    INVOICE            for every order — bills the ship-able lines
                       (available_qty > 0) at their unit prices
    PROCUREMENT NOTE   for orders with NOT_AVAILABLE lines — what to buy,
                       from which suggested supplier

Every generated invoice is recorded in `invoices_generated`
(idempotent: one row per order, replaced on regeneration).
"""
from __future__ import annotations

import datetime as _dt
import sqlite3
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from ..config import ROOT, get

# Unicode-capable fonts (₹ etc.) — fall back to Helvetica if unavailable
FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
try:
    pdfmetrics.registerFont(
        TTFont("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    pdfmetrics.registerFont(
        TTFont("DejaVuSans-Bold",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
    FONT, FONT_BOLD = "DejaVuSans", "DejaVuSans-Bold"
except Exception:
    pass


def _styles():
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = FONT
    return styles

INVOICE_DIR = ROOT / "data" / "invoices"
PROCUREMENT_DIR = ROOT / "data" / "procurement"

BRAND = get("project.name", "ArthaMart")
BRAND_LINE = "Intelligent Retailing — Forecasting · Pricing · Personalisation"
SHOP_ADDRESS = "ArthaMart Retail, Station Road, Bareilly, Uttar Pradesh 243001"
SHOP_PHONE = "+91 98765 43210"
GSTIN = "09ARTHA2026M1Z5"

# deterministic supplier suggestions per category (mandi-style sourcing)
SUPPLIERS = {
    "Grain": "Agarwal Grain Mandi, Bareilly",
    "Pulses": "Shree Dal Bhandar Wholesale",
    "Oil": "Fortune Distributors (UP)",
    "Spice": "MDH Wholesale Spices",
    "Vegetable": "Sabzi Mandi Yard, Bareilly",
    "Fruit": "Fruit Market Association, Bareilly",
    "Dairy": "Mother Dairy Cold-Chain",
    "Beverages": "Coca-Cola Beverages Distributor",
    "Snacks": "Haldiram Regional Distributor",
    "Namkeen": "Haldiram Regional Distributor",
    "Confectionery": "Cadbury Stockist, Bareilly",
    "Household": "HUL Super Stockist",
    "Personal Care": "HUL Super Stockist",
    "Pet Care": "Drools Pet Distributor",
    "Dry Fruit": "Khari Baoli Dry Fruit Traders",
    "Grocery": "Local Wholesale Market",
}
DEFAULT_SUPPLIER = "Local Wholesale Market"


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------

def _order(conn: sqlite3.Connection, order_id: int):
    row = conn.execute(
        "SELECT o.*, c.name AS customer_name FROM customer_orders o "
        "LEFT JOIN customers c USING (customer_id) WHERE o.order_id = ?",
        (order_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown order_id={order_id}")
    return row


def _billable_lines(conn, order_id: int):
    return conn.execute(
        "SELECT l.*, p.name, p.unit FROM order_line_items l "
        "JOIN products p USING (product_id) "
        "WHERE l.order_id = ? AND l.product_id IS NOT NULL "
        "  AND COALESCE(l.available_qty, 0) > 0 ORDER BY l.line_id",
        (order_id,)).fetchall()


def _unavailable_lines(conn, order_id: int):
    return conn.execute(
        "SELECT l.*, p.name, p.unit, p.category FROM order_line_items l "
        "JOIN products p USING (product_id) "
        "WHERE l.order_id = ? AND l.product_id IS NOT NULL "
        "  AND COALESCE(l.available_qty, 0) <= 0 ORDER BY l.line_id",
        (order_id,)).fetchall()


# ---------------------------------------------------------------------------
# shared styling
# ---------------------------------------------------------------------------

def _table_style(header_bg: colors.Color) -> TableStyle:
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.Color(0.95, 0.95, 0.98)]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])


def _cell_style(size: int = 8.5, bold: bool = False,
                align: str = "LEFT", color: str = "#222222") -> ParagraphStyle:
    return ParagraphStyle(
        f"cell{size}{'b' if bold else ''}{align}{color[1:]}",
        fontName=FONT_BOLD if bold else FONT, fontSize=size,
        leading=size + 2.5, alignment={"LEFT": 0, "RIGHT": 2,
                                       "CENTER": 1}[align],
        textColor=colors.HexColor(color))


def _cell(text, size: int = 8.5, bold: bool = False,
          align: str = "LEFT", color: str = "#222222") -> Paragraph:
    """Wrapped table cell — plain strings overflow ReportLab table cells
    and overlap neighbours, Paragraphs wrap and grow the row instead."""
    from xml.sax.saxutils import escape
    return Paragraph(escape(str(text)), _cell_style(size, bold, align, color))


def _header_row(cells, sizes=None) -> list:
    return [_cell(c, size=9, bold=True, align="LEFT", color="#FFFFFF")
            for c in cells]


def _header_block(story: list, title: str):
    styles = _styles()
    title_style = ParagraphStyle("inv_title", parent=styles["Normal"],
                                 fontSize=22, leading=26,
                                 textColor=colors.HexColor("#1a237e"))
    tag_style = ParagraphStyle("inv_tag", parent=styles["Normal"],
                               fontSize=8, leading=10,
                               textColor=colors.HexColor("#555555"))
    left = [
        Paragraph(f"<b>{BRAND}</b>", title_style),
        Paragraph(BRAND_LINE, tag_style),
    ]
    logo = ROOT / "assets" / "logo.png"
    if logo.exists():
        from reportlab.platypus import Image as RLImage
        right: list = [RLImage(str(logo), width=16 * mm, height=16 * mm)]
    else:
        right = [Spacer(1, 1)]
    hdr = Table([[left, right]], colWidths=[152 * mm, 28 * mm])
    hdr.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(f"<b>{title}</b>", styles["Normal"]))
    story.append(Paragraph(
        f'<font size="8">{SHOP_ADDRESS} · {SHOP_PHONE} · GSTIN: {GSTIN}'
        "</font>", styles["Normal"]))
    story.append(Spacer(1, 4 * mm))


# ---------------------------------------------------------------------------
# invoice
# ---------------------------------------------------------------------------

def generate_invoice_pdf(conn: sqlite3.Connection, order_id: int,
                         out_dir: Path | None = None) -> dict:
    """Render the customer invoice for the ship-able portion of an order."""
    order = _order(conn, order_id)
    lines = _billable_lines(conn, order_id)
    out_dir = Path(out_dir) if out_dir else INVOICE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"invoice_{order_id}.pdf"

    subtotal = 0.0
    rows = [_header_row(["#", "Item", "Qty", "Unit", "Rate (₹)",
                         "Amount (₹)"])]
    for i, ln in enumerate(lines, 1):
        amount = round(float(ln["unit_price"]) * float(ln["available_qty"]), 2)
        subtotal += amount
        qty = f"{ln['available_qty']:g}"
        rows.append([
            _cell(i, align="CENTER"),
            _cell(ln["name"]),                       # wraps, never overlaps
            _cell(qty, align="RIGHT"),
            _cell(ln["unit"] or "pcs", align="CENTER"),
            _cell(f"{ln['unit_price']:.2f}", align="RIGHT"),
            _cell(f"{amount:.2f}", align="RIGHT", bold=True),
        ])

    if not lines:
        rows.append([_cell(""), _cell("No billable lines"), _cell(""),
                     _cell(""), _cell(""), _cell("0.00", align="RIGHT")])

    rounded = round(subtotal)
    round_off = round(rounded - subtotal, 2)

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    story: list = []
    _header_block(story, f"TAX INVOICE — INV-{order_id}")

    styles = _styles()
    meta = (f"Customer: <b>{order['customer_name'] or 'Walk-in'}</b> &nbsp;&nbsp; "
            f"Order date: {order['order_date']} &nbsp;&nbsp; "
            f"Source: {order['source']} &nbsp;&nbsp; "
            f"Order status: {order['status']}")
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 4 * mm))

    t = Table(rows, colWidths=[8 * mm, 76 * mm, 13 * mm, 14 * mm,
                               26 * mm, 28 * mm])
    t.setStyle(_table_style(colors.Color(0.10, 0.14, 0.49)))
    t.hAlign = "LEFT"
    story.append(t)
    story.append(Spacer(1, 4 * mm))

    totals = [
        [_cell("Subtotal", align="RIGHT"),
         _cell(f"₹ {subtotal:.2f}", align="RIGHT")],
        [_cell("Round-off", align="RIGHT"),
         _cell(f"₹ {round_off:+.2f}", align="RIGHT")],
        [_cell("Grand Total", align="RIGHT", bold=True, size=10),
         _cell(f"₹ {rounded:.2f}", align="RIGHT", bold=True, size=10)],
    ]
    tt = Table(totals, colWidths=[139 * mm, 26 * mm])
    tt.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("FONTNAME", (0, -1), (-1, -1), FONT_BOLD),
        ("FONTNAME", (0, 0), (-1, -2), FONT),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    story.append(tt)
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(
        '<font size="8" color="#555555">Generated by the ArthaMart '
        "Intelligent Retailing System — dual-regime pricing applied at "
        "order confirmation. Thank you for your business!</font>",
        styles["Normal"]))

    doc.build(story)
    return {"pdf_path": str(pdf_path), "total_amount": float(rounded),
            "line_count": len(lines)}


# ---------------------------------------------------------------------------
# procurement note
# ---------------------------------------------------------------------------

def generate_procurement_note_pdf(conn: sqlite3.Connection, order_id: int,
                                  out_dir: Path | None = None) -> str | None:
    """Procurement note for the NOT_AVAILABLE lines of an order.
    Returns None when nothing was unavailable."""
    _order(conn, order_id)
    lines = _unavailable_lines(conn, order_id)
    if not lines:
        return None
    out_dir = Path(out_dir) if out_dir else PROCUREMENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"procurement_{order_id}.pdf"

    eoq_map = {r["product_id"]: r["qty"] for r in conn.execute(
        "SELECT product_id, qty FROM purchase_orders "
        "WHERE status = 'RECOMMENDED'")}

    rows = [_header_row(["Product", "Category", "Required", "Unit",
                         "Suggested supplier", "Ref. EOQ"])]
    for ln in lines:
        rows.append([
            _cell(ln["name"]),
            _cell(ln["category"] or "-"),
            _cell(f"{ln['ordered_qty']:g}", align="RIGHT"),
            _cell(ln["unit"] or "pcs", align="CENTER"),
            _cell(SUPPLIERS.get(ln["category"], DEFAULT_SUPPLIER)),
            _cell(str(eoq_map.get(ln["product_id"], "-")), align="RIGHT"),
        ])

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            topMargin=15 * mm, bottomMargin=15 * mm)
    story: list = []
    _header_block(story, f"PROCUREMENT NOTE — ORDER {order_id}")

    styles = _styles()
    story.append(Paragraph(
        "The following items could not be fulfilled from current stock. "
        "Quantities below are the unmet customer demand; reference EOQ "
        "comes from the Phase-5 reorder planner.", styles["Normal"]))
    story.append(Spacer(1, 4 * mm))

    t = Table(rows, colWidths=[44 * mm, 20 * mm, 16 * mm, 12 * mm,
                               62 * mm, 18 * mm])
    t.setStyle(_table_style(colors.Color(0.55, 0.14, 0.14)))
    story.append(t)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        '<font size="8" color="#555555">Generated by the ArthaMart '
        "Intelligent Retailing System.</font>", styles["Normal"]))

    doc.build(story)
    return str(pdf_path)


# ---------------------------------------------------------------------------
# combined entry point + persistence
# ---------------------------------------------------------------------------

def generate_for_order(conn: sqlite3.Connection, order_id: int,
                       save_record: bool = True) -> dict:
    """Invoice + procurement note for one order; record in DB."""
    inv = generate_invoice_pdf(conn, order_id)
    proc = generate_procurement_note_pdf(conn, order_id)

    if save_record:
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("DELETE FROM invoices_generated WHERE order_id = ?",
                     (order_id,))
        conn.execute(
            "INSERT INTO invoices_generated "
            "(order_id, pdf_path, invoice_date, total_amount, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (order_id, inv["pdf_path"], _dt.date.today().isoformat(),
             inv["total_amount"], now))
        conn.commit()

    return {"invoice": inv["pdf_path"], "procurement_note": proc,
            "total_amount": inv["total_amount"],
            "line_count": inv["line_count"]}
