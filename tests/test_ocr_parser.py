"""Phase 2 — parser unit tests (no OCR binary needed)."""
import pytest

from src.ocr.parser import parse_line, parse_slip_text


def test_quantity_is_last_number():
    r = parse_line("Biscuit Parle-G 800g 4", 1)
    assert not r.skipped
    assert r.name == "Biscuit Parle-G 800g"
    assert r.quantity == 4


def test_unit_suffix_stays_in_name():
    r = parse_line("Fortune Sunflower Oil 1L 8", 2)
    assert r.name == "Fortune Sunflower Oil 1L"
    assert r.quantity == 8

    r2 = parse_line("Coca-Cola 750ml 6", 3)
    assert r2.name == "Coca-Cola 750ml"
    assert r2.quantity == 6


def test_pack_size_not_taken_as_quantity():
    r = parse_line("Maggi Masala 70g (pack-12) 1", 4)
    assert r.quantity == 1
    assert "pack-12" in r.name


def test_price_run_takes_first_as_quantity():
    # qty rate amount pattern
    r = parse_line("Rice Basmati (loose) 10 92 920", 5)
    assert r.quantity == 10
    assert r.name == "Rice Basmati (loose)"


def test_x_notation():
    r = parse_line("Tata Tea Premium 500g x3", 6)
    assert r.quantity == 3
    assert r.qty_source == "x-notation"


def test_missing_quantity_defaults_to_one():
    r = parse_line("Sugar (loose)", 7)
    assert r.quantity == 1
    assert r.qty_source == "default"


def test_leading_enumeration_stripped():
    r = parse_line("2. Toor Dal (loose) 5", 8)
    assert r.name == "Toor Dal (loose)"
    assert r.quantity == 5


def test_decimal_quantity_for_bulk():
    r = parse_line("Onion (loose) 2.5", 9)
    assert r.quantity == 2.5


@pytest.mark.parametrize("line", [
    "ORDER SLIP — ARTHAMART",
    "DATE: 15-08-2026",
    "TOTAL 1234",
    "SR NO ITEM QTY",
    "",
])
def test_header_and_noise_lines_skipped(line):
    r = parse_line(line, 1)
    assert r.skipped


def test_full_slip_text():
    text = """
ORDER SLIP — ARTHAMART
DATE: 15-08-2026
1. Biscuit Parle-G 800g 4
2. Rice Basmati (loose) 10
3. Tata Tea Premium 500g 2
TOTAL 16
"""
    res = parse_slip_text(text)
    assert len(res.items) == 3
    assert [i.quantity for i in res.items] == [4, 10, 2]
    assert len(res.skipped_lines) >= 3
