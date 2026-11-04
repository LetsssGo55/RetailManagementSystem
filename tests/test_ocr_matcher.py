"""Phase 2 — fuzzy matcher unit tests (uses real catalog from DB)."""
import pytest

from src.ocr.matcher import (MATCHED, REVIEW, UNMATCHED, load_catalog,
                             match_name, normalize_name)


@pytest.fixture(scope="module")
def catalog():
    cat = load_catalog()
    assert len(cat) > 0, "catalog empty — run `python main.py setup` first"
    return cat


def test_exact_match(catalog):
    r = match_name("Biscuit Parle-G 800g", catalog)
    assert r.status == MATCHED
    assert r.score >= 95
    assert r.matched_name == "Biscuit Parle-G 800g"


def test_hyphen_space_variation(catalog):
    r = match_name("Coca Cola 750ml", catalog)
    assert r.status == MATCHED
    assert r.matched_name == "Coca-Cola 750ml"


def test_typo_absorbed(catalog):
    r = match_name("Biscult Parle-G 800g", catalog)
    assert r.status == MATCHED

    r2 = match_name("Surf Exel Detergent 1kg", catalog)
    assert r2.status == MATCHED


def test_case_insensitive(catalog):
    r = match_name("TATA TEA PREMIUM 500G", catalog)
    assert r.status == MATCHED


def test_garbage_is_unmatched(catalog):
    r = match_name("Xylophonic Widget Z9", catalog)
    assert r.status in (UNMATCHED, REVIEW)
    assert r.status != MATCHED


def test_normalize_name():
    assert normalize_name("Coca-Cola 750ml") == normalize_name("coca cola 750ml")
    assert normalize_name("  Aashirvaad  Atta   5kg ") == "aashirvaad atta 5kg"


def test_threshold_respected(catalog):
    # impossibly high threshold forces REVIEW
    r = match_name("Coca Cola 750ml", catalog, threshold=101)
    assert r.status == REVIEW
