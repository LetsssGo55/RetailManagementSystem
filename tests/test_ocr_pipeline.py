"""Phase 2 — end-to-end OCR pipeline test (skips without tesseract binary)."""
import pytest

from src.ocr import engine
from src.ocr.matcher import MATCHED
from src.ocr.pipeline import process_slip
from src.ocr.slip_generator import CLEAN_ITEMS, generate_all

pytestmark = pytest.mark.skipif(
    not engine.tesseract_available(),
    reason="tesseract binary not installed",
)


@pytest.fixture(scope="module")
def slips(tmp_path_factory):
    out = tmp_path_factory.mktemp("slips")
    return generate_all(out)


def test_clean_slip_end_to_end(slips):
    res = process_slip(slips["clean"])
    s = res.summarise()
    assert s["items"] >= len(CLEAN_ITEMS) - 1, \
        f"parser lost lines: {res.raw_text!r}"
    assert s["matched"] >= len(CLEAN_ITEMS) - 1, \
        f"too few matches: {[(m.query_raw, m.status) for m in res.matches]}"


def test_noisy_slip_deskewed_and_parsed(slips):
    res = process_slip(slips["noisy"])
    s = res.summarise()
    # deskew should detect the 2.5° rotation (within tolerance)
    assert s["deskew_angle"] != 0.0
    assert s["matched"] >= 3


def test_typo_slip_fuzzy_recovered(slips):
    res = process_slip(slips["typo"])
    s = res.summarise()
    # fuzzy matching should recover most typo'd names
    assert s["matched"] + s["review"] >= 3


def test_matched_results_carry_product_ids(slips):
    res = process_slip(slips["clean"])
    matched = [m for m in res.matches if m.status == MATCHED]
    assert matched
    assert all(m.product_id is not None and m.product_id > 0 for m in matched)
    assert all(m.quantity > 0 for m in matched)
