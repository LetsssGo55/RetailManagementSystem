"""
ArthaMart — End-to-End OCR Pipeline
-------------------------------------
    image → preprocess → Tesseract → parse → fuzzy match → order draft

`process_slip()` is the single entry point used by the CLI demo,
the Order Queue (Phase 3) and the Streamlit dashboard (Phase 10).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import engine, matcher, parser
from .preprocess import preprocess


@dataclass
class SlipResult:
    image_path: str | None
    raw_text: str
    parse: parser.ParseResult
    matches: list[matcher.MatchResult]
    deskew_angle: float = 0.0
    matched_count: int = 0
    review_count: int = 0
    unmatched_count: int = 0

    def summarise(self) -> dict:
        return {
            "lines": len(self.parse.lines),
            "items": len(self.parse.items),
            "matched": self.matched_count,
            "review": self.review_count,
            "unmatched": self.unmatched_count,
            "deskew_angle": round(self.deskew_angle, 2),
        }


def process_slip(image_src, conn=None,
                 save_preprocessed: str | Path | None = None) -> SlipResult:
    """
    Run the complete OCR intake pipeline on one slip image.

    Parameters
    ----------
    image_src : path / ndarray
    conn : optional SQLite connection (reused catalog load)
    save_preprocessed : optional path to dump the binarised image
    """
    prep = preprocess(image_src, save_path=save_preprocessed)
    text = engine.ocr_text(prep["image"])
    parse = parser.parse_slip_text(text)

    catalog = matcher.load_catalog(conn)
    matches = matcher.match_lines(parse.items, catalog)

    result = SlipResult(
        image_path=str(image_src) if not hasattr(image_src, "shape") else None,
        raw_text=text,
        parse=parse,
        matches=matches,
        deskew_angle=prep["angle"],
        matched_count=sum(m.status == matcher.MATCHED for m in matches),
        review_count=sum(m.status == matcher.REVIEW for m in matches),
        unmatched_count=sum(m.status == matcher.UNMATCHED for m in matches),
    )
    return result
