"""
ArthaMart — Fuzzy Product Matcher
----------------------------------
Links OCR-extracted product names to the catalog using rapidfuzz.

Match status (threshold from config: ocr.similarity_threshold = 72):
    MATCHED    score >= threshold           → auto-accept
    REVIEW     review_floor <= score < thr  → human confirms in Order Queue
    UNMATCHED  below review floor           → manual entry

Normalization handles common OCR/typo variation:
    "Coca Cola 750ml" ≈ "Coca-Cola 750ml"   (hyphen/space)
    "Biscult Parle-G" ≈ "Biscuit Parle-G"   (char-level fuzz absorbs it)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from ..config import get, DB_PATH


MATCHED = "MATCHED"
REVIEW = "REVIEW"
UNMATCHED = "UNMATCHED"


def normalize_name(text: str) -> str:
    """Lowercase, unify separators, drop punctuation, collapse spaces."""
    t = text.lower()
    t = t.replace("-", " ").replace("/", " ").replace(".", " ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass
class CatalogProduct:
    product_id: int
    name: str
    regime: str          # MRP | BULK
    unit: str
    normalized: str


@dataclass
class MatchResult:
    query_raw: str
    quantity: float
    product_id: int | None
    matched_name: str | None
    score: float
    status: str

    @property
    def ok(self) -> bool:
        return self.status == MATCHED


# ---------------------------------------------------------------------------
# catalog loading
# ---------------------------------------------------------------------------

def load_catalog(conn=None) -> list[CatalogProduct]:
    """Load active products from SQLite (pass a connection to reuse)."""
    close = False
    if conn is None:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        close = True
    try:
        rows = conn.execute(
            "SELECT product_id, name, regime, unit FROM products "
            "WHERE is_active = 1 ORDER BY product_id"
        ).fetchall()
    finally:
        if close:
            conn.close()
    return [CatalogProduct(int(r[0]), r[1], r[2], r[3], normalize_name(r[1]))
            for r in rows]


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------

def _score(query_norm: str, candidate_norm: str) -> float:
    return max(
        fuzz.token_set_ratio(query_norm, candidate_norm),
        fuzz.token_sort_ratio(query_norm, candidate_norm),
    )


def match_name(query: str, catalog: list[CatalogProduct],
               threshold: float | None = None,
               review_floor: float | None = None) -> MatchResult:
    """Find the best catalog product for an OCR-extracted name."""
    if threshold is None:
        threshold = float(get("ocr.similarity_threshold", 72))
    if review_floor is None:
        review_floor = float(get("ocr.review_floor", 55))

    q = normalize_name(query)
    if not q or not catalog:
        return MatchResult(query, 0.0, None, None, 0.0, UNMATCHED)

    best, best_score = None, 0.0
    for prod in catalog:
        s = _score(q, prod.normalized)
        if s > best_score:
            best, best_score = prod, s

    if best is None or best_score < review_floor:
        return MatchResult(query, 0.0, None, None, best_score, UNMATCHED)

    status = MATCHED if best_score >= threshold else REVIEW
    return MatchResult(query, 0.0, best.product_id, best.name,
                       best_score, status)


def top_candidates(query: str, catalog: list[CatalogProduct],
                   k: int = 3) -> list[tuple[CatalogProduct, float]]:
    """Top-k catalog candidates for a name — used by the review queue."""
    q = normalize_name(query)
    if not q:
        return []
    scored = [(p, _score(q, p.normalized)) for p in catalog]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


def match_lines(parsed_items, catalog: list[CatalogProduct],
                threshold: float | None = None) -> list[MatchResult]:
    """Match a list of parser.ParsedLine items. Quantity is carried through."""
    out = []
    for item in parsed_items:
        r = match_name(item.name, catalog, threshold=threshold)
        r.quantity = item.quantity
        r.query_raw = item.raw
        out.append(r)
    return out
