"""
ArthaMart — Order Processing status model (Phase 3).
=====================================================

Order lifecycle
---------------
    DRAFT            created from OCR slip; lines may still be unresolved
    PENDING_REVIEW   ≥1 line needs a human (REVIEW / UNMATCHED)
    CONFIRMED        every line has a product_id; availability scored
    FULFILLED        stock deducted; final
    CANCELLED        terminal

Line lifecycle (single `order_line_items.status` column)
--------------------------------------------------------
    MATCHED / REVIEW / UNMATCHED   set by OCR fuzzy matcher (Phase 2)
    CONFIRMED                      human/queue resolved the product_id
    READY                          stock >= ordered qty
    PARTIAL                        0 < stock < ordered qty
    NOT_AVAILABLE                  stock == 0
    FULFILLED                      available qty shipped & stock deducted

The READY / PARTIAL / NOT_AVAILABLE trio is the roadmap's availability
classification; it overwrites the earlier status once availability runs.
"""
from __future__ import annotations

from enum import Enum


class OrderStatus(str, Enum):
    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    CONFIRMED = "CONFIRMED"
    FULFILLED = "FULFILLED"
    CANCELLED = "CANCELLED"


class LineStatus(str, Enum):
    # from OCR matcher
    MATCHED = "MATCHED"
    REVIEW = "REVIEW"
    UNMATCHED = "UNMATCHED"
    # after resolution
    CONFIRMED = "CONFIRMED"
    # after availability classification
    READY = "READY"
    PARTIAL = "PARTIAL"
    NOT_AVAILABLE = "NOT_AVAILABLE"


# Line statuses that still need a human decision in the review queue.
UNRESOLVED = {LineStatus.REVIEW, LineStatus.UNMATCHED}

# Line statuses that carry a valid product_id (resolvable / orderable).
HAS_PRODUCT = {LineStatus.MATCHED, LineStatus.CONFIRMED, LineStatus.READY,
               LineStatus.PARTIAL, LineStatus.NOT_AVAILABLE}

# Terminal order states (no further transitions allowed).
TERMINAL_ORDERS = {OrderStatus.FULFILLED, OrderStatus.CANCELLED}
