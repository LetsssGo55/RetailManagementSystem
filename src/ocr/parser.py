"""
ArthaMart — Order Slip Parser
------------------------------
Converts raw OCR text into structured line items:

    "Biscuit Parle-G 800g 4"  →  name="Biscuit Parle-G 800g", quantity=4

Quantity rules (per synopsis): the quantity is the LAST pure number on the
line. Numbers glued to units (800g, 1L, 750ml, pack-12) belong to the
product name, so they never collide. If a line carries a price column
("Rice loose 10 45 450"), the trailing numeric run is treated as
qty / rate / amount and the FIRST number of that run is the quantity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Line-start tokens that indicate a header / non-item line
_HEADER_RE = re.compile(
    r"^(order\s*slip|arthamart|shop\b|store\b|date\b|time\b|total|sub\s*total|"
    r"grand\s*total|invoice|bill\b|receipt|item\b|qty\b|quantity\b|amount\b|"
    r"rate\b|price\b|sr\.?\s*no|s\.n\.|customer|phone|mobile|address|thanks|"
    r"visit\s*again|regards)",
    re.IGNORECASE,
)

_TRAILING_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)[.,;:]?\s*$")  # last pure number
_X_NOTATION_RE = re.compile(r"[xX×]\s*(\d+(?:\.\d+)?)\s*$")  # "... x4"
_LEADING_ENUM_RE = re.compile(r"^(\d{1,2}\s*[.):\-]\s+|[-•*]\s+)")


@dataclass
class ParsedLine:
    line_no: int
    raw: str
    name: str
    quantity: float
    qty_source: str            # 'trailing' | 'x-notation' | 'price-run' | 'default'
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class ParseResult:
    lines: list[ParsedLine] = field(default_factory=list)

    @property
    def items(self) -> list[ParsedLine]:
        return [ln for ln in self.lines if not ln.skipped]

    @property
    def skipped_lines(self) -> list[ParsedLine]:
        return [ln for ln in self.lines if ln.skipped]


def _clean_name(name: str) -> str:
    """Collapse whitespace and strip stray punctuation."""
    name = re.sub(r"\s+", " ", name).strip(" \t|_:;,.–—")
    return name


def parse_line(line: str, line_no: int) -> ParsedLine:
    """Parse one OCR line into a ParsedLine (skipped flag set if not an item)."""
    raw = line.strip()
    if not raw:
        return ParsedLine(line_no, raw, "", 0, "default",
                          skipped=True, skip_reason="empty")

    text = _LEADING_ENUM_RE.sub("", raw).strip()

    # header check on a separator-tolerant copy ("ORDER -SLIP" → "ORDER SLIP")
    header_probe = re.sub(r"[-–—_/]+", " ", text)
    header_probe = re.sub(r"\s+", " ", header_probe).strip()
    if _HEADER_RE.match(header_probe):
        return ParsedLine(line_no, raw, "", 0, "default",
                          skipped=True, skip_reason="header")

    if not re.search(r"[A-Za-z]", text):
        return ParsedLine(line_no, raw, "", 0, "default",
                          skipped=True, skip_reason="no-letters")

    # 1) explicit x-notation: "Parle-G x4"
    m = _X_NOTATION_RE.search(text)
    if m:
        qty = float(m.group(1))
        name = _clean_name(text[: m.start()])
        if name:
            return ParsedLine(line_no, raw, name, qty, "x-notation")

    # 2) trailing pure number(s)
    nums: list[str] = []
    rest = text
    while True:
        m = _TRAILING_NUM_RE.search(rest)
        if not m:
            break
        nums.insert(0, m.group(1))
        rest = rest[: m.start()].rstrip()
        # stop scanning once we hit text that isn't whitespace-separated numbers
        if rest and not re.search(r"\d$", rest):
            break

    if nums:
        qty = float(nums[0])
        name = _clean_name(rest)
        src = "price-run" if len(nums) >= 3 else "trailing"
    else:
        qty, name, src = 1.0, _clean_name(text), "default"

    if not name:
        return ParsedLine(line_no, raw, "", qty, src,
                          skipped=True, skip_reason="no-name")

    return ParsedLine(line_no, raw, name, qty, src)


def parse_slip_text(text: str) -> ParseResult:
    """Parse the full OCR output of one slip."""
    result = ParseResult()
    for i, line in enumerate(text.splitlines(), start=1):
        parsed = parse_line(line, i)
        # keep skipped lines too — dashboard shows them for manual review
        result.lines.append(parsed)
    return result
