"""
ArthaMart — OCR Engine
-----------------------
Thin wrapper around Tesseract (via pytesseract) with graceful degradation.

If the tesseract binary is missing, `ocr_text()` raises OCREngineError with
an actionable message instead of a raw stack trace, and `tesseract_available()`
lets callers check up-front. On Windows the common install location
(C:\\Program Files\\Tesseract-OCR) is auto-detected even without PATH setup.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import get


class OCREngineError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Windows auto-detection: point pytesseract at the standard install dir
# ---------------------------------------------------------------------------

def _autodetect_tesseract() -> None:
    if os.name != "nt":
        return
    try:
        import pytesseract
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            str(Path.home() / "AppData" / "Local" / "Programs" /
                "Tesseract-OCR" / "tesseract.exe"),
        ]
        for exe in candidates:
            if Path(exe).exists():
                pytesseract.pytesseract.tesseract_cmd = exe
                return
    except ImportError:
        pass


_autodetect_tesseract()


def _get_pytest_module():
    try:
        import pytesseract  # noqa: F401
        return pytesseract
    except ImportError as e:
        raise OCREngineError(
            "pytesseract is not installed. Run:  pip install pytesseract"
        ) from e


def tesseract_available() -> bool:
    """True if the tesseract binary can be found."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def install_hint() -> str:
    return ("Tesseract binary not found.\n"
            "  Debian/Ubuntu : sudo apt-get install tesseract-ocr\n"
            "  macOS (brew)  : brew install tesseract\n"
            "  Windows       : install from https://github.com/UB-Mannheim/tesseract/wiki")


def ocr_text(image: np.ndarray | str | Path) -> str:
    """
    Run OCR on a preprocessed image (ndarray) or image path.
    Returns raw extracted text.
    """
    if not tesseract_available():
        raise OCREngineError(install_hint())

    pytesseract = _get_pytest_module()
    tconfig = get("ocr.tesseract_config", "--psm 6")

    if isinstance(image, (str, Path)):
        import cv2
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OCREngineError(f"Cannot read image: {image}")

    try:
        text = pytesseract.image_to_string(image, config=tconfig)
    except pytesseract.TesseractError as e:
        raise OCREngineError(f"Tesseract failed: {e}") from e
    return text


def ocr_lines(image: np.ndarray | str | Path) -> list[str]:
    """OCR + split into non-empty stripped lines."""
    text = ocr_text(image)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]
