"""
ArthaMart — OCR Preprocessing Pipeline
---------------------------------------
Prepares scanned/photographed order slips for OCR:

    grayscale → upscale → denoise → binarize (Otsu) → deskew

All knobs come from config.yaml under `ocr.preprocessing`.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import get


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_image(src) -> np.ndarray:
    """Accept a file path / Path / ndarray and return a BGR ndarray."""
    if isinstance(src, np.ndarray):
        return src.copy()
    path = str(src)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def upscale_if_small(gray: np.ndarray, min_width: int = 1400) -> np.ndarray:
    """Small photos OCR poorly — upscale width to at least `min_width`."""
    h, w = gray.shape[:2]
    if w >= min_width:
        return gray
    scale = min_width / w
    return cv2.resize(gray, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)


def denoise(gray: np.ndarray, strength: int = 10) -> np.ndarray:
    """Median filter (speckle) + non-local means (good for scanned paper)."""
    out = cv2.medianBlur(gray, 3)
    if strength <= 0:
        return out
    return cv2.fastNlMeansDenoising(out, None, h=strength,
                                    templateWindowSize=7, searchWindowSize=21)


def binarize_otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold — robust to uneven slip lighting."""
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Ensure dark text on light background (slips should be white bg).
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    return binary


def estimate_skew_angle(binary: np.ndarray, max_angle: float = 10.0) -> float:
    """Estimate text skew via minAreaRect over ink pixels."""
    coords = np.column_stack(np.where(binary < 128))
    if len(coords) < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2 returns angle in [-90, 0); convert to a small correction angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) > max_angle:
        return 0.0
    return angle


def deskew(binary: np.ndarray, angle: float | None = None) -> tuple[np.ndarray, float]:
    """Rotate to correct skew. Returns (image, angle_used)."""
    if angle is None:
        angle = estimate_skew_angle(binary)
    if abs(angle) < 0.3:                       # not worth rotating
        return binary, 0.0
    h, w = binary.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, m, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def preprocess(src, save_path: str | Path | None = None) -> dict:
    """
    Full preprocessing pipeline.

    Returns dict with keys:
        image   — preprocessed ndarray (uint8, 0/255)
        angle   — deskew angle applied (degrees)
        shape   — (h, w) of the result
    """
    denoise_strength = get("ocr.preprocessing.denoise_strength", 10)
    use_otsu = get("ocr.preprocessing.otsu", True)
    use_deskew = get("ocr.preprocessing.deskew", True)

    img = load_image(src)
    gray = to_grayscale(img)
    gray = upscale_if_small(gray)
    gray = denoise(gray, denoise_strength)
    binary = binarize_otsu(gray) if use_otsu else gray

    angle = 0.0
    if use_deskew:
        binary, angle = deskew(binary)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), binary)

    return {"image": binary, "angle": angle, "shape": binary.shape[:2]}
