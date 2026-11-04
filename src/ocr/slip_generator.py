"""
ArthaMart — Sample Order Slip Generator
-----------------------------------------
Renders realistic wholesale order-slip images (PIL) for OCR testing:

    slip_clean.png   — clean, upright, well-spaced
    slip_noisy.png   — rotated ~2.5°, salt-and-pepper noise, light blur
    slip_typo.png    — OCR-style typos (hyphen drops, char swaps)

Items are drawn from the real catalog so matcher evaluation is meaningful.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..config import ROOT

SLIP_DIR = ROOT / "assets" / "sample_slips"

# (rendered text on slip, quantity) — names deliberately close to catalog
CLEAN_ITEMS = [
    ("Biscuit Parle-G 800g", 4),
    ("Rice Basmati (loose)", 10),
    ("Tata Tea Premium 500g", 2),
    ("Toor Dal (loose)", 5),
    ("Coca-Cola 750ml", 6),
    ("Surf Excel Detergent 1kg", 3),
]

NOISY_ITEMS = [
    ("Fortune Sunflower Oil 1L", 8),
    ("Aashirvaad Atta 5kg", 2),
    ("Sugar (loose)", 12),
    ("Maggi Masala 70g (pack-12)", 1),
    ("Colgate MaxFresh 300g", 4),
]

# deliberate typos the matcher should still resolve
TYPO_ITEMS = [
    ("Coca Cola 750ml", 6),          # hyphen dropped
    ("Biscult Parle-G 800g", 3),     # i→l confusion
    ("Rise Basmati loose", 5),       # e→i, parens dropped
    ("Tata Tee Premium 500g", 2),    # a→e
    ("Surf Exel Detergent 1kg", 2),  # c dropped
]


def _find_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    candidates += glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)[:1]
    for c in candidates:
        if os.path.exists(c):
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def render_slip(items: list[tuple[str, int]],
                title: str = "ORDER SLIP",
                font_size: int = 38,
                line_gap: int = 20) -> Image.Image:
    """Draw a simple white slip with header + numbered item lines."""
    font = _find_font(font_size)
    header_font = _find_font(font_size + 6)

    pad_x, pad_y = 70, 55
    line_h = font_size + line_gap
    width = 1100
    height = pad_y * 2 + line_h * (len(items) + 3)

    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)

    draw.text((pad_x, pad_y), title, font=header_font, fill=0)
    y = pad_y + line_h * 2
    for i, (name, qty) in enumerate(items, 1):
        draw.text((pad_x, y), f"{i}. {name}", font=font, fill=0)
        # right-aligned quantity
        qtext = str(qty)
        bbox = draw.textbbox((0, 0), qtext, font=font)
        draw.text((width - pad_x - (bbox[2] - bbox[0]), y), qtext,
                  font=font, fill=0)
        y += line_h

    draw.line([(pad_x, pad_y + line_h), (width - pad_x, pad_y + line_h)],
              fill=0, width=2)
    return img


def add_noise(img: Image.Image, seed: int = 42) -> Image.Image:
    """Salt-and-pepper noise + mild blur, mimicking a phone photo."""
    rng = np.random.default_rng(seed)
    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, 9, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    # salt & pepper speckles
    salt = rng.random(arr.shape) < 0.003
    pepper = rng.random(arr.shape) < 0.003
    arr[salt] = 255
    arr[pepper] = 0
    out = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.4))
    return out


def rotate(img: Image.Image, angle: float) -> Image.Image:
    return img.rotate(angle, resample=Image.BICUBIC, expand=True,
                      fillcolor=255)


def generate_all(out_dir: Path | None = None) -> dict[str, Path]:
    """Create the three demo slips. Returns {name: path}."""
    out_dir = Path(out_dir) if out_dir else SLIP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    clean = render_slip(CLEAN_ITEMS, "ORDER SLIP — ARTHAMART")
    p = out_dir / "slip_clean.png"
    clean.save(p); paths["clean"] = p

    noisy = add_noise(render_slip(NOISY_ITEMS, "ORDER SLIP"))
    noisy = rotate(noisy, 2.5)
    p = out_dir / "slip_noisy.png"
    noisy.save(p); paths["noisy"] = p

    typo = render_slip(TYPO_ITEMS, "ORDER SLIP")
    # kept clean on purpose — this slip tests fuzzy matching, not denoising
    p = out_dir / "slip_typo.png"
    typo.save(p); paths["typo"] = p

    return paths


if __name__ == "__main__":
    created = generate_all()
    for name, p in created.items():
        print(f"  {name:6s} → {p}")
