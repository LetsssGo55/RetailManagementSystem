"""
ArthaMart — PDF and image receipt intake.

PDF pages are rasterized with pypdfium2 and passed through the existing OCR,
parser and fuzzy catalogue matcher. Upload size/page limits protect the server.
"""
from __future__ import annotations

from io import BytesIO
import numpy as np

from . import pipeline
from .parser import ParseResult


class ReceiptError(ValueError):
    """Invalid or unsupported uploaded receipt."""


MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 8


def _process_images(images, conn, label: str):
    parsed, matches, angles = [], [], []
    for image in images:
        result = pipeline.process_slip(image, conn=conn)
        parsed.extend(result.parse.lines)
        matches.extend(result.matches)
        angles.append(result.deskew_angle)
    if not matches:
        raise ReceiptError("No product lines could be read from the uploaded receipt.")
    return pipeline.SlipResult(
        image_path=label,
        raw_text="\n".join(line.raw for line in parsed),
        parse=ParseResult(lines=parsed),
        matches=matches,
        deskew_angle=sum(angles) / len(angles) if angles else 0.0,
        matched_count=sum(m.status == "MATCHED" for m in matches),
        review_count=sum(m.status == "REVIEW" for m in matches),
        unmatched_count=sum(m.status == "UNMATCHED" for m in matches),
    )


def process_receipt(payload: bytes, filename: str, conn=None):
    """Decode an image or PDF and run the existing OCR + product matcher."""
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        raise ReceiptError("The uploaded receipt is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ReceiptError("Receipt exceeds the 15 MB upload limit.")
    data = bytes(payload)

    if data.startswith(b"%PDF-"):
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ReceiptError(
                "PDF OCR requires pypdfium2. Install project requirements and restart the app."
            ) from exc
        try:
            document = pdfium.PdfDocument(data)
            if len(document) == 0:
                raise ReceiptError("The PDF contains no pages.")
            if len(document) > MAX_PDF_PAGES:
                raise ReceiptError(f"PDF is limited to {MAX_PDF_PAGES} pages.")
            images = []
            for page in document:
                rgb = np.asarray(page.render(scale=2.0).to_pil().convert("RGB"))
                images.append(rgb[:, :, ::-1].copy())  # OpenCV expects BGR
            return _process_images(images, conn, filename or "receipt.pdf")
        except ReceiptError:
            raise
        except Exception as exc:
            raise ReceiptError(f"Could not read PDF receipt: {exc}") from exc

    try:
        from PIL import Image
        image = Image.open(BytesIO(data))
        image.verify()
        rgb = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
        return _process_images([rgb[:, :, ::-1].copy()], conn, filename or "receipt-image")
    except ReceiptError:
        raise
    except Exception as exc:
        raise ReceiptError("Upload a valid PNG, JPG, JPEG, TIFF, BMP or PDF receipt.") from exc
