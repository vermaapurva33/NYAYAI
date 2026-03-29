"""
word_extractor.py — Unified word + coordinate extractor for NyayAI.

Two paths:
  - Digital PDF  (has text layer) → PyMuPDF  → exact bounding boxes
  - Scanned PDF  (image only)     → Surya OCR → estimated word boxes

Both paths produce the same canonical token format:
    {word, page, x0, y0, x1, y1, source, confidence}

Teaching note:
    PyMuPDF (fitz) reads the PDF's internal structure directly — no image
    processing. It's 100x faster than OCR and gives exact coordinates.
    Surya is a state-of-the-art open-source OCR model (2024) that works
    at line level. We split lines into words and estimate x positions
    proportionally from character count.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

# Threshold: if fewer than this many words found via text layer,
# the PDF is likely scanned (or the text layer is broken)
_DIGITAL_MIN_WORDS_PER_PAGE = 10


# ─────────────────────────────────────────────────────────────────────────────
# Digital path (PyMuPDF)
# ─────────────────────────────────────────────────────────────────────────────

def extract_words_digital(pdf_path: Path) -> List[Dict]:
    """
    Extract word tokens with exact bounding boxes from a digital PDF.

    Returns:
        List of dicts: {word, page, x0, y0, x1, y1, source, confidence}
    """
    doc = fitz.open(str(pdf_path))
    tokens: List[Dict] = []

    for page_no in range(len(doc)):
        page = doc.load_page(page_no)

        # get_text("words") returns tuples:
        # (x0, y0, x1, y1, word, block_no, line_no, word_no)
        # sort=True → reading order (top→bottom, left→right)
        words = page.get_text("words", sort=True)

        for w in words:
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            word = word.strip()
            if word:
                tokens.append({
                    "word":       word,
                    "page":       page_no,
                    "x0":         x0,
                    "y0":         y0,
                    "x1":         x1,
                    "y1":         y1,
                    "source":     "digital",
                    "confidence": 1.0,  # exact — not OCR
                })

    doc.close()
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Scanned path (Surya OCR)
# ─────────────────────────────────────────────────────────────────────────────

def _load_surya_models():
    """
    Load Surya 0.5 OCR models.

    Requires transformers>=4.41,<4.45 — Surya 0.5 was designed for this range.
    The detection loader stayed in model.py in Surya 0.5 (not processor.py).
    """
    from surya.model.detection.model import load_model as load_det_model
    # Surya 0.5: load_processor for detection is in model.py (not processor.py)
    from surya.model.detection.model import load_processor as load_det_proc
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_proc

    det_model     = load_det_model()
    det_processor = load_det_proc()
    rec_model     = load_rec_model()
    rec_processor = load_rec_proc()

    return det_model, det_processor, rec_model, rec_processor





def extract_words_scanned(pdf_path: Path) -> List[Dict]:
    """
    Extract word tokens from a scanned PDF using Surya OCR.

    Surya gives line-level bounding boxes. We split each line into words
    and estimate per-word x positions proportionally based on character count.

    Teaching note:
        Why 300 DPI? OCR models are trained on high-res images. Rendering
        at 72 DPI (PDF default) makes characters too small. We render at
        300 DPI, run OCR, then scale coordinates back to PDF space (/4.167).
    """
    try:
        from surya.ocr import run_ocr as surya_ocr
        from PIL import Image
        det_model, det_processor, rec_model, rec_processor = _load_surya_models()
    except ImportError as _e:
        raise ImportError(
            f"surya-ocr import failed: {_e}. "
            "Reinstall with: pip install surya-ocr>=0.4,<0.6"
        ) from _e

    doc = fitz.open(str(pdf_path))
    tokens: List[Dict] = []

    SCALE = 300 / 72  # DPI scale factor: PDF coords → image pixels

    for page_no in range(len(doc)):
        page = doc.load_page(page_no)

        # Render page at 300 DPI for best OCR accuracy
        mat = fitz.Matrix(SCALE, SCALE)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        predictions = surya_ocr(
            [img], [["en"]], det_model, det_processor, rec_model, rec_processor
        )

        for line in predictions[0].text_lines:
            line_text = line.text.strip()
            words_in_line = line_text.split()
            if not words_in_line:
                continue

            # line.bbox = [x0, y0, x1, y1] in image pixel coords
            img_x0, img_y0, img_x1, img_y1 = line.bbox
            line_width = img_x1 - img_x0

            # Estimate character width from full line text
            char_width = line_width / max(len(line_text), 1)

            cursor_x = img_x0
            for word in words_in_line:
                word_pixel_width = len(word) * char_width

                tokens.append({
                    "word":       word,
                    "page":       page_no,
                    # Scale back from image pixels to PDF coordinate space
                    "x0":         cursor_x / SCALE,
                    "y0":         img_y0 / SCALE,
                    "x1":         (cursor_x + word_pixel_width) / SCALE,
                    "y1":         img_y1 / SCALE,
                    "source":     "ocr",
                    "confidence": getattr(line, "confidence", 1.0),
                })

                # Advance cursor: word width + 1 space character
                cursor_x += word_pixel_width + char_width

    doc.close()
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Detection: is this PDF digital or scanned?
# ─────────────────────────────────────────────────────────────────────────────

def is_scanned(pdf_path: Path, sample_pages: int = 3) -> bool:
    """
    Detect whether a PDF is scanned (image-only, no text layer).

    Checks the first few pages for selectable text. If fewer than
    _DIGITAL_MIN_WORDS_PER_PAGE words are found on average, we treat
    the PDF as scanned and route to OCR.
    """
    doc = fitz.open(str(pdf_path))
    page_count = min(sample_pages, len(doc))

    total_words = sum(
        len(doc.load_page(i).get_text("words"))
        for i in range(page_count)
    )
    doc.close()

    avg_words = total_words / max(page_count, 1)
    return avg_words < _DIGITAL_MIN_WORDS_PER_PAGE


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────────────────────────────────────

def extract_words(pdf_path: Path) -> List[Dict]:
    """
    Unified entry point: automatically chooses digital or OCR path.

    If the PDF is scanned but Surya OCR fails (import error, model download
    issue, etc.), gracefully falls back to PyMuPDF and logs a warning.
    Users always get a result — even if coordinates are less precise.

    Usage:
        from src.ocr.word_extractor import extract_words
        tokens = extract_words(Path("data/raw/judgment.pdf"))
    """
    pdf_path = Path(pdf_path)

    if not is_scanned(pdf_path):
        print(f"[INFO] Detected digital PDF — using PyMuPDF: {pdf_path.name}")
        return extract_words_digital(pdf_path)

    # Scanned — try Surya first, fall back to PyMuPDF if it fails
    print(f"[INFO] Detected scanned PDF — attempting Surya OCR: {pdf_path.name}")
    try:
        return extract_words_scanned(pdf_path)
    except ImportError as e:
        print(f"[WARN] Surya OCR unavailable ({e}). Falling back to PyMuPDF.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[WARN] Surya OCR failed ({type(e).__name__}: {e}). Falling back to PyMuPDF.")

    # Fallback: PyMuPDF on scanned PDF
    # For image-only PDFs this will return fewer words, but the
    # error detector will still run on whatever text it finds.
    print(f"[INFO] Running PyMuPDF fallback on scanned PDF: {pdf_path.name}")
    tokens = extract_words_digital(pdf_path)
    if not tokens:
        print("[WARN] No text found via PyMuPDF fallback — PDF may be purely image-based.")
    return tokens
