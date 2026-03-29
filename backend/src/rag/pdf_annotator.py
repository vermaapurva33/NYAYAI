"""
pdf_annotator.py — Annotate a PDF with error highlights and correction popups.

Takes a list of error dicts (from error_detector.py) and draws colored
bounding boxes on the original PDF using PyMuPDF.

Color scheme:
    🟡 Yellow  — spelling errors     (SPELL)
    🟠 Orange  — grammar errors      (GRAM)
    🔴 Red     — semantic errors      (SEM)  ← most critical for legal docs

Teaching note:
    PyMuPDF (fitz) can draw shapes, add highlights, and attach text annotations
    (popup comments) directly into a PDF's annotation layer. These annotations
    are visible in any PDF viewer (Adobe, Preview, Chrome, Evince).
    We save with garbage=4 and deflate=True to compress and clean up the file.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict

# Error type → (R, G, B) in 0.0–1.0 range (PyMuPDF uses normalized floats)
_ERROR_COLORS = {
    "SPELL": (1.0, 0.80, 0.0),   # Yellow
    "GRAM":  (1.0, 0.47, 0.0),   # Orange
    "SEM":   (0.87, 0.12, 0.12), # Red
}

_DEFAULT_COLOR = (0.5, 0.0, 0.8)  # Purple fallback for unknown types


def annotate_pdf(
    original_pdf: Path,
    errors: List[Dict],
    output_pdf: Path,
) -> Path:
    """
    Draw bounding boxes and popup comments on the PDF for each detected error.

    Args:
        original_pdf:  Source PDF file
        errors:        List of error dicts from ErrorDetector.detect()
                       Each must have: page, x0, y0, x1, y1, word, error_type, confidence
        output_pdf:    Where to save the annotated PDF

    Returns:
        Path to the annotated PDF file

    Output PDF will have:
        - Colored rectangle border around each error word
        - Semi-transparent highlight in the same color
        - Text annotation popup (click the pin icon in the PDF viewer to read)
    """
    doc = fitz.open(str(original_pdf))

    # Group errors by page for efficient processing
    errors_by_page: Dict[int, List[Dict]] = {}
    for err in errors:
        page_no = err["page"]
        errors_by_page.setdefault(page_no, []).append(err)

    for page_no, page_errors in errors_by_page.items():
        if page_no >= len(doc):
            continue

        page = doc.load_page(page_no)

        for error in page_errors:
            x0 = error.get("x0", 0)
            y0 = error.get("y0", 0)
            x1 = error.get("x1", x0 + 20)
            y1 = error.get("y1", y0 + 12)
            error_type = error.get("error_type", "UNKNOWN")
            color = _ERROR_COLORS.get(error_type, _DEFAULT_COLOR)

            rect = fitz.Rect(x0, y0, x1, y1)

            # ── 1. Colored rectangle border ─────────────────────────────────
            # draw_rect draws the outline; fill=None → transparent interior
            page.draw_rect(rect, color=color, width=1.5)

            # ── 2. Semi-transparent highlight ───────────────────────────────
            # Highlights appear as a colored overlay on the text
            try:
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=list(color))
                highlight.update()
            except Exception:
                pass  # Some PDF geometries don't support highlights

            # ── 3. Text annotation (popup comment) ─────────────────────────
            # This creates a small icon; clicking it shows the comment in the viewer
            confidence_pct = f"{error.get('confidence', 0.0):.0%}"
            comment = (
                f"⚠ NyayAI Error Detected\n"
                f"Type:       {error_type}\n"
                f"Word:       \"{error.get('word', '')}\"\n"
                f"Confidence: {confidence_pct}"
            )

            # Place the annotation icon just to the left of the error word
            icon_point = fitz.Point(max(x0 - 8, 0), y0)
            annot = page.add_text_annot(icon_point, comment, icon="Note")
            annot.set_colors(stroke=list(color))
            annot.update()

    # Save with compression: garbage=4 removes unused objects, deflate=True compresses
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_pdf), garbage=4, deflate=True)
    doc.close()

    print(f"[OK] Annotated PDF with {len(errors)} errors → {output_pdf}")
    return output_pdf


def count_errors_by_type(errors: List[Dict]) -> Dict[str, int]:
    """Summary count of errors by type — useful for the API response."""
    counts: Dict[str, int] = {}
    for err in errors:
        t = err.get("error_type", "UNKNOWN")
        counts[t] = counts.get(t, 0) + 1
    return counts
