from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


def extract_page_text(pdf_path: Path, page_no: int) -> Optional[str]:
    """
    Try to extract real text from a PDF page.

    Returns:
    - text (str) if a text layer exists
    - None if the page has no usable text

    This function is intentionally conservative.
    If we're not sure, we return None and let OCR handle it.
    """

    try:
        doc = fitz.Document(pdf_path) # works same as fitz.open()
    except Exception:
        # Broken or unreadable PDF
        return None

    if page_no < 0 or page_no >= doc.page_count:
        return None

    try:
        page = doc.load_page(page_no)
        text = page.get_text("text")
    except Exception:
        return None

    if not text:
        return None


    # Very small text blocks are usually headers, footers, or noise
    cleaned = text.strip()

    if len(cleaned) < 200 or len(cleaned.split()) < 30:
        return None


    return cleaned


    
