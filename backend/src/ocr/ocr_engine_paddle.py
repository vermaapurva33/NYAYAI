from pathlib import Path
from paddleocr import PaddleOCR
from typing import Optional, List, Dict
from src.common.config import USE_GPU, LANGUAGES

_engine: Optional[PaddleOCR] = None


def _get_engine() -> PaddleOCR:
    global _engine

    if _engine is not None:
        return _engine

    lang = 'en'  # Simplify for now; extend later as needed

    # Try GPU first
    if USE_GPU:
        try:
            _engine = PaddleOCR(
                use_angle_cls=True,
                use_gpu=True,
                lang=lang,
            )
            print("PaddleOCR running on GPU during ocr")
            return _engine
        except Exception as e:
            print(f"[WARN] GPU OCR init failed, falling back to CPU: {e}")

    # Fallback to CPU
    _engine = PaddleOCR(
        use_angle_cls=True,
        use_gpu=False,
        lang=lang,
    )
    print("[INFO] PaddleOCR running on CPU")
    return _engine


def run_ocr(image_path: Path, regions=None) -> List[Dict]:
    """
    Run OCR on a single image.
    Returns list of blocks with text, confidence, and bounding box coordinates.

    PaddleOCR result format: result[page_index][detection_index]
    Each detection: [quad_box, (text, confidence)]
    """
    engine = _get_engine()
    result = engine.ocr(str(image_path), cls=True)

    blocks: List[Dict] = []

    if not result:
        return blocks

    # Outer loop: pages (PaddleOCR returns a list of pages even for single images)
    for page_result in result:
        if not page_result:
            continue
        # Inner loop: individual text detections on this page
        for item in page_result:
            try:
                box, (text, confidence) = item

                # Normalize in case of wrapped tuples/lists
                if isinstance(text, (tuple, list)):
                    text = text[0]
                if isinstance(confidence, (tuple, list)):
                    confidence = confidence[0]

                text = str(text).strip()
                confidence = float(confidence)

                if not text:
                    continue

                # ── CONFIDENCE GATING ─────────────────────────────────────────
                # Drop low-confidence OCR results immediately — garbage text
                # pollutes everything downstream. Threshold 0.6 is conservative.
                if confidence < 0.6:
                    continue

                # ── BBOX EXTRACTION ───────────────────────────────────────────
                # box is a quad: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                # We convert to axis-aligned rectangle (x0,y0,x1,y1)
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]

                blocks.append({
                    "text":       text,
                    "confidence": confidence,
                    "box":        box,          # original quad
                    "x0":         min(xs),      # leftmost x
                    "y0":         min(ys),      # topmost y
                    "x1":         max(xs),      # rightmost x
                    "y1":         max(ys),      # bottommost y
                })

            except Exception:
                continue

    # Sort blocks top-to-bottom then left-to-right (reading order)
    # Bucket rows with 20px tolerance (handles slight vertical misalignment)
    blocks.sort(key=lambda b: (round(b["y0"] / 20) * 20, b["x0"]))

    return blocks



def run_ocr(image_path: Path, regions=None) -> List[Dict]:
    """
    Run OCR on a single image.
    This function assumes the caller has already decided
    that OCR is actually needed.
    """

    engine = _get_engine()
    # print("running ocr")

    result = engine.ocr(str(image_path), cls=True)

    blocks: List[Dict] = []

    if not result:
        return blocks

    # PaddleOCR returns: [ [ [box, (text, conf)], ... ] ]
    for page in result:
        for item in page:
            try:
                box, (text, confidence) = item

                # Normalize shapes
                if isinstance(text, (tuple, list)):
                    text = text[0]

                if isinstance(confidence, (tuple, list)):
                    confidence = confidence[0]

                text = str(text).strip()
                confidence = float(confidence)

                if not text:
                    continue

                blocks.append({
                    "text": text,
                    "confidence": confidence
                })

            except Exception:
                continue

    return blocks

