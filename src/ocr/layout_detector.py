from pathlib import Path
from typing import List, Dict, Optional

from paddleocr import PaddleOCR
from src.common.config import USE_GPU


# Detection-only PaddleOCR instance.
_detector: Optional[PaddleOCR] = None


def _get_detector() -> PaddleOCR:
    global _detector

    if _detector is None:
        _detector = PaddleOCR(
            use_angle_cls=False,
            use_gpu=USE_GPU,
            rec=False  # detection only, recognition disabled.
        )

    return _detector


def detect_layout(image_path: Path) -> List[Dict]:
    """
    Detect text regions on a page.
    Returns a list of bounding boxes.
    If no regions are found, returns an empty list.
    """

    detector = _get_detector()
    result = detector.ocr(str(image_path), cls=False)

    regions: List[Dict] = []

    if not result:
        return regions

    for item in result:
        box = item[0]
        regions.append({
            "bbox": box,
            "type": "text"
        })

    return regions
