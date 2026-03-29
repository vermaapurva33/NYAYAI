import unicodedata
from typing import Dict


def postprocess_text(text: str) -> Dict:
    normalized = unicodedata.normalize("NFKC", text)

    confidence = 1.0 if normalized.strip() else 0.0

    return {
        "text": normalized,
        "confidence": confidence
    }
