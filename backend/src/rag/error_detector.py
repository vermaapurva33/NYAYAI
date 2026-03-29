"""
error_detector.py — Inference wrapper for NyayAI error detection model.

Uses a fine-tuned InLegalBERT token classifier to detect three error types:
    SPELL  — spelling mistakes (e.g. "imprissoned" → "imprisoned")
    GRAM   — grammatical errors (wrong tense, agreement)
    SEM    — semantic errors (wrong section number, wrong legal clause)

Teaching note on Token Classification:
    The model sees a window of ~30 words at a time. For each word (token),
    it outputs a label. We use IOB tagging (Inside-Outside-Beginning):
        O       = not an error
        B-SPELL = first token of a spelling error span
        I-SPELL = continuation of a spelling error (multi-token misspellings)
        B-SEM   = first token of a wrong section/clause reference

    This is identical to Named Entity Recognition (NER) — just instead of
    tagging "PERSON" and "ORG", we tag "SPELL" and "SEM".
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch

MODEL_PATH = "models/nyayai-error-detector"

# Error type → colored bounding box (RGB 0–255)
ERROR_DISPLAY = {
    "SPELL": {"color": (255, 200, 0),   "label": "Spelling Error"},
    "GRAM":  {"color": (255, 120, 0),   "label": "Grammar Error"},
    "SEM":   {"color": (220, 30,  30),  "label": "Semantic Error"},
}


class ErrorDetector:
    """
    Wraps the fine-tuned InLegalBERT token classifier for inference.

    Usage:
        detector = ErrorDetector()
        errors = detector.detect(word_tokens)
        # errors → [{"word": "imprissoned", "page": 0, "x0": ..., "error_type": "SPELL", ...}]
    """

    def __init__(self, model_path: str = MODEL_PATH):
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers torch"
            )

        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run scripts/train_error_detector.py first, or download a pre-trained checkpoint."
            )

        print(f"[INFO] Loading error detection model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        self.id2label = self.model.config.id2label
        print(f"[INFO] Model loaded. Labels: {list(self.id2label.values())}")

    def detect(
        self,
        word_tokens: List[Dict],
        window_size: int = 30,
        stride: int = 15,
        min_confidence: float = 0.5,
    ) -> List[Dict]:
        """
        Slide a context window over word tokens and run the classifier.

        Args:
            word_tokens:    Output of word_extractor.extract_words()
            window_size:    Number of words per inference window (default 30)
            stride:         Step size between windows (default 15 = 50% overlap)
            min_confidence: Minimum model confidence to report an error (default 0.7)

        Returns:
            List of error dicts. Each dict contains all fields from the input
            word token PLUS: error_type (str), confidence (float).

        Teaching note on sliding windows:
            Why stride < window_size? Because errors can appear at the boundary
            of a window. With 50% overlap, every word is seen in at least 2
            windows, so boundary errors are never missed.
        """
        errors: List[Dict] = []
        n = len(word_tokens)

        for start in range(0, n, stride):
            end = min(start + window_size, n)
            window = word_tokens[start:end]
            words = [t["word"] for t in window]

            enc = self.tokenizer(
                words,
                is_split_into_words=True,   # input is already tokenized (word list)
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )

            with torch.no_grad():
                logits = self.model(**enc).logits  # shape: [1, seq_len, num_labels]

            probs = torch.softmax(logits, dim=-1)[0]     # [seq_len, num_labels]
            pred_ids = torch.argmax(probs, dim=-1)       # [seq_len]

            # word_ids maps each subword token back to the original word index
            word_ids = enc.word_ids(batch_index=0)
            seen_word_ids = set()

            for token_idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue  # [CLS], [SEP], [PAD] tokens

                # Only label the FIRST subword of each word (skip continuations)
                if word_id in seen_word_ids:
                    continue
                seen_word_ids.add(word_id)

                label = self.id2label[pred_ids[token_idx].item()]
                if label == "O":
                    continue  # not an error

                confidence = probs[token_idx][pred_ids[token_idx]].item()
                if confidence < min_confidence:
                    continue  # below threshold — skip uncertain predictions

                # Strip IOB prefix: "B-SPELL" → "SPELL"
                error_type = label.replace("B-", "").replace("I-", "")

                errors.append({
                    **window[word_id],          # word, page, x0, y0, x1, y1, source
                    "error_type":  error_type,
                    "confidence":  confidence,
                })

        # Deduplicate: overlapping windows may flag the same word twice
        # Keep the highest-confidence prediction for each (page, position) pair
        seen: Dict[tuple, int] = {}  # key → index in errors
        unique: List[Dict] = []

        for err in errors:
            key = (err["page"], round(err["x0"]), round(err["y0"]))
            if key in seen:
                # Keep higher confidence version
                existing_idx = seen[key]
                if err["confidence"] > unique[existing_idx]["confidence"]:
                    unique[existing_idx] = err
            else:
                seen[key] = len(unique)
                unique.append(err)

        # Sort errors by page then position for clean output
        unique.sort(key=lambda e: (e["page"], e.get("y0", 0), e.get("x0", 0)))

        print(f"[INFO] Found {len(unique)} errors in {n} word tokens")
        return unique


def detect_errors_in_pdf(
    pdf_path: Path,
    model_path: str = MODEL_PATH,
    output_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Convenience function: extract words → detect errors → optionally annotate PDF.

    Args:
        pdf_path:    Input PDF path
        model_path:  Path to fine-tuned model directory
        output_path: If provided, write annotated PDF here

    Returns:
        List of error dicts (with word + coordinates + error_type)
    """
    from src.ocr.word_extractor import extract_words

    tokens = extract_words(pdf_path)
    if not tokens:
        print("[WARN] No words extracted from PDF")
        return []

    detector = ErrorDetector(model_path)
    errors = detector.detect(tokens)

    if output_path and errors:
        from src.rag.pdf_annotator import annotate_pdf
        annotate_pdf(pdf_path, errors, output_path)
        print(f"[INFO] Annotated PDF written to: {output_path}")

    return errors
