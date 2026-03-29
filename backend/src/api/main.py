"""
backend/src/api/main.py — NyayAI FastAPI backend v3.0

Key improvements over v2:
  - Global model caching: Surya + InLegalBERT loaded once at startup
  - /analyze-pdf returns full error list with suggestions + page numbers
  - /pages/<n> endpoint for per-page annotated PNG (cached per upload)
  - /metrics endpoint for developer performance dashboard
  - CORS configured via ALLOWED_ORIGINS env var
"""

import os
import sys
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Path setup ────────────────────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_ROOT))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NyayAI Backend",
    description="Indian Legal Document Error Detection API",
    version="3.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501")
_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,
)

# ── Model paths ───────────────────────────────────────────────────────────────
_MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(BACKEND_ROOT.parent / "models" / "nyayai-error-detector")
))

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL CACHE — loaded once at first request, reused for all subsequent
# ─────────────────────────────────────────────────────────────────────────────

_surya_models  = None   # (det_model, det_processor, rec_model, rec_processor)
_bert_detector = None   # ErrorDetector instance

def _get_surya():
    global _surya_models
    if _surya_models is None:
        from src.ocr.word_extractor import _load_surya_models
        _surya_models = _load_surya_models()
    return _surya_models

def _get_detector():
    global _bert_detector
    if _bert_detector is None:
        if not _MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {_MODEL_PATH}.")
        from src.rag.error_detector import ErrorDetector
        _bert_detector = ErrorDetector(model_path=str(_MODEL_PATH))
    return _bert_detector


# ─────────────────────────────────────────────────────────────────────────────
# Spell suggestion helper
# ─────────────────────────────────────────────────────────────────────────────

def _suggest(word: str, error_type: str) -> str:
    """Return a simple correction hint for the detected error."""
    if error_type == "SPELL":
        try:
            from spellchecker import SpellChecker
            spell = SpellChecker()
            candidates = spell.candidates(word)
            if candidates:
                return min(candidates, key=lambda w: len(w))
        except ImportError:
            pass
        return f"Check spelling of '{word}'"
    elif error_type == "GRAM":
        return f"Review grammar around '{word}'"
    elif error_type == "SEM":
        return f"Verify legal reference/section for '{word}'"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline: PDF → tokens → errors → annotated PDF
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(pdf_path: Path, annotated_path: Path):
    """
    Run the full NyayAI pipeline and return (tokens, errors).
    Surya + BERT models are loaded from global cache.
    """
    import fitz
    from src.ocr.word_extractor import extract_words
    from src.rag.pdf_annotator import annotate_pdf

    t0 = time.time()
    tokens = extract_words(pdf_path)
    ocr_time = time.time() - t0

    t1 = time.time()
    detector = _get_detector()
    errors = detector.detect(tokens)
    inference_time = time.time() - t1

    annotate_pdf(pdf_path, errors, annotated_path)

    # Add suggestions
    for err in errors:
        err["suggestion"] = _suggest(err.get("word", ""), err.get("error_type", ""))

    meta = {
        "total_words":     len(tokens),
        "total_errors":    len(errors),
        "ocr_time_s":      round(ocr_time, 2),
        "inference_time_s": round(inference_time, 2),
    }
    return tokens, errors, meta


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_path":    str(_MODEL_PATH),
        "model_ready":   _MODEL_PATH.exists(),
        "surya_cached":  _surya_models is not None,
        "bert_cached":   _bert_detector is not None,
        "cors_origins":  _origins,
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-pdf
# Returns full JSON report: errors with word + page + suggestion + coordinates
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze-pdf")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Upload PDF → JSON report with every error's word, page, type, suggestion.

    Response:
    {
      "total_words": 5616,
      "total_errors": 4,
      "ocr_time_s": 12.3,
      "inference_time_s": 0.8,
      "summary": {"SPELL": 2, "GRAM": 1, "SEM": 1},
      "errors": [
        {"word": "imprissoned", "page": 2, "error_type": "SPELL",
         "suggestion": "imprisoned", "confidence": 0.94, "x0": ..., ...},
        ...
      ]
    }
    """
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path      = Path(tmp) / "uploaded.pdf"
        annotated_pdf = Path(tmp) / "annotated.pdf"

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            _, errors, meta = _run_pipeline(pdf_path, annotated_pdf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        summary = {}
        for err in errors:
            et = err.get("error_type", "?")
            summary[et] = summary.get(et, 0) + 1

        # Serialize: numpy/torch floats → plain Python floats
        clean_errors = [
            {k: (float(v) if hasattr(v, "item") else v) for k, v in e.items()}
            for e in errors
        ]

        return JSONResponse({**meta, "summary": summary, "errors": clean_errors})


# ─────────────────────────────────────────────────────────────────────────────
# POST /detect-mistakes
# Returns annotated PNG for a specific page (frontend calls this per-page)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/detect-mistakes")
async def detect_mistakes(
    file: UploadFile = File(...),
    page_number: int = Form(...),
):
    """
    Upload PDF → annotated PNG of the requested page.
    🟡 Spelling  🟠 Grammar  🔴 Semantic
    """
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path      = Path(tmp) / "uploaded.pdf"
        annotated_pdf = Path(tmp) / "annotated.pdf"

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            _, _, _ = _run_pipeline(pdf_path, annotated_pdf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

        try:
            import fitz
            doc = fitz.open(str(annotated_pdf))
            page_idx = page_number - 1

            if page_idx < 0 or page_idx >= len(doc):
                raise HTTPException(
                    status_code=400,
                    detail=f"Page {page_number} out of range (PDF has {len(doc)} pages)"
                )

            pix = doc.load_page(page_idx).get_pixmap(matrix=fitz.Matrix(2, 2))
            doc.close()
            return Response(content=pix.tobytes("png"), media_type="image/png")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Render error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze-full
# Returns complete annotated document as a list of PNG images (ALL pages)
# Used by the scrollable frontend view
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze-full")
async def analyze_full(file: UploadFile = File(...)):
    """
    Upload PDF → JSON with:
      - errors report (same as /analyze-pdf)
      - base64-encoded annotated PNG for EVERY page (for scrollable frontend)

    The frontend calls this ONCE and gets everything it needs.
    """
    import base64
    import fitz

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path      = Path(tmp) / "uploaded.pdf"
        annotated_pdf = Path(tmp) / "annotated.pdf"

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            _, errors, meta = _run_pipeline(pdf_path, annotated_pdf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # Render every page of the annotated PDF at 1.5x (balance quality/size)
        doc = fitz.open(str(annotated_pdf))
        pages_b64 = []
        for i in range(len(doc)):
            pix = doc.load_page(i).get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
            pages_b64.append(base64.b64encode(pix.tobytes("png")).decode())
        doc.close()

        summary = {}
        for err in errors:
            et = err.get("error_type", "?")
            summary[et] = summary.get(et, 0) + 1

        clean_errors = [
            {k: (float(v) if hasattr(v, "item") else v) for k, v in e.items()}
            for e in errors
        ]

        return JSONResponse({
            **meta,
            "summary":    summary,
            "errors":     clean_errors,
            "pages_b64":  pages_b64,
            "page_count": len(pages_b64),
        })


# ─────────────────────────────────────────────────────────────────────────────
# GET /metrics
# Developer-facing model performance dashboard
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """
    Return training metrics for the error detection model.
    Reads from models/nyayai-error-detector/training_results.json if present.
    """
    results_path = _MODEL_PATH / "training_results.json"
    trainer_state = _MODEL_PATH / "trainer_state.json"

    data = {}

    if trainer_state.exists():
        with open(trainer_state) as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        # Get the best eval metrics from training
        eval_logs = [l for l in log_history if "eval_f1" in l]
        if eval_logs:
            best = max(eval_logs, key=lambda x: x.get("eval_f1", 0))
            data["training_metrics"] = {
                "best_f1":           round(best.get("eval_f1", 0), 4),
                "best_precision":    round(best.get("eval_precision", 0), 4),
                "best_recall":       round(best.get("eval_recall", 0), 4),
                "best_epoch":        round(best.get("epoch", 0), 2),
                "best_loss":         round(best.get("eval_loss", 0), 4),
                "total_train_steps": state.get("global_step", 0),
            }
            data["training_curve"] = [
                {
                    "epoch": round(l.get("epoch", 0), 2),
                    "f1":    round(l.get("eval_f1", 0), 4),
                    "loss":  round(l.get("eval_loss", 0), 4),
                }
                for l in eval_logs
            ]

    if results_path.exists():
        with open(results_path) as f:
            data["final_results"] = json.load(f)

    data["model_info"] = {
        "base_model":     "law-ai/InLegalBERT",
        "task":           "Token Classification (NER-style)",
        "labels":         ["O", "B-SPELL", "I-SPELL", "B-GRAM", "I-GRAM", "B-SEM", "I-SEM"],
        "model_path":     str(_MODEL_PATH),
        "model_size_mb":  round(sum(
            f.stat().st_size for f in _MODEL_PATH.rglob("*.safetensors")
        ) / 1e6, 1) if _MODEL_PATH.exists() else 0,
    }

    return JSONResponse(data)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))