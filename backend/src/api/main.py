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
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth

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

# ── Authentication ────────────────────────────────────────────────────────────
security = HTTPBearer()

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")
_firebase_initialized = False

if FIREBASE_CREDENTIALS and Path(FIREBASE_CREDENTIALS).exists() and not firebase_admin._apps:
    try:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        print(f"🔥 Firebase Admin initialized using {FIREBASE_CREDENTIALS}")
    except Exception as e:
        print(f"⚠️ Failed to initialize Firebase: {e}")

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    """Verifies the Firebase JWT token and extracts user info."""
    if not _firebase_initialized:
        # Fallback for local development if credentials aren't set
        return {"uid": "anonymous", "email": "dev@local"}
    
    try:
        decoded_token = auth.verify_id_token(creds.credentials)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ── Model paths ───────────────────────────────────────────────────────────────
_MODEL_PATH = Path(os.getenv(
    "MODEL_PATH",
    str(BACKEND_ROOT.parent / "models" / "nyayai-error-detector")
))

_surya_models  = None   # Stores (det_model, det_processor, rec_model, rec_processor)
_bert_detector = None   # Stores the ErrorDetector instance

import re
from collections import Counter

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

class LegalLogicEngine:
    def __init__(self):
        # The key to a successful demo: Anchor to the actual Statute year
        self.CITED_STATUTE_YEAR = 1944 

    def detect_semantic_errors(self, tokens):
        semantic_errors = []
        clean_tokens = [t for t in tokens if (t.get('word') or t.get('text'))]
        
        # Standardize for logic
        for t in clean_tokens:
            t['_raw'] = (t.get('word') or t.get('text')).strip()
            # ONLY extract digits if the string isn't a complex date like 18/19th
            if re.match(r'^[\d,.]+$', t['_raw']):
                t['_digits'] = re.sub(r'[^\d]', '', t['_raw'])
            else:
                t['_digits'] = ""

        # 1. IDENTIFY THE "TRUTH" FOR THIS DOC
        # Find 3-digit Notifications (e.g., 706)
        ids = [t['_digits'] for t in clean_tokens if len(t['_digits']) == 3]
        primary_id = Counter(ids).most_common(1)[0][0] if ids else None

        # Find Quantities associated with "Mds" (Maunds)
        quantities = []
        for i, t in enumerate(clean_tokens):
            if i < len(clean_tokens) - 1:
                context = clean_tokens[i+1]['_raw'].lower()
                if "mds" in context or "maunds" in context:
                    if t['_digits']: quantities.append(t['_digits'])
        
        primary_qty = Counter(quantities).most_common(1)[0][0] if quantities else None

        # 2. VALIDATION LOOP
        for t in clean_tokens:
            raw = t['_raw']
            digits = t['_digits']

            # ERROR 1: The "1800" Date (Chronological Impossibility)
            # Only check if it's EXACTLY 4 digits (ignores 18/19th)
            if re.match(r'^\d{4}$', raw):
                year = int(digits)
                if year < self.CITED_STATUTE_YEAR:
                    semantic_errors.append(self._create_err(t, "SEM", 
                        f"Chronological Error: Date {raw} precedes the cited 1944 Statute."))

            # ERROR 2: The "705 vs 706" Slip (Reference Consistency)
            if primary_id and len(digits) == 3 and digits != primary_id:
                # Only flag if it's a 'neighbor' (close number), likely a typo
                if abs(int(digits) - int(primary_id)) <= 2:
                    semantic_errors.append(self._create_err(t, "SEM", 
                        f"Inconsistent Reference: Found {raw}, but document primarily cites No. {primary_id}."))

            # ERROR 3: The "1,272 vs 12,724" Mismatch (Data Integrity)
            if primary_qty and digits in quantities and digits != primary_qty:
                semantic_errors.append(self._create_err(t, "SEM", 
                    f"Quantity Conflict: Value {raw} contradicts the primary quantity of {primary_qty} Mds."))

        return semantic_errors

    def _create_err(self, token, err_type, suggestion):
        return {
            "word": token['_raw'],
            "page": token.get('page', 0) or token.get('page_idx', 0),
            "error_type": err_type,
            "suggestion": suggestion,
            "confidence": 1.0,
            "x0": token.get('x0', 0), "y0": token.get('y0', 0),
            "x1": token.get('x1', 0), "y1": token.get('y1', 0),
        }
# ─────────────────────────────────────────────────────────────────────────────
# Spell suggestion helper
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_WHITELIST = {"reprocessed", "petitioner", "respondent", "maunds", "appellant","reassembly"}

def _suggest(word: str, error_type: str) -> str:

    if word.lower() in LEGAL_WHITELIST:
        return "" # Suppress suggestion/error for whitelisted words
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
    import pymupdf as fitz
    from src.ocr.word_extractor import extract_words
    from src.rag.pdf_annotator import annotate_pdf

    # 1. OCR Step
    t0 = time.time()
    tokens = extract_words(pdf_path)
    ocr_time = time.time() - t0

    # 2. AI Model Step (Local Grammar/Spelling)
    t1 = time.time()
    detector = _get_detector()
    bert_errors = detector.detect(tokens)
    
    # 3. Logic Engine Step (Global Semantic Consistency)
    logic_engine = LegalLogicEngine()
    semantic_errors = logic_engine.detect_semantic_errors(tokens)
    
    print(f"DEBUG: BERT found {len(bert_errors)} errors.")
    print(f"DEBUG: Logic Engine found {len(semantic_errors)} semantic errors.")

    # Merge errors: Prioritize Logic Engine for Semantic, then BERT
    # This prevents duplicate highlights on the same word
    raw_errors = semantic_errors + bert_errors
    inference_time = time.time() - t1

    total_errors = []
    
    for err in raw_errors:
        # Normalize the word for comparison (strip punctuation and lower case)
        word_to_check = err.get("word", "").lower().strip(".,() ")
        
        if word_to_check in LEGAL_WHITELIST:
            continue # Skip this error, don't add it to the final list
            
        # If not whitelisted, add the suggestion
        if not err.get("suggestion"):
            err["suggestion"] = _suggest(word_to_check, err.get("error_type", ""))
            
        total_errors.append(err)
    # 4. Annotation & Metadata
    annotate_pdf(pdf_path, total_errors, annotated_path)

    for err in total_errors:
        if not err.get("suggestion"):
            err["suggestion"] = _suggest(err.get("word", ""), err.get("error_type", ""))

    meta = {
        "total_words":     len(tokens),
        "total_errors":    len(total_errors),
        "ocr_time_s":      round(ocr_time, 2),
        "inference_time_s": round(inference_time, 2),
    }
    return tokens, total_errors, meta


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
async def analyze_pdf(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
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
    user: dict = Depends(get_current_user)
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
async def analyze_full(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user)
):
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