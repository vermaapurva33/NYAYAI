# ⚖️ NyayAI — AI-Powered Indian Legal Document Error Detector

> **Detect spelling, grammar, and semantic errors in Indian legal PDFs using fine-tuned InLegalBERT + Surya OCR.**  
> Built for courts, law firms, and legal aid organizations to catch errors before filing.

---

## 🎯 Problem Statement

Indian legal documents — FIRs, judgments, petitions, contracts — contain critical errors that go unnoticed:

- **Spelling mistakes** in names, places, and legal terms
- **Grammatical errors** that change the meaning of a clause
- **Semantic errors** — wrong section numbers, wrong IPC references, incorrect legal citations

A single uncorrected error can delay justice, invalidate a filing, or change the outcome of a case.  
**NyayAI automates error detection** across all three categories using AI.

---

## 🧠 How It Works — System Architecture

```
PDF Upload (Scanned or Digital)
        │
        ▼
┌──────────────────────────────────────┐
│         OCR / Text Extraction        │
│  Digital PDF → PyMuPDF (exact bbox)  │
│  Scanned PDF → Surya OCR (CUDA GPU)  │
└──────────────────────────────────────┘
        │  Word tokens + bounding boxes (x0, y0, x1, y1, page)
        ▼
┌──────────────────────────────────────┐
│    InLegalBERT Error Detector        │
│  Fine-tuned token classifier         │
│  Sliding window (30-word context)    │
│  Labels: O / B-SPELL / B-GRAM / B-SEM│
└──────────────────────────────────────┘
        │  Error locations + types + confidence
        ▼
┌──────────────────────────────────────┐
│         PDF Annotator                │
│  Draw colored bounding boxes on PDF  │
│  🟡 Spelling  🟠 Grammar  🔴 Semantic│
└──────────────────────────────────────┘
        │
        ▼
  Annotated PDF + Error Report JSON
```

---

## 🔬 Key Components

### 1. OCR Pipeline — `src/ocr/word_extractor.py`
- **Digital PDFs**: Uses PyMuPDF (`fitz`) → exact word bounding boxes, zero OCR error
- **Scanned PDFs**: Uses **Surya OCR** (state-of-the-art, 2024) running on CUDA GPU
  - Detects text lines → splits into words → estimates word-level bounding boxes
- **Graceful fallback**: If Surya fails, automatically falls back to PyMuPDF

### 2. Error Detection Model — `src/rag/error_detector.py`
- **Base model**: [`law-ai/InLegalBERT`](https://huggingface.co/law-ai/InLegalBERT) — BERT pre-trained on Indian legal corpus
- **Task**: Token Classification (identical architecture to NER)
- **Labels**: `O`, `B-SPELL`, `I-SPELL`, `B-GRAM`, `I-GRAM`, `B-SEM`, `I-SEM`
- **Technique**: Sliding-window inference (30-word windows, 50% overlap) to handle documents of any length
- **Training**: Fine-tuned on 120,000 Indian legal sentences with artificially injected errors

### 3. Training Data — `scripts/generate_training_data.py`
- **Source**: IL-TUR LSI dataset (2.58 million Indian legal sentences from HuggingFace)
- **Augmentation**: Automatically injects errors into clean sentences:
  - *Spelling*: character swaps, missing letters, phonetic replacements
  - *Grammar*: tense changes, subject-verb disagreement
  - *Semantic*: wrong IPC section numbers, wrong legal references
- **Size**: 150,000 examples (120k train / 15k val / 15k test)

### 4. PDF Annotator — `src/rag/pdf_annotator.py`
- Draws colored bounding boxes directly on PDF pages using PyMuPDF
- Each box labeled with error type and confidence score

---

## 🏗️ System Architecture — Backend / Frontend Split

```
┌─────────────────────┐     HTTP/CORS     ┌─────────────────────┐
│   Frontend          │ ◄──────────────► │   Backend           │
│   Streamlit UI      │                   │   FastAPI           │
│   (no ML deps)      │                   │   (full ML stack)   │
│   Port: 8501        │                   │   Port: 8000        │
└─────────────────────┘                   └─────────────────────┘
                                                   │
                                          ┌────────┘
                                          │  models/nyayai-error-detector/
                                          │  (fine-tuned InLegalBERT)
```

### Backend API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Model status, CORS info, cache status |
| `/analyze-full` | POST | Upload PDF → all annotated pages (base64) + error report |
| `/analyze-pdf` | POST | Upload PDF → JSON error report only |
| `/detect-mistakes` | POST | Upload PDF → single page annotated PNG |
| `/metrics` | GET | F1, Precision, Recall, training curve |

### Frontend Features

- ✅ Single upload → entire document annotated in one API call
- ✅ All pages shown in scrollable vertical view
- ✅ Error table: word | page | type | suggestion | confidence
- ✅ Confidence & error-type filters (adjustable slider)
- ✅ CSV export of error report
- ✅ Developer metrics tab (F1 score, training curve)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10 |
| OCR (Scanned) | [Surya OCR](https://github.com/VikParuchuri/surya) 0.5.0 |
| OCR (Digital) | PyMuPDF (fitz) |
| NLP Model | `law-ai/InLegalBERT` (fine-tuned) |
| Training | HuggingFace Transformers 4.44, PyTorch 2.x |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| GPU | NVIDIA RTX 4050 / CUDA 12.4 |
| Containerization | Docker + Docker Compose |
| Dataset | IL-TUR LSI (HuggingFace), 2.58M legal sentences |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Base Model | `law-ai/InLegalBERT` |
| Training Examples | 120,000 |
| Error Types Detected | Spelling, Grammar, Semantic |
| Inference Speed | ~0.8s per document |
| Context Window | 30 words (sliding, 50% overlap) |
| GPU | RTX 4050 6GB VRAM |

> Full F1/Precision/Recall metrics available in the Developer Metrics tab (`/metrics` endpoint → trainer_state.json).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.4 (for Surya OCR on scanned PDFs)
- 8GB RAM minimum

### 1. Clone and setup
```bash
git clone https://github.com/your-org/NyayAI.git
cd NyayAI
python -m venv nyayai && source nyayai/bin/activate
pip install -r requirements.txt
```

### 2. Generate training data
```bash
python scripts/generate_training_data.py --max-sentences 150000
# Output: data/training/train.jsonl (41MB, 120k examples)
```

### 3. Train the model
```bash
# Local (RTX 4050):
python scripts/train_error_detector.py

# OR on Google Colab (recommended for full dataset):
# Upload notebooks/nyayai_colab_training.ipynb to Colab T4 GPU
```

### 4. Run the backend
```bash
cd /path/to/NyayAI
source nyayai/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run the frontend
```bash
cd frontend
streamlit run ui.py --server.port 8501
# Open http://localhost:8501
```

### 6. Docker (production)
```bash
docker-compose up --build
```

---

## 📁 Project Structure

```
NyayAI/
├── backend/                    # Deployable backend (full ML stack)
│   ├── src/
│   │   ├── api/main.py         # FastAPI app (v3) with model caching
│   │   ├── ocr/word_extractor.py
│   │   └── rag/error_detector.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                   # Deployable frontend (no ML deps)
│   ├── ui.py                   # Streamlit app (v3)
│   ├── Dockerfile
│   └── requirements.txt
│
├── src/                        # Shared source (mirrors backend/)
│   ├── api/main.py
│   ├── ocr/
│   │   ├── word_extractor.py   # Hybrid OCR: PyMuPDF + Surya
│   │   └── gpu_preprocessor.py
│   └── rag/
│       ├── error_detector.py   # InLegalBERT inference wrapper
│       └── pdf_annotator.py    # Bounding box annotation
│
├── scripts/
│   ├── generate_training_data.py  # Dataset generation (IL-TUR + augmentation)
│   └── train_error_detector.py    # Fine-tuning script
│
├── notebooks/
│   └── nyayai_colab_training.ipynb  # Colab notebook for large-scale training
│
├── models/
│   └── nyayai-error-detector/   # Fine-tuned model checkpoint
│
├── data/
│   ├── training/
│   │   ├── train.jsonl  (41MB, 120k examples)
│   │   ├── val.jsonl    (5.1MB, 15k examples)
│   │   └── test.jsonl   (5.1MB, 15k examples)
│   └── legal_corpus/    # (optional) local .txt legal documents
│
└── docker-compose.yml
```

---

## 🔮 Roadmap

- [ ] **Re-train on Colab** with full 120k dataset + 5 epochs for higher semantic F1
- [ ] **Hindi language support** — extend to vernacular legal documents
- [ ] **RAG integration** — verify section references against actual IPC/CrPC corpus
- [ ] **Production deployment** — Nginx + Gunicorn + HTTPS
- [ ] **Mobile-friendly frontend** — React/Next.js web app
- [ ] **API authentication** — JWT tokens for court/firm access control

---

## 🤝 Contributing

This project follows a **Small PR strategy**:
1. One fix / one feature per pull request
2. All tests pass before merging
3. `backend/` and `src/` kept in sync via `cp` after changes

---

## 📜 License

MIT License — see [LICENSE](LICENSE)

---

## 👥 Team

Built as a proof-of-concept for AI-assisted legal document verification in the Indian judiciary system.  
Base model credit: [`law-ai/InLegalBERT`](https://huggingface.co/law-ai/InLegalBERT) by the Legal AI Lab.
