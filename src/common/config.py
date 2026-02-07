from pathlib import Path

# ---------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------

# repo root
BASE_DIR = Path(__file__).resolve().parents[2]

# data layout
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # input files
PROCESSED_DIR = DATA_DIR / "processed"
TEMP_DIR = DATA_DIR / "temp"        # per-document scratch space


# ---------------------------------------------------------------------
# OCR-related settings
# ---------------------------------------------------------------------

# DPI used when converting PDFs to images
# 300 is a good balance between accuracy and size
PDF_DPI = 300

# Languages passed to OCR engines
# order matters for some models
LANGUAGES = ['en','devanagari']  # English and Devanagari


# ---------------------------------------------------------------------
# Runtime behaviour
# ---------------------------------------------------------------------

# Whether GPU usage is allowed.
# Actual availability is still checked at runtime.
USE_GPU = True

# Hard limit to avoid pathological PDFs
MAX_PDF_PAGES = 2000 # max pages in input PDF, can be adjusted as needed
