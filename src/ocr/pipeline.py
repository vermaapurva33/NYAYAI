"""
OCR pipeline.

Assumptions this file is built on:

1) Input PDFs/images are NOT trusted.
   They can be broken, huge, or malicious.
   This code assumes it runs inside a container
   and never relies on network access.

2) Every step must be restartable.
   If the process dies halfway, rerunning should
   continue from disk, not from scratch.

3) GPU is optional.
   If it exists, we use it.
   If it doesn't, things should still work.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
import hashlib

from src.common.config import TEMP_DIR
from src.ocr.pdf_to_images import pdf_to_images
from src.ocr.gpu_preprocessor import preprocess_image
from src.ocr.layout_detector import detect_layout
from src.ocr.ocr_engine_paddle import run_ocr
from src.ocr.text_reconstructor import reconstruct_text
from src.ocr.text_postprocessor import postprocess_text
from src.ocr.pdf_text_extractor import extract_page_text



# ---------------------------------------------------------------------
# One page = one unit of work
# ---------------------------------------------------------------------

@dataclass
class Page:
    page_no: int

    image_path: Path
    preprocessed_path: Optional[Path] = None  # these are used in preprocess -> ocr.

    layout: Optional[List[Dict]] = None
    ocr_blocks: Optional[List[Dict]] = None

    text: Optional[str] = None
    text_layer: Optional[str] = None
    confidence: Optional[float] = None

    failed: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------
# Per-document state
# ---------------------------------------------------------------------

class Document:
    def __init__(self, input_path: Path, work_dir: Path, doc_id: str):
        self.id = doc_id
        self.input_path = input_path
        self.work_dir = work_dir

        self.pages: Dict[int, Page] = {}
        self.meta: Dict = {}

        self.status = "init"


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

class OCRPipeline:

    def __init__(self, use_layout: bool = True, use_postprocess: bool = True):
        self.use_layout = use_layout
        self.use_postprocess = use_postprocess
        self.has_gpu = self._check_gpu()

    # ---------------- public ----------------

    def run(self, path: str | Path) -> Document:
        path = Path(path)
        doc = self._setup_document(path)

        self._extract_pages(doc)
        self._extract_text_layer(doc)
        self._preprocess(doc)


        if self.use_layout:
            self._find_layout(doc)

        self._ocr(doc)
        self._rebuild_text(doc)

        if self.use_postprocess:
            self._cleanup(doc)

        doc.status = "done"
        return doc

    # ---------------- helpers ----------------

    def _check_gpu(self) -> bool:
        try:
            import cv2
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except Exception:
            return False

    def _doc_hash(self, path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()[:16]

    # ---------------- stages ----------------

    def _setup_document(self, path: Path) -> Document:
        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError("Input file does not exist")

        if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
            raise ValueError("Unsupported file type")

        doc_id = self._doc_hash(path)
        work_dir = TEMP_DIR / doc_id #doc id is a variable name so no quotes
        work_dir.mkdir(parents=True, exist_ok=True)

        return Document(path, work_dir, doc_id)

    def _extract_pages(self, doc: Document) -> None:
        pages_dir = doc.work_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        existing = sorted(
            pages_dir.glob("page-*.png"),
            key=lambda p: int(p.stem.split("-")[1]) #custom sorting key
        )
        
        if not existing:
            images = pdf_to_images(doc.input_path, pages_dir)
        else:
             # Extract page numbers that exist
            existing_nums = [int(p.stem.split("-")[1]) for p in existing]

            # Check if pages are contiguous: 1,2,3,4,...N
            expected_nums = list(range(1, max(existing_nums) + 1))

            is_contiguous = existing_nums == expected_nums

            if not is_contiguous:
                # Incomplete or corrupted extraction
                # Strategy: wipe and redo extraction
                for p in existing:
                    p.unlink()

                images = pdf_to_images(doc.input_path, pages_dir)
            else:
                images = existing

        # Register pages in document state
        for i, img in enumerate(images):
            if i not in doc.pages:
                doc.pages[i] = Page(
                    page_no=i,
                    image_path=img
                )

    def _extract_text_layer(self, doc: Document) -> None:
        """
        Populate page.text_layer if the PDF page already has real text.
        This uses pdf_text_extractor.py and does not do OCR.
        """
        for page in doc.pages.values():
            try:
                text = extract_page_text(doc.input_path, page.page_no)
            except Exception:
                continue

            if text:
                page.text_layer = text
                page.text = text
                page.confidence = 1.0


    def _preprocess(self, doc: Document) -> None:
        out_dir = doc.work_dir / "preprocessed"
        out_dir.mkdir(exist_ok=True)

        for page in doc.pages.values():
            if page.failed or page.text_layer is not None: #IF THE PAGE ALREADY HAS A TEXT LAYER, SKIP PREPROCESSING.
                continue


            out = out_dir / page.image_path.name # preprocessed image path

            if out.exists():
                page.preprocessed_path = out
                continue

            try:
                page.preprocessed_path = preprocess_image(
                    page.image_path,
                    out
                )
            except Exception as e:
                page.failed = True
                page.error = f"preprocess: {e}"

    def _find_layout(self, doc: Document) -> None:
        for page in doc.pages.values():
            if page.failed or page.text_layer is not None or not page.preprocessed_path: # we might need to change this if we want to check layout on text_layer pages
                continue
            try:
                page.layout = detect_layout(page.preprocessed_path)
            except Exception as e:
                page.failed = True
                page.error = f"layout: {e}"

    def _ocr(self, doc: Document) -> None:
        for page in doc.pages.values():
            if page.failed or page.text_layer is not None or not page.preprocessed_path:
                continue


            try:
                page.ocr_blocks = run_ocr(
                    page.preprocessed_path,
                    page.layout
                )
            except Exception as e:
                page.failed = True
                page.error = f"ocr: {e}"

    def _rebuild_text(self, doc: Document) -> None:
        for page in doc.pages.values():
            if page.failed or not page.ocr_blocks:
                continue

            try:
                page.text = reconstruct_text(page.ocr_blocks)
            except Exception as e:
                page.failed = True
                page.error = f"text: {e}"

    def _cleanup(self, doc: Document) -> None:
        for page in doc.pages.values():
            if page.failed or not page.text:
                continue

            try:
                result = postprocess_text(page.text)
                page.text = result.get("text")
                page.confidence = result.get("confidence")
            except Exception as e:
                page.failed = True
                page.error = f"post: {e}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ocr.pipeline data/raw/doc1.pdf")
        sys.exit(1)

    input_path = sys.argv[1]

    pipeline = OCRPipeline()
    doc = pipeline.run(input_path)

    print("Document ID:", doc.id)
    print("Status:", doc.status)
    print("Pages processed:", len(doc.pages))

    for i, page in doc.pages.items():
        print(f"\n--- Page {i} ---")
        if page.failed:
            print("FAILED:", page.error)
        else:
            print("Confidence:", page.confidence)
            print("Text preview:")
            print((page.text or "")[:500])

