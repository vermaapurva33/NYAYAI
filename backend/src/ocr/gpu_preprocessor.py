# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false

import cv2
import numpy as np
from pathlib import Path
from src.common.config import USE_GPU


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Grayscale
# ──────────────────────────────────────────────────────────────────────────────

def _grayscale(img: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to grayscale if not already single-channel."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Denoising  (MUST come before deskew — clean signal → better edge detection)
# ──────────────────────────────────────────────────────────────────────────────

def _denoise(img: np.ndarray, use_gpu: bool) -> np.ndarray:
    """
    Remove scan noise before any geometric correction.
    GPU path uses cuda fastNlMeansDenoising; CPU path uses the standard version.
    h=10 is good for light scan noise. Raise to 15 for heavy noise.
    """
    if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_img = cv2.cuda.fastNlMeansDenoising(gpu_img, h=10)
        return gpu_img.download()
    else:
        return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Deskew  (MUST come after denoise — noisy edges give wrong angles)
# ──────────────────────────────────────────────────────────────────────────────

def _deskew(img: np.ndarray) -> np.ndarray:
    """
    Detect and correct document skew using Hough line detection.

    Teaching note:
        Scanned documents are rarely perfectly straight — the scanner glass
        or hand placement causes a small tilt. Even 1-2 degrees of skew
        degrades OCR significantly. We detect horizontal text baselines
        using the Hough transform and rotate the image to correct the angle.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return img  # No lines detected — can't determine skew

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only use near-horizontal lines (text baselines, not vertical strokes)
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))

    # Skip correction for tiny skew — rotation itself introduces minor blurring
    if abs(median_angle) < 0.5:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE  # avoid black borders at edges
    )
    return rotated


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — CLAHE  (MUST come before binarization — fix uneven lighting first)
# ──────────────────────────────────────────────────────────────────────────────

def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization.

    Teaching note:
        Old court documents and photocopies often have uneven lighting —
        darker corners, faded center, or ink that varies across the page.
        Global histogram equalization (regular HE) would over-brighten already
        bright regions. CLAHE works on small tiles (8x8) and clips the
        contrast boost to avoid over-amplifying noise. clipLimit=2.0 is
        conservative and safe for most documents.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Morphological Cleanup
# ──────────────────────────────────────────────────────────────────────────────

def _morphological_cleanup(img: np.ndarray) -> np.ndarray:
    """
    Close small gaps within characters caused by broken ink strokes from scanning.

    Teaching note:
        When a pen stroke is thin or the scanner resolution is low, characters
        like 'i', 'l', 'r' can have tiny gaps. Morphological CLOSE fills these
        gaps by dilating (expanding) then eroding (shrinking). The (2,1) kernel
        is narrow horizontally — it closes horizontal gaps within a character
        without merging adjacent characters.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


# ──────────────────────────────────────────────────────────────────────────────
# Step 6 — Adaptive Binarization  (ALWAYS last — converts grey image to B&W)
# ──────────────────────────────────────────────────────────────────────────────

def _binarize(img: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding — better than global Otsu for documents with
    uneven backgrounds (shadows, stains, faded edges).

    Teaching note:
        Global thresholding (Otsu) picks ONE threshold for the whole image.
        If the background is slightly grey on the left and bright white on the
        right, a single threshold will leave noise on one side.
        Adaptive thresholding computes a local threshold for each 31x31 pixel
        block — so it adapts to local lighting conditions.
        blockSize must be odd and larger than the largest character stroke.
        C=10 is the constant subtracted from the local mean — this prevents
        very faint areas from being "forced" to black.
    """
    return cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_image(input_path: Path, output_path: Path) -> Path:
    """
    Full 6-step preprocessing pipeline for OCR.

    Order is critical:
        grayscale → denoise → deskew → CLAHE → morph → binarize

    Each step improves the input for the NEXT step — doing them out of
    order degrades quality (e.g. binarizing before CLAHE loses contrast info).
    """
    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {input_path}")

    img = _grayscale(img)
    img = _denoise(img, USE_GPU)
    img = _deskew(img)
    img = _enhance_contrast(img)
    img = _morphological_cleanup(img)
    img = _binarize(img)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    return output_path

