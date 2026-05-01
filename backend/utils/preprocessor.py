"""
ECG Image Preprocessor
=======================
Handles: JPG, PNG, BMP, TIFF, DICOM, PDF
Handles: Any size, blurry images, low contrast, rotated, noisy
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import os

# Optional PDF support
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Optional DICOM support
try:
    import pydicom
    DICOM_SUPPORT = True
except ImportError:
    DICOM_SUPPORT = False


def detect_file_type(file_bytes: bytes, filename: str) -> str:
    """Detect file type from bytes and filename"""
    ext = os.path.splitext(filename.lower())[1]

    if file_bytes[:4] == b'%PDF':
        return 'pdf'
    if file_bytes[:4] == b'DICM' or ext == '.dcm':
        return 'dicom'
    if file_bytes[:3] == b'\xff\xd8\xff':
        return 'jpeg'
    if file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'png'

    ext_map = {'.jpg': 'jpeg', '.jpeg': 'jpeg', '.png': 'png',
               '.bmp': 'bmp', '.tiff': 'tiff', '.tif': 'tiff',
               '.pdf': 'pdf', '.dcm': 'dicom'}
    return ext_map.get(ext, 'jpeg')


def load_from_pdf(file_bytes: bytes) -> np.ndarray:
    """Convert PDF ECG report to image"""
    if not PDF_SUPPORT:
        raise ValueError("pdf2image not installed. Run: pip install pdf2image")
    # FIX: Use dpi=150 instead of 300 — faster load, still sharp enough for 224x224 model
    pages = convert_from_bytes(file_bytes, dpi=150, first_page=1, last_page=1)
    img = np.array(pages[0])
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def load_from_dicom(file_bytes: bytes) -> np.ndarray:
    """Load DICOM ECG file"""
    if not DICOM_SUPPORT:
        raise ValueError("pydicom not installed. Run: pip install pydicom")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        ds = pydicom.dcmread(tmp_path)
        pixel_array = ds.pixel_array
        pixel_array = ((pixel_array - pixel_array.min()) /
                       (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        if len(pixel_array.shape) == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2BGR)
        return pixel_array
    finally:
        os.unlink(tmp_path)


def load_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """Load image from any format"""
    file_type = detect_file_type(file_bytes, filename)

    if file_type == 'pdf':
        return load_from_pdf(file_bytes)
    elif file_type == 'dicom':
        return load_from_dicom(file_bytes)
    else:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            pil_img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img


def detect_blur(img: np.ndarray) -> float:
    """Return blur score — lower = more blurry (threshold ~100)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def enhance_blurry_image(img: np.ndarray) -> np.ndarray:
    """Sharpen blurry ECG image"""
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    return cv2.filter2D(sharpened, -1, kernel)


def auto_rotate(img: np.ndarray) -> np.ndarray:
    """Auto-detect and fix rotation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)

    if lines is None:
        return img

    angles = []
    for rho, theta in lines[:10, 0]:
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    if abs(median_angle) < 1:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Enhance low contrast ECG image using CLAHE"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def remove_noise_fast(img: np.ndarray) -> np.ndarray:
    """
    FIX: fastNlMeansDenoisingColored takes 30-60s on CPU → causes 500 timeout crash.
    bilateralFilter is 50x faster and works great for ECG images.
    """
    return cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)


def preprocess_ecg(file_bytes: bytes, filename: str) -> dict:
    """
    Full preprocessing pipeline for any ECG input.
    Returns: processed image + quality metadata
    """
    img = load_image(file_bytes, filename)
    original_shape = img.shape
    issues_detected = []

    h, w = img.shape[:2]
    if h < 100 or w < 100:
        issues_detected.append("Very small image — upscaled")
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    elif h > 4000 or w > 4000:
        issues_detected.append("Very large image — downscaled")
        scale = 2000 / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))

    blur_score = detect_blur(img)
    if blur_score < 100:
        issues_detected.append(f"Blurry image detected (score: {blur_score:.1f}) — enhanced")
        img = enhance_blurry_image(img)

    img = enhance_contrast(img)
    img = remove_noise_fast(img)   # FIX: was remove_noise() which used slow NlMeans
    img = auto_rotate(img)

    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    return {
        "processed_image": img_rgb,
        "original_shape": original_shape,
        "blur_score": float(round(blur_score, 2)),
        "is_blurry": bool(blur_score < 100),
        "issues_detected": issues_detected,
        "file_type": detect_file_type(file_bytes, filename),
        "quality_score": int(min(100, max(0, int(blur_score / 5))))
    }
