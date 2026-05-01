"""
/api/analyze — Main ECG Analysis Endpoint
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import traceback

from utils.preprocessor import preprocess_ecg
from utils.model import predict, CLASSES
from utils.gradcam import generate_heatmap_overlay

router = APIRouter()

ALLOWED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
    '.pdf', '.dcm', '.webp'
}
MAX_FILE_SIZE_MB = 50


@router.post("/analyze")
async def analyze_ecg(file: UploadFile = File(...)):
    """
    Analyze ECG image/PDF and return full diagnostic report.
    
    Accepts: JPG, PNG, BMP, TIFF, PDF, DICOM
    Any size, blurry images handled automatically.
    """
    try:
        # Validate file extension
        import os
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file bytes
        file_bytes = await file.read()
        
        # Validate file size
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB"
            )
        
        if len(file_bytes) < 100:
            raise HTTPException(status_code=400, detail="File appears to be empty or corrupted.")
        
        # ── Step 1: Preprocess ─────────────────────────────────────────────
        preprocessing = preprocess_ecg(file_bytes, file.filename)
        processed_image = preprocessing["processed_image"]
        
        # ── Step 2: Predict ────────────────────────────────────────────────
        prediction = predict(processed_image)
        
        # ── Step 3: Grad-CAM ───────────────────────────────────────────────
        from utils.model import CLASSES
        predicted_idx = CLASSES.index(prediction["disease"])
        
        try:
            heatmaps = generate_heatmap_overlay(processed_image, predicted_idx)
        except Exception as e:
            print(f"Heatmap generation failed: {e}")
            heatmaps = {
                "heatmap_base64": None,
                "overlay_base64": None,
                "original_base64": None
            }
        
        # ── Step 4: Build Response ─────────────────────────────────────────
        response = {
            "success": True,
            "filename": file.filename,
            "file_size_mb": round(size_mb, 2),
            "preprocessing": {
                "original_size": [int(x) for x in preprocessing["original_shape"][:2]],
                "file_type": preprocessing["file_type"],
                "blur_score": preprocessing["blur_score"],
                "is_blurry": preprocessing["is_blurry"],
                "quality_score": preprocessing["quality_score"],
                "issues_handled": preprocessing["issues_detected"]
            },
            "prediction": prediction,
            "images": heatmaps,
            "disclaimer": (
                "⚠️ This AI analysis is for informational purposes only. "
                "It is NOT a substitute for professional medical diagnosis. "
                "Always consult a qualified cardiologist."
            )
        }
        
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )
