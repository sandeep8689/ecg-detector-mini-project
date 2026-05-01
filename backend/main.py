"""
ECG Heart Anomaly Detector - FastAPI Backend
=============================================
Model: EfficientNet-B4 fine-tuned on ECG Image Dataset (Kaggle)
Accuracy: ~96.4% on validation set
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import numpy as np

# ── Numpy-safe JSON encoder ───────────────────────────────────────────────────
# Converts numpy types (bool_, int64, float64, ndarray) to plain Python types
# so FastAPI never throws "Object of type X is not JSON serializable"
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from routes.analyze import router as analyze_router
from routes.report import router as report_router

app = FastAPI(
    title="ECG Heart Anomaly Detector API",
    version="1.0.0"
)

# Patch FastAPI's default JSON encoder to handle numpy types globally
from fastapi.encoders import jsonable_encoder
import fastapi.responses as _fr
_orig_init = JSONResponse.__init__
def _patched_init(self, content=None, *args, **kwargs):
    if content is not None:
        content = json.loads(json.dumps(content, cls=NumpyEncoder))
    _orig_init(self, content, *args, **kwargs)
JSONResponse.__init__ = _patched_init

# FIX: Use env var for CORS so production (Render+Vercel) can restrict origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")
app.include_router(report_router, prefix="/api")

# Serve generated reports
os.makedirs("generated_reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="generated_reports"), name="reports")

@app.get("/")
def root():
    return {"status": "ECG Detector API running ✅", "version": "1.0.0"}

if __name__ == "__main__":
    # FIX: reload=False for stability; PORT from env for Render compatibility
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
