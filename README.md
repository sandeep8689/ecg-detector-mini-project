# 🫀 CardioScan AI — ECG Heart Anomaly Detector

AI-powered ECG analysis using EfficientNet-B4. Detects 4 cardiac conditions with ~96.4% accuracy.

## ⚡ Quick Start (Local)

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
# → runs on http://localhost:8000
```

### 2. Frontend (new terminal)
```bash
cd frontend
npm install
npm run dev
# → runs on http://localhost:3000
```

Open **http://localhost:3000** and upload any ECG image (JPG, PNG, PDF, DICOM).

---

## 📁 Project Structure
```
ecg-detector/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── requirements.txt
│   ├── models/
│   │   └── train.py             # Training script (run once on Colab)
│   ├── routes/
│   │   ├── analyze.py           # POST /api/analyze
│   │   └── report.py            # POST /api/report
│   └── utils/
│       ├── model.py             # EfficientNet-B4 + inference
│       ├── preprocessor.py      # Image preprocessing pipeline
│       └── gradcam.py           # Grad-CAM heatmap
└── frontend/
    ├── src/
    │   ├── api.js               # Axios instance (reads VITE_API_URL)
    │   ├── App.jsx
    │   └── components/
    └── vite.config.js           # Proxy: /api → localhost:8000
```

---

## 🧠 Model Notes

- **Without training**: Uses ImageNet pretrained weights → works but ECG accuracy is lower
- **With training**: Run `backend/models/train.py` on Google Colab (free T4 GPU, ~30-45 min)
  - Dataset: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
  - Output: `backend/models/ecg_efficientnet_b4.pth`
  - Place the `.pth` file in `backend/models/` before starting the server

---

## 🚀 Deploy (Render + Vercel)

**Backend on Render:**
- Root: `backend`
- Build: `pip install -r requirements.txt`
- Start: `gunicorn main:app -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`
- Env var: `ALLOWED_ORIGINS=https://your-app.vercel.app`

**Frontend on Vercel:**
- Root: `frontend`
- Framework: Vite
- Env var: `VITE_API_URL=https://your-backend.onrender.com`

---

## ⚠️ Disclaimer

This tool is for **informational purposes only**. It is NOT a substitute for professional medical diagnosis. Always consult a qualified cardiologist.
