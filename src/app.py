"""
app.py - FastAPI service for Privacy Blurrer.

Endpoints:
  GET  /health   -> liveness check
  POST /predict  -> binary person segmentation mask

Step 6 additions:
  - Structured JSON logging to logs/predictions.jsonl on every /predict call
  - Drift detection via Alibi Detect KSDrift (monitor.py)
"""

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path


import numpy as np
import torch
import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image
from torchvision import transforms

from src.monitor import check_drift, load_detector

# ── Logging setup ──────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_prediction(record: dict):
    """Append a JSON record to the prediction log file."""
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Model setup ────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "experiments" / "best.pt"
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Privacy Blurrer", version="1.0.0")

model = None
detector = None


@app.on_event("startup")
def load_model():
    global model, detector

    # Load segmentation model
    if not MODEL_PATH.exists():
        logger.warning(f"Model weights not found at {MODEL_PATH}. /predict will fail.")
    else:
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        model = m
        logger.info(f"Model loaded from {MODEL_PATH} on {DEVICE}")

    # Load drift detector (optional: warn if missing, don't crash)
    try:
        detector = load_detector()
        logger.info("Drift detector loaded successfully.")
    except FileNotFoundError as e:
        logger.warning(f"Drift detector not available: {e}. Drift detection will be skipped.")
        detector = None


# ── Transform ──────────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    start_time = time.time()

    # Read and preprocess image
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    original_size = image.size  # (W, H)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)  # (1, C, H, W)

    # Drift detection
    drift_result = {"is_drift": None, "drift_score": None, "features": None}
    if detector is not None:
        try:
            drift_result = check_drift(tensor.squeeze(0), detector=detector)
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")

    # Inference
    with torch.no_grad():
        output = model(tensor)                      # (1, 1, H, W)
        mask = torch.sigmoid(output).squeeze()      # (H, W)
        binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8) * 255

    latency_ms = round((time.time() - start_time) * 1000, 2)

    # Mask statistics
    mask_coverage = round(float((binary_mask > 0).mean()), 4)

    # Structured log
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": file.filename,
        "original_size": list(original_size),
        "mask_coverage": mask_coverage,
        "latency_ms": latency_ms,
        "drift": {
            "is_drift": drift_result["is_drift"],
            "drift_score": drift_result["drift_score"],
            "features_mean_rgb": drift_result["features"],
        },
    }
    log_prediction(log_record)

    if drift_result["is_drift"]:
        logger.warning(
            f"DRIFT DETECTED | file={file.filename} | "
            f"score={drift_result['drift_score']} | "
            f"features={drift_result['features']}"
        )
    else:
        logger.info(
            f"Prediction OK | file={file.filename} | "
            f"coverage={mask_coverage} | latency={latency_ms}ms | "
            f"drift=False | score={drift_result['drift_score']}"
        )

    # Return mask as PNG
    mask_image = Image.fromarray(binary_mask)
    mask_image = mask_image.resize(original_size, Image.NEAREST)
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")