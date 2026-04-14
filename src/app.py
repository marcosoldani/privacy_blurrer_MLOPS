"""
app.py - FastAPI service for Privacy Blurrer.

Endpoints:
  GET  /health   -> liveness check
  POST /predict  -> binary person segmentation mask

Step 6 additions:
  - Structured JSON logging to logs/predictions.jsonl
  - Drift detection via Alibi Detect KSDrift (monitor.py)

Step 7 additions:
  - Input validation (format, dimensions, file size)
  - Guardrails with clear HTTP error codes
  - Improved logging (preprocess_ms / inference_ms split, rejection_reason)
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
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from src.monitor import check_drift, load_detector

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_DIMENSION_PX = 32
MAX_DIMENSION_PX = 4096
ALLOWED_FORMATS = {"JPEG", "JPG", "PNG", "BMP", "WEBP"}

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

    if not MODEL_PATH.exists():
        logger.warning(f"Model weights not found at {MODEL_PATH}. /predict will fail.")
    else:
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE)
        m.eval()
        model = m
        logger.info(f"Model loaded from {MODEL_PATH} on {DEVICE}")

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


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_image(contents: bytes, filename: str) -> Image.Image:
    """
    Validate the uploaded image.
    Raises HTTPException with appropriate status codes on failure.
    Returns a PIL Image on success.
    """
    # 1. File size check
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents) / (1024*1024):.1f}MB. Maximum allowed: {MAX_FILE_SIZE_MB}MB."
        )

    # 2. Valid image format check
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Checks for corruption
        image = Image.open(io.BytesIO(contents))  # Re-open after verify
    except (UnidentifiedImageError, Exception):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: '{filename}'. Please upload a valid image."
        )

    # 3. Format whitelist check
    fmt = image.format or ""
    if fmt.upper() not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Allowed formats: {', '.join(ALLOWED_FORMATS)}."
        )

    # 4. Dimension checks
    w, h = image.size
    if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small: {w}x{h}px. Minimum size: {MIN_DIMENSION_PX}x{MIN_DIMENSION_PX}px."
        )
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large: {w}x{h}px. Maximum size: {MAX_DIMENSION_PX}x{MAX_DIMENSION_PX}px."
        )

    return image


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "detector_loaded": detector is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    request_start = time.time()
    contents = await file.read()

    # ── Validation (guardrail) ──────────────────────────────────────────────
    preprocess_start = time.time()
    try:
        image = validate_image(contents, file.filename)
    except HTTPException as e:
        rejection_reason = e.detail
        latency_ms = round((time.time() - request_start) * 1000, 2)

        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filename": file.filename,
            "validation_passed": False,
            "rejection_reason": rejection_reason,
            "latency_ms": latency_ms,
        }
        log_prediction(log_record)
        logger.warning(f"Request rejected | file={file.filename} | reason={rejection_reason}")
        raise

    # ── Preprocessing ───────────────────────────────────────────────────────
    original_size = image.size  # (W, H)
    image_rgb = image.convert("RGB")
    tensor = preprocess(image_rgb).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
    preprocess_ms = round((time.time() - preprocess_start) * 1000, 2)

    # ── Drift detection ─────────────────────────────────────────────────────
    drift_result = {"is_drift": None, "drift_score": None, "features": None}
    if detector is not None:
        try:
            drift_result = check_drift(tensor.squeeze(0), detector=detector)
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")

    # ── Inference ───────────────────────────────────────────────────────────
    inference_start = time.time()
    with torch.no_grad():
        output = model(tensor)
        mask = torch.sigmoid(output).squeeze()
        binary_mask = (mask > 0.5).cpu().numpy().astype(np.uint8) * 255
    inference_ms = round((time.time() - inference_start) * 1000, 2)

    total_latency_ms = round((time.time() - request_start) * 1000, 2)
    mask_coverage = round(float((binary_mask > 0).mean()), 4)

    # ── Structured log ──────────────────────────────────────────────────────
    log_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": file.filename,
        "validation_passed": True,
        "original_size": list(original_size),
        "mask_coverage": mask_coverage,
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "latency_ms": total_latency_ms,
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
            f"score={drift_result['drift_score']} | features={drift_result['features']}"
        )
    else:
        logger.info(
            f"Prediction OK | file={file.filename} | "
            f"coverage={mask_coverage} | preprocess={preprocess_ms}ms | "
            f"inference={inference_ms}ms | total={total_latency_ms}ms | "
            f"drift={drift_result['is_drift']} | score={drift_result['drift_score']}"
        )

    # ── Return mask as PNG ──────────────────────────────────────────────────
    mask_image = Image.fromarray(binary_mask)
    mask_image = mask_image.resize(original_size, Image.NEAREST)
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")