"""
app.py - FastAPI service for Privacy Blurrer.

Endpoints:
  GET  /health            -> liveness check
  POST /predict           -> binary segmentation mask (PNG, bianco/nero)
  POST /blur?blur_type=X  -> immagine con persone anonimizzate (PNG)
                             blur_type: gaussian | pixelate | blackout
  POST /feedback          -> raccoglie giudizio utente (good|bad) sull'output
  GET  /feedback/stats    -> aggregato {total, positive, percentage}

Funzionalita':
  - Structured JSON logging in logs/predictions.jsonl
  - User feedback loop in logs/feedback.jsonl (drift indicator + hard
    example mining per re-labeling futuro)
  - Drift detection via alibi-detect KSDrift (monitor.py)
  - Input validation: formato, dimensioni, dimensione file
  - Guardrails con codici HTTP espliciti (400, 413, 503)
"""

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field
from torchvision import transforms

from src.monitor import check_drift, load_detector

# ── Costanti ───────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_DIMENSION_PX = 32
MAX_DIMENSION_PX = 4096
ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}
BLUR_KERNEL_SIZE = 51  # Gaussian: kernel dispari, piu' grande = piu' sfocatura
PIXELATE_BLOCK = 20  # Pixelate: dimensione blocco in pixel
BLUR_TYPES = {"gaussian", "pixelate", "blackout"}

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.jsonl"
FEEDBACK_FILE = LOG_DIR / "feedback.jsonl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_prediction(record: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def log_feedback(record: dict):
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Modello ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "experiments" / "best.pt"
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, detector

    if not MODEL_PATH.exists():
        logger.warning(f"Pesi non trovati in {MODEL_PATH}. /predict e /blur non funzioneranno.")
    else:
        m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        m.to(DEVICE)
        m.eval()
        model = m
        logger.info(f"Modello caricato da {MODEL_PATH} su {DEVICE}")

    try:
        detector = load_detector()
        logger.info("Drift detector caricato.")
    except FileNotFoundError as e:
        logger.warning(f"Drift detector non disponibile: {e}")
        detector = None

    yield


app = FastAPI(title="Privacy Blurrer", version="1.0.0", lifespan=lifespan)

# CORS: consente chiamate dal frontend in sviluppo (qualsiasi porta localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Transform ──────────────────────────────────────────────────────────────────
preprocess_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]
)


# ── Validazione ────────────────────────────────────────────────────────────────
def validate_image(contents: bytes, filename: str) -> Image.Image:
    """Valida l'immagine caricata. Lancia HTTPException in caso di errore."""

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File troppo grande: {len(contents)/(1024*1024):.1f}MB. "
                f"Max: {MAX_FILE_SIZE_MB}MB."
            ),
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail=f"File non valido o corrotto: '{filename}'.")

    fmt = image.format or ""
    if fmt.upper() not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato non supportato: '{fmt}'. "
                f"Formati ammessi: {', '.join(ALLOWED_FORMATS)}."
            ),
        )

    w, h = image.size
    if w < MIN_DIMENSION_PX or h < MIN_DIMENSION_PX:
        raise HTTPException(
            status_code=400,
            detail=f"Immagine troppo piccola: {w}x{h}px. Minimo: {MIN_DIMENSION_PX}px per lato.",
        )
    if w > MAX_DIMENSION_PX or h > MAX_DIMENSION_PX:
        raise HTTPException(
            status_code=400,
            detail=f"Immagine troppo grande: {w}x{h}px. Massimo: {MAX_DIMENSION_PX}px per lato.",
        )

    return image


# ── Inferenza condivisa ────────────────────────────────────────────────────────
def run_inference(image: Image.Image):
    """
    Esegue preprocessing + inferenza su un'immagine PIL.

    Restituisce:
        binary_mask  : np.ndarray (H, W) uint8, valori 0/255, alla dimensione originale
        drift_result : dict con is_drift, drift_score, features
        preprocess_ms: tempo preprocessing (ms)
        inference_ms : tempo inferenza (ms)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modello non caricato.")

    original_w, original_h = image.size

    t0 = time.time()
    tensor = preprocess_transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    preprocess_ms = round((time.time() - t0) * 1000, 2)

    drift_result = {"is_drift": None, "drift_score": None, "features": None}
    if detector is not None:
        try:
            drift_result = check_drift(tensor.squeeze(0), detector=detector)
        except Exception as e:
            logger.warning(f"Drift detection fallita: {e}")

    t1 = time.time()
    with torch.no_grad():
        output = model(tensor)
        mask_prob = torch.sigmoid(output).squeeze()
        binary_mask = (mask_prob > 0.5).cpu().numpy().astype(np.uint8) * 255
    inference_ms = round((time.time() - t1) * 1000, 2)

    mask_pil = Image.fromarray(binary_mask).resize((original_w, original_h), Image.NEAREST)
    binary_mask = np.array(mask_pil)

    return binary_mask, drift_result, preprocess_ms, inference_ms


# ── Funzioni di anonimizzazione ────────────────────────────────────────────────


def _gaussian(img_np: np.ndarray) -> np.ndarray:
    k = BLUR_KERNEL_SIZE
    return cv2.GaussianBlur(img_np, (k, k), sigmaX=0)


def _pixelate(img_np: np.ndarray) -> np.ndarray:
    h, w = img_np.shape[:2]
    block = PIXELATE_BLOCK
    small = cv2.resize(
        img_np, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _blackout(img_np: np.ndarray) -> np.ndarray:
    return np.zeros_like(img_np)


_BLUR_FN = {
    "gaussian": _gaussian,
    "pixelate": _pixelate,
    "blackout": _blackout,
}


def apply_blur(
    image: Image.Image, binary_mask: np.ndarray, blur_type: str = "gaussian"
) -> Image.Image:
    """
    Anonimizza le persone rilevate nella maschera.

    Args:
        image       : immagine originale PIL RGB
        binary_mask : np.ndarray (H, W), valori 0 o 255
        blur_type   : "gaussian" | "pixelate" | "blackout"

    Returns:
        Immagine PIL RGB con le persone anonimizzate.
    """
    fn = _BLUR_FN.get(blur_type, _gaussian)
    img_np = np.array(image.convert("RGB"))
    anon = fn(img_np)
    mask_bool = binary_mask == 255
    mask_3ch = np.stack([mask_bool, mask_bool, mask_bool], axis=-1)
    result_np = np.where(mask_3ch, anon, img_np).astype(np.uint8)
    return Image.fromarray(result_np)


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Restituisce la maschera binaria (PNG bianco/nero).
    Bianco (255) = persona, Nero (0) = sfondo.
    """
    contents = await file.read()

    try:
        image = validate_image(contents, file.filename)
    except HTTPException as e:
        log_prediction(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoint": "/predict",
                "filename": file.filename,
                "validation_passed": False,
                "rejection_reason": e.detail,
            }
        )
        raise

    binary_mask, drift_result, preprocess_ms, inference_ms = run_inference(image)
    mask_coverage = round(float((binary_mask > 0).mean()), 4)

    log_prediction(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": "/predict",
            "filename": file.filename,
            "original_size": list(image.size),
            "mask_coverage": mask_coverage,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "latency_ms": round(preprocess_ms + inference_ms, 2),
            "validation_passed": True,
            "drift": drift_result,
        }
    )

    _log_drift(file.filename, "/predict", drift_result, mask_coverage, inference_ms)

    return Response(content=_pil_to_png_bytes(Image.fromarray(binary_mask)), media_type="image/png")


@app.post("/blur")
async def blur(
    file: UploadFile = File(...),
    blur_type: str = Query(
        default="gaussian", description="Tipo di anonimizzazione: gaussian | pixelate | blackout"
    ),
):
    """
    Restituisce l'immagine originale con le persone anonimizzate.

    Query param:
        blur_type: gaussian (default) | pixelate | blackout
    """
    if blur_type not in BLUR_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"blur_type non valido: '{blur_type}'. "
                f"Valori ammessi: {', '.join(sorted(BLUR_TYPES))}."
            ),
        )

    contents = await file.read()

    try:
        image = validate_image(contents, file.filename)
    except HTTPException as e:
        log_prediction(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoint": "/blur",
                "filename": file.filename,
                "validation_passed": False,
                "rejection_reason": e.detail,
            }
        )
        raise

    binary_mask, drift_result, preprocess_ms, inference_ms = run_inference(image)
    result_image = apply_blur(image, binary_mask, blur_type=blur_type)
    mask_coverage = round(float((binary_mask > 0).mean()), 4)

    log_prediction(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": "/blur",
            "filename": file.filename,
            "original_size": list(image.size),
            "mask_coverage": mask_coverage,
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "latency_ms": round(preprocess_ms + inference_ms, 2),
            "validation_passed": True,
            "blur_type": blur_type,
            "drift": drift_result,
        }
    )

    _log_drift(
        file.filename, f"/blur?blur_type={blur_type}", drift_result, mask_coverage, inference_ms
    )

    return Response(content=_pil_to_png_bytes(result_image), media_type="image/png")


def _log_drift(filename, endpoint, drift_result, mask_coverage, inference_ms):
    if drift_result["is_drift"]:
        logger.warning(
            f"DRIFT DETECTED | endpoint={endpoint} | file={filename} | "
            f"score={drift_result['drift_score']}"
        )
    else:
        logger.info(
            f"OK | endpoint={endpoint} | file={filename} | "
            f"coverage={mask_coverage} | inference={inference_ms}ms"
        )


# ── Feedback loop ──────────────────────────────────────────────────────────────
class FeedbackIn(BaseModel):
    filename: str = Field(..., max_length=256)
    action: str = Field(..., pattern="^(predict|gaussian|pixelate|blackout)$")
    rating: str = Field(..., pattern="^(good|bad)$")


@app.post("/feedback")
def feedback(payload: FeedbackIn):
    """Raccoglie il giudizio dell'utente sull'output di una richiesta."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": payload.filename,
        "action": payload.action,
        "rating": payload.rating,
    }
    log_feedback(record)
    logger.info(
        f"FEEDBACK | file={payload.filename} action={payload.action} rating={payload.rating}"
    )
    return {"ok": True}


@app.get("/feedback/stats")
def feedback_stats():
    """Aggregato dei feedback ricevuti finora."""
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "positive": 0, "percentage": None}

    total = 0
    positive = 0
    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if rec.get("rating") == "good":
                positive += 1

    percentage = round(100 * positive / total, 1) if total > 0 else None
    return {"total": total, "positive": positive, "percentage": percentage}
