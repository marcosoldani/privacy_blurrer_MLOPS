"""
Privacy Blurrer — FastAPI serving endpoint.

Avvio locale:
    uvicorn src.app:app --host 0.0.0.0 --port 8000

Endpoint:
    POST /predict   — accetta un'immagine, restituisce la maschera binaria in PNG
    GET  /health    — healthcheck
"""
import io
import torch
import numpy as np
from pathlib import Path
from PIL import Image

import segmentation_models_pytorch as smp
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

# ── Configurazione ────────────────────────────────────────────────────────────
MODEL_PATH = Path("experiments/best.pt")
IMG_SIZE   = (256, 256)
THRESHOLD  = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Caricamento modello (una sola volta all'avvio) ───────────────────────────
model = smp.Unet("resnet34", encoder_weights=None, classes=1)
if MODEL_PATH.exists():
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
else:
    # In assenza del checkpoint (es. test/CI) il modello gira con pesi random
    pass
model.to(DEVICE)
model.eval()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Privacy Blurrer", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Il file deve essere un'immagine.")

    # Lettura e preprocessing
    data  = await file.read()
    img   = Image.open(io.BytesIO(data)).convert("RGB").resize(IMG_SIZE)
    x     = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
    x     = x.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(x)
        mask   = (torch.sigmoid(logits) > THRESHOLD).squeeze().cpu().numpy().astype(np.uint8)

    # Risposta: maschera PNG (0 = sfondo, 255 = persona)
    out_img = Image.fromarray(mask * 255)
    buf     = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)

    return Response(content=buf.read(), media_type="image/png")
