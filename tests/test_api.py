"""
API tests via FastAPI TestClient.

Coprono validation, guardrails e happy path degli endpoint HTTP:
  - /health              : liveness probe
  - /predict (4 casi)    : payload non-immagine, file oversized, immagine
                           sotto la dimensione minima, happy path
  - /blur    (2 casi)    : blur_type non in whitelist, happy path

Validation tests girano sempre. Happy-path tests sono skippati se best.pt
non e' presente (es. CI senza modello).
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.app import app

BEST_PT_PRESENT = Path("experiments/best.pt").exists()
needs_model = pytest.mark.skipif(
    not BEST_PT_PRESENT, reason="best.pt non disponibile (CI senza modello)"
)


@pytest.fixture(scope="module")
def client():
    """TestClient con lifespan attivo: triggera il caricamento di
    modello + detector come accade in produzione."""
    with TestClient(app) as c:
        yield c


def _png_bytes(size=(64, 64), color=(128, 128, 128)) -> bytes:
    """PNG fittizio in memoria, dimensione configurabile."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Liveness ──────────────────────────────────────────────────────────────────


def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


# ── Validation guardrails (no model needed) ───────────────────────────────────


def test_predict_rejects_non_image_payload(client):
    files = {"file": ("data.txt", b"not an image", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
    assert "non valido" in r.json()["detail"].lower()


def test_predict_rejects_oversized_file(client):
    """File da 11MB > limite 10MB definito in MAX_FILE_SIZE_BYTES."""
    big = b"\x00" * (11 * 1024 * 1024)
    files = {"file": ("big.png", big, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 413
    assert "troppo grande" in r.json()["detail"].lower()


def test_predict_rejects_undersized_image(client):
    """Immagine 16x16 sotto la soglia minima di 32px."""
    tiny = _png_bytes(size=(16, 16))
    files = {"file": ("tiny.png", tiny, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
    assert "troppo piccola" in r.json()["detail"].lower()


def test_blur_rejects_invalid_blur_type(client):
    """`blur_type` deve essere nella whitelist {gaussian, pixelate, blackout}."""
    files = {"file": ("x.png", _png_bytes(), "image/png")}
    r = client.post("/blur?blur_type=disco", files=files)
    assert r.status_code == 400
    assert "blur_type" in r.json()["detail"].lower()


# ── Happy paths (require model) ───────────────────────────────────────────────


@needs_model
def test_predict_returns_png_mask(client):
    files = {"file": ("x.png", _png_bytes(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    assert len(r.content) > 0


@needs_model
def test_blur_gaussian_returns_image(client):
    files = {"file": ("x.png", _png_bytes(), "image/png")}
    r = client.post("/blur?blur_type=gaussian", files=files)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    assert len(r.content) > 0

def test_predict_logs_rejection_on_invalid_payload(client, tmp_path, monkeypatch):
    """Quando /predict rifiuta input, il log strutturato cattura il rejection_reason."""
    fake_log = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("src.app.LOG_FILE", fake_log)

    files = {"file": ("data.txt", b"not an image", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400

    assert fake_log.exists()
    import json
    with open(fake_log) as f:
        rec = json.loads(f.readlines()[-1])
    assert rec["validation_passed"] is False
    assert "rejection_reason" in rec


def test_blur_logs_rejection_on_invalid_payload(client, tmp_path, monkeypatch):
    """Stesso comportamento per /blur."""
    fake_log = tmp_path / "predictions.jsonl"
    monkeypatch.setattr("src.app.LOG_FILE", fake_log)

    files = {"file": ("data.txt", b"not an image", "text/plain")}
    r = client.post("/blur?blur_type=gaussian", files=files)
    assert r.status_code == 400

    assert fake_log.exists()
    import json
    with open(fake_log) as f:
        rec = json.loads(f.readlines()[-1])
    assert rec["validation_passed"] is False


def test_predict_returns_503_when_model_missing(client, monkeypatch):
    """Senza modello caricato, /predict ritorna 503."""
    import io
    from PIL import Image

    monkeypatch.setattr("src.app.model", None)

    img = Image.new("RGB", (64, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    files = {"file": ("x.png", buf.getvalue(), "image/png")}

    r = client.post("/predict", files=files)
    assert r.status_code == 503
