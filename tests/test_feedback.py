"""
Test sul feedback loop.

Coprono entrambi gli endpoint del feedback:
  - POST /feedback           : scrittura di un giudizio dell'utente
  - GET  /feedback/stats     : aggregato dei feedback ricevuti

I test usano monkeypatch per ridirigere FEEDBACK_FILE in tmp_path, in modo da
non sporcare lo stato reale di logs/feedback.jsonl durante i test.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient con FEEDBACK_FILE puntato a un file temporaneo isolato."""
    fake_feedback = tmp_path / "feedback.jsonl"
    monkeypatch.setattr("src.app.FEEDBACK_FILE", fake_feedback)
    with TestClient(app) as c:
        yield c, fake_feedback


# ── POST /feedback ────────────────────────────────────────────────────────────


def test_feedback_post_valid_payload(client):
    """Happy path: payload valido → 200, ok=true, riga scritta nel JSONL."""
    c, feedback_file = client
    payload = {"filename": "test.png", "action": "gaussian", "rating": "good"}

    r = c.post("/feedback", json=payload)

    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert feedback_file.exists()

    with open(feedback_file, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["filename"] == "test.png"
    assert record["action"] == "gaussian"
    assert record["rating"] == "good"
    assert "timestamp" in record


def test_feedback_post_rejects_invalid_rating(client):
    """Pydantic guardrail: rating fuori whitelist → 422."""
    c, _ = client
    payload = {"filename": "test.png", "action": "gaussian", "rating": "maybe"}

    r = c.post("/feedback", json=payload)
    assert r.status_code == 422


def test_feedback_post_rejects_invalid_action(client):
    """Pydantic guardrail: action fuori whitelist → 422."""
    c, _ = client
    payload = {"filename": "test.png", "action": "rainbow", "rating": "good"}

    r = c.post("/feedback", json=payload)
    assert r.status_code == 422


# ── GET /feedback/stats ───────────────────────────────────────────────────────


def test_feedback_stats_empty_when_no_file(client):
    """Nessun feedback ricevuto → stats vuoti."""
    c, _ = client
    r = c.get("/feedback/stats")

    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["positive"] == 0
    assert body["percentage"] is None


def test_feedback_stats_aggregates_correctly(client):
    """Dopo 3 feedback (2 good, 1 bad) → total=3, positive=2, percentage=66.7."""
    c, _ = client

    c.post("/feedback", json={"filename": "a.png", "action": "gaussian", "rating": "good"})
    c.post("/feedback", json={"filename": "b.png", "action": "pixelate", "rating": "good"})
    c.post("/feedback", json={"filename": "c.png", "action": "blackout", "rating": "bad"})

    r = c.get("/feedback/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert body["positive"] == 2
    assert body["percentage"] == 66.7