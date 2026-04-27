"""
Test sulle funzioni del drift detector in src/monitor.py.

Coprono extract_features, fit_detector (incluso il salvataggio su disco e il
roundtrip pickle/load), e il path di errore di load_detector quando il file
non esiste.
"""

import numpy as np
import pytest
import torch

from src.monitor import (
    extract_features,
    fit_detector,
    load_detector,
)


def test_extract_features_shape_and_dtype():
    """extract_features ritorna (1, 3) float32 con valori in [0, 1]."""
    tensor = torch.zeros((3, 64, 64))
    tensor[0] = 0.2
    tensor[1] = 0.5
    tensor[2] = 0.8

    features = extract_features(tensor)

    assert features.shape == (1, 3)
    assert features.dtype == np.float32
    np.testing.assert_allclose(features[0], [0.2, 0.5, 0.8], atol=1e-6)


def test_fit_detector_saves_and_loads(tmp_path):
    """fit_detector salva un pickle valido, load_detector lo recupera."""
    rng = np.random.default_rng(seed=0)
    reference = rng.uniform(0.3, 0.7, size=(50, 3)).astype(np.float32)
    save_path = tmp_path / "detector.pkl"

    detector = fit_detector(reference, save_path=save_path)

    assert detector.n == 50
    assert detector.n_features == 3
    assert save_path.exists()

    loaded = load_detector(path=save_path)
    assert loaded.n == 50
    assert loaded.n_features == 3

    sample = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    result = loaded.predict(sample)
    assert "data" in result
    assert "is_drift" in result["data"]


def test_load_detector_raises_on_missing_file(tmp_path):
    """load_detector lancia FileNotFoundError se il pickle non esiste."""
    missing_path = tmp_path / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError, match="Detector non trovato"):
        load_detector(path=missing_path)

def test_check_drift_loads_default_detector(tmp_path, monkeypatch):
    """check_drift senza detector esplicito → carica quello in DETECTOR_PATH."""
    rng = np.random.default_rng(seed=0)
    reference = rng.uniform(0.3, 0.7, size=(50, 3)).astype(np.float32)
    save_path = tmp_path / "detector.pkl"
    fit_detector(reference, save_path=save_path)

    # Punta DETECTOR_PATH al pickle appena creato
    monkeypatch.setattr("src.monitor.DETECTOR_PATH", save_path)

    from src.monitor import check_drift

    tensor = torch.zeros((3, 32, 32))
    tensor[0] = 0.5
    tensor[1] = 0.5
    tensor[2] = 0.5

    result = check_drift(tensor)  # detector=None → trigger del default
    assert "is_drift" in result
    assert "drift_score" in result
    assert "features" in result