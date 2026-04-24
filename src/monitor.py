"""
monitor.py - Drift detection per Privacy Blurrer.

Implementa un detector di drift custom senza dipendenze esterne (solo numpy).

Approccio:
- Fit: si salvano le distribuzioni delle medie R, G, B del training set
  (N valori float, uno per immagine).
- Check: per ogni canale si verifica se la media dell'immagine corrente
  cade fuori dall'intervallo [p_low, p_high] della distribuzione di riferimento.
  Se almeno un canale e' fuori soglia -> drift rilevato.
"""

import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DETECTOR_PATH   = Path(__file__).parent.parent / "experiments" / "detector.json"
P_VAL_THRESHOLD = 0.05   # valori fuori dal 2.5%-97.5% percentile = drift


def extract_features(image_tensor) -> np.ndarray:
    arr      = image_tensor.cpu().numpy()
    features = np.array([[arr[0].mean(), arr[1].mean(), arr[2].mean()]])
    return features.astype(np.float32)


def fit_detector(reference_features: np.ndarray, save_path: Path = DETECTOR_PATH):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    alpha = P_VAL_THRESHOLD / 2

    payload = {
        "n_samples":       int(reference_features.shape[0]),
        "n_features":      int(reference_features.shape[1]),
        "percentile_low":  np.percentile(reference_features, alpha * 100, axis=0).tolist(),
        "percentile_high": np.percentile(reference_features, (1 - alpha) * 100, axis=0).tolist(),
        "mean_ref":        reference_features.mean(axis=0).tolist(),
        "std_ref":         reference_features.std(axis=0).tolist(),
    }

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        f"Detector fittato su {payload['n_samples']} campioni, salvato in {save_path}."
    )
    return payload


def load_detector(path: Path = DETECTOR_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Detector non trovato in {path}. "
            "Esegui prima: python scripts/fit_detector.py"
        )
    with open(path, "r") as f:
        payload = json.load(f)
    logger.info(f"Detector caricato da {path} ({payload['n_samples']} campioni di riferimento)")
    return payload


def check_drift(image_tensor, detector=None) -> dict:
    """
    Verifica se l'immagine corrente e' fuori dalla distribuzione di training.

    Per ogni canale (R, G, B) confronta la media dell'immagine corrente
    con il range [percentile_low, percentile_high] del training set.
    Se almeno un canale e' fuori range -> drift.

    Returns:
        dict: is_drift (bool), drift_score (float), features (list)
    """
    if detector is None:
        detector = load_detector()

    p_low   = np.array(detector["percentile_low"],  dtype=np.float32)
    p_high  = np.array(detector["percentile_high"], dtype=np.float32)
    mean_r  = np.array(detector["mean_ref"],        dtype=np.float32)
    std_r   = np.array(detector["std_ref"],         dtype=np.float32)

    features    = extract_features(image_tensor)[0]
    out_of_range = (features < p_low) | (features > p_high)
    is_drift    = bool(out_of_range.any())
    drift_score = float(np.mean(np.abs(features - mean_r) / (std_r + 1e-8)))

    return {
        "is_drift":    is_drift,
        "drift_score": round(drift_score, 4),
        "features":    features.tolist(),
    }