"""
monitor.py - Drift detection per Privacy Blurrer.

Implementa drift detection con alibi-detect (KSDrift) sulle statistiche
di primo ordine dei canali RGB.

Approccio:
- Feature: vettore [mean_R, mean_G, mean_B] estratto da ciascuna immagine.
- Fit: KSDrift con p_val=0.05 (correzione Bonferroni di default sui 3 canali)
  addestrato sulle feature del training set.
- Check: il detector applica il test di Kolmogorov-Smirnov canale per canale e
  segnala drift se almeno un canale supera la soglia corretta.
"""

import logging
import pickle
from pathlib import Path

import numpy as np

from alibi_detect.cd import KSDrift

logger = logging.getLogger(__name__)

DETECTOR_PATH = Path(__file__).parent.parent / "experiments" / "detector.pkl"
P_VAL_THRESHOLD = 0.05


def extract_features(image_tensor) -> np.ndarray:
    """Media normalizzata dei canali R, G, B. Shape: (1, 3) float32."""
    arr = image_tensor.cpu().numpy()
    features = np.array([[arr[0].mean(), arr[1].mean(), arr[2].mean()]], dtype=np.float32)
    return features


def fit_detector(reference_features: np.ndarray, save_path: Path = DETECTOR_PATH) -> KSDrift:
    """Fitta un KSDrift sulle feature di riferimento e lo serializza su disco."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    detector = KSDrift(
        x_ref=reference_features.astype(np.float32),
        p_val=P_VAL_THRESHOLD,
    )

    with open(save_path, "wb") as f:
        pickle.dump(detector, f)

    logger.info(
        f"Detector KSDrift fittato su {detector.n} campioni × {detector.n_features} feature, "
        f"salvato in {save_path}."
    )
    return detector


def load_detector(path: Path = DETECTOR_PATH) -> KSDrift:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Detector non trovato in {path}. " "Esegui prima: python scripts/fit_detector.py"
        )
    with open(path, "rb") as f:
        detector = pickle.load(f)
    logger.info(f"Detector KSDrift caricato da {path} ({detector.n} campioni di riferimento)")
    return detector


def check_drift(image_tensor, detector: KSDrift = None) -> dict:
    """
    Applica KSDrift all'immagine corrente.

    Returns:
        dict: {
            is_drift    (bool):  True se almeno un canale e' fuori distribuzione
            drift_score (float): KS distance media sui 3 canali, in [0, 1]
            features    (list):  [mean_R, mean_G, mean_B] dell'immagine
        }
    """
    if detector is None:
        detector = load_detector()

    features = extract_features(image_tensor)
    result = detector.predict(features)["data"]

    return {
        "is_drift": bool(result["is_drift"]),
        "drift_score": round(float(np.mean(result["distance"])), 4),
        "features": features[0].tolist(),
    }
