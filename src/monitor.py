"""
monitor.py - Drift detection for Privacy Blurrer inference pipeline.

Uses Alibi Detect KSDrift on 3 lightweight image statistics:
mean R channel, mean G channel, mean B channel.
"""

import pickle
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

DETECTOR_PATH = Path(__file__).parent.parent / "detector.pkl"


def extract_features(image_tensor) -> np.ndarray:
    """
    Extract 3 scalar features from a CHW float tensor (values in [0,1]).
    Returns shape (1, 3): [mean_R, mean_G, mean_B].
    """
    arr = image_tensor.cpu().numpy()  # (C, H, W)
    features = np.array([[arr[0].mean(), arr[1].mean(), arr[2].mean()]])
    return features.astype(np.float32)


def fit_detector(reference_features: np.ndarray, save_path: Path = DETECTOR_PATH):
    """
    Fit a KSDrift detector on reference features and save to disk.

    Args:
        reference_features: shape (N, 3) array of training image features.
        save_path: where to persist the fitted detector.
    """
    from alibi_detect.cd import KSDrift

    detector = KSDrift(reference_features, p_val=0.05)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(detector, f)
    logger.info(f"Detector fitted on {len(reference_features)} samples and saved to {save_path}")
    return detector


def load_detector(path: Path = DETECTOR_PATH):
    """Load a previously fitted detector from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Detector not found at {path}. "
            "Run scripts/fit_detector.py first."
        )
    with open(path, "rb") as f:
        detector = pickle.load(f)
    logger.info(f"Detector loaded from {path}")
    return detector


def check_drift(image_tensor, detector=None) -> dict:
    """
    Run drift detection on a single image tensor.

    Args:
        image_tensor: torch.Tensor of shape (C, H, W), values in [0, 1].
        detector: optional pre-loaded detector (loaded from disk if None).

    Returns:
        dict with keys:
            is_drift (bool): True if drift detected.
            drift_score (float): mean p-value across KS tests (lower = more drift).
            features (list): the 3 extracted feature values [mean_R, mean_G, mean_B].
    """
    if detector is None:
        detector = load_detector()

    features = extract_features(image_tensor)
    result = detector.predict(features)

    is_drift = bool(result["data"]["is_drift"])
    p_vals = result["data"]["p_val"]
    drift_score = float(np.mean(p_vals))

    return {
        "is_drift": is_drift,
        "drift_score": round(drift_score, 4),
        "features": features[0].tolist(),
    }
