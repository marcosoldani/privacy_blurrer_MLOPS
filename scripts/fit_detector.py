"""
Fitta il detector di drift sulle immagini del training set e salva il risultato.

Uso:
    python scripts/fit_detector.py

Legge i dati preprocessati da data/processed/train/images/
ed estrae le feature (mean R, G, B) da ogni immagine.
Salva il detector in experiments/detector.pkl (KSDrift serializzato).
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Permette di importare src/ dalla root del progetto
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch  # noqa: E402
from src.monitor import extract_features, fit_detector  # noqa: E402

TRAIN_DIR = Path("data/processed/train/images")
DETECTOR_PATH = Path("experiments/detector.pkl")


def load_image_tensor(img_path: Path):
    """Carica un'immagine e la converte in tensore CHW float [0,1]."""
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)
    return tensor


def main():
    images = sorted(TRAIN_DIR.glob("*.png"))
    if not images:
        images = sorted(TRAIN_DIR.glob("*.jpg"))

    if not images:
        print(f"Nessuna immagine trovata in {TRAIN_DIR}")
        sys.exit(1)

    print(f"Estrazione feature da {len(images)} immagini di training...")

    all_features = []
    for i, img_path in enumerate(images):
        tensor = load_image_tensor(img_path)
        features = extract_features(tensor)  # (1, 3)
        all_features.append(features[0])

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(images)}")

    reference = np.stack(all_features, axis=0)  # (N, 3)
    print(f"Mean features (R, G, B): {reference.mean(axis=0).round(4)}")

    print("Fitting KSDrift detector...")
    fit_detector(reference, save_path=DETECTOR_PATH)
    print(f"Detector salvato in: {DETECTOR_PATH.resolve()}")


if __name__ == "__main__":
    main()
