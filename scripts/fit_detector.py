"""
fit_detector.py - One-shot script to fit the KSDrift detector on the training set.

Run from project root:
    conda run -n <env> python scripts/fit_detector.py

Reads preprocessed training images from data/processed/train/,
extracts (mean_R, mean_G, mean_B) features, fits a KSDrift detector,
and saves it to detector.pkl at the project root.
"""

import sys
from pathlib import Path

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from monitor import fit_detector, DETECTOR_PATH

TRAIN_IMG_DIR = Path("data/processed/train/images")
IMG_SIZE = 256


def load_image_as_tensor(path: Path) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # -> (C, H, W) in [0, 1]
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def main():
    if not TRAIN_IMG_DIR.exists():
        print(f"ERROR: Training image directory not found: {TRAIN_IMG_DIR}")
        print("Make sure you have run src/preprocess.py first.")
        sys.exit(1)

    image_paths = sorted(TRAIN_IMG_DIR.glob("*.png"))
    if not image_paths:
        print(f"ERROR: No .png images found in {TRAIN_IMG_DIR}")
        sys.exit(1)

    print(f"Found {len(image_paths)} training images. Extracting features...")

    features = []
    for path in image_paths:
        tensor = load_image_as_tensor(path)
        arr = tensor.numpy()  # (C, H, W)
        features.append([arr[0].mean(), arr[1].mean(), arr[2].mean()])

    reference_features = np.array(features, dtype=np.float32)
    print(f"Feature matrix shape: {reference_features.shape}")
    print(f"Mean features (R, G, B): {reference_features.mean(axis=0).round(4)}")

    print(f"Fitting KSDrift detector...")
    fit_detector(reference_features, save_path=DETECTOR_PATH)
    print(f"Detector saved to: {DETECTOR_PATH}")


if __name__ == "__main__":
    main()
