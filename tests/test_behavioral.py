"""
Behavioral + Perturbation tests.

Verificano che il modello si comporti in modo "ragionevole" sotto
trasformazioni degli input che non dovrebbero cambiare la semantica della
maschera:

  - Flip orizzontale: la maschera dell'immagine ribaltata deve essere
    (approssimativamente) il flip della maschera dell'immagine originale.
  - Noise gaussiano lieve: la maschera dovrebbe restare quasi identica.

Aggregati su 10 immagini reali del test set per stabilita' statistica.
Skipped se best.pt o data/processed/test/ non disponibili.
"""

from pathlib import Path

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision import transforms

BEST_PT = Path("experiments/best.pt")
TEST_IMG_DIR = Path("data/processed/test/images")
N_IMAGES = 10
IMG_SIZE = 256

MIN_FLIP_IOU = 0.75  # misurato mean=0.855, threshold = mean - 0.10
MIN_NOISE_IOU = 0.85  # misurato mean=0.958, threshold = mean - 0.10


@pytest.fixture(scope="module")
def model():
    if not BEST_PT.exists():
        pytest.skip(f"Checkpoint non disponibile: {BEST_PT}")
    m = smp.Unet("resnet34", encoder_weights=None, classes=1)
    m.load_state_dict(torch.load(BEST_PT, map_location="cpu", weights_only=True))
    m.eval()
    return m


@pytest.fixture(scope="module")
def test_images():
    if not TEST_IMG_DIR.exists():
        pytest.skip(f"Test images non disponibili: {TEST_IMG_DIR}")
    paths = sorted(TEST_IMG_DIR.glob("*.png"))[:N_IMAGES]
    if not paths:
        pytest.skip(f"Nessuna immagine in {TEST_IMG_DIR}")
    return paths


def _predict_mask(model, img_pil: Image.Image) -> np.ndarray:
    transform = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
    x = transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    return (torch.sigmoid(logits).squeeze().numpy() > 0.5).astype(np.uint8)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = ((a == 1) & (b == 1)).sum()
    union = ((a == 1) | (b == 1)).sum()
    return float(inter) / max(int(union), 1)


def test_horizontal_flip_invariance(model, test_images):
    """Flip orizzontale dell'input deve produrre il flip della maschera."""
    ious = []
    for path in test_images:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)

        mask_orig = _predict_mask(model, img)
        mask_flip_input = _predict_mask(model, Image.fromarray(np.fliplr(arr)))

        ious.append(_iou(mask_flip_input, np.fliplr(mask_orig)))

    mean_iou = float(np.mean(ious))
    assert mean_iou >= MIN_FLIP_IOU, (
        f"Flip invariance IoU regression: mean={mean_iou:.3f} < {MIN_FLIP_IOU}. "
        f"Il modello sta producendo maschere diverse quando l'input e' ribaltato."
    )


def test_gaussian_noise_robustness(model, test_images):
    """Noise gaussiano lieve (std=5/255) non deve distruggere la maschera."""
    rng = np.random.default_rng(seed=0)
    ious = []
    for path in test_images:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)

        mask_clean = _predict_mask(model, img)

        noisy = np.clip(arr.astype(np.float32) + rng.normal(0, 5, arr.shape), 0, 255)
        mask_noisy = _predict_mask(model, Image.fromarray(noisy.astype(np.uint8)))

        ious.append(_iou(mask_clean, mask_noisy))

    mean_iou = float(np.mean(ious))
    assert mean_iou >= MIN_NOISE_IOU, (
        f"Noise robustness IoU regression: mean={mean_iou:.3f} < {MIN_NOISE_IOU}. "
        f"Il modello e' troppo sensibile a piccoli noise gaussiani."
    )
