"""
Evaluation tests.

Verificano la qualita' del modello "in produzione" (`experiments/best.pt`)
contro soglie minime su un val set fissato. Se la qualita' regredisce sotto
soglia, il test fallisce e blocca il deploy del nuovo checkpoint.

Skipped se best.pt o data/processed/val/ non esistono (CI senza training).
"""

from pathlib import Path

import pytest
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.dataset import PersonSegmentationDataset
from src.train import iou_score

BEST_PT = Path("experiments/best.pt")
VAL_DIR = Path("data/processed/val")
MIN_VAL_IOU = 0.80  # misurato 0.8382 al 2026-04-25, threshold = misura - 0.04


@pytest.mark.skipif(
    not BEST_PT.exists() or not VAL_DIR.exists(),
    reason="best.pt o data/processed/val/ non disponibili",
)
def test_model_meets_min_iou_threshold():
    """Regression guard: il modello attuale deve superare MIN_VAL_IOU sul val set."""
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    model.load_state_dict(torch.load(BEST_PT, map_location="cpu", weights_only=True))
    model.eval()

    val_ds = PersonSegmentationDataset(VAL_DIR)
    loader = DataLoader(val_ds, batch_size=8)

    ious = []
    with torch.no_grad():
        for x, y in loader:
            ious.append(iou_score(model(x), y))
    mean_iou = sum(ious) / len(ious)

    assert mean_iou >= MIN_VAL_IOU, (
        f"Model regression: mean val IoU {mean_iou:.4f} < threshold {MIN_VAL_IOU}. "
        f"Ricontrolla il checkpoint o aggiorna la soglia in tests/test_evaluation.py."
    )
