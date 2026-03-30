"""
Pipeline (infrastructure) tests: verify the training and inference code
runs without errors on a tiny synthetic dataset.
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image

import segmentation_models_pytorch as smp

from src.dataset import PersonSegmentationDataset
from src.preprocess import preprocess
from src.predict import predict_mask


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def raw_data(tmp_path):
    """Tiny raw dataset: 5 image/mask pairs."""
    img_dir  = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()
    for i in range(5):
        img  = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        mask = Image.fromarray(np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255)
        img.save(img_dir  / f"img_{i:02d}.jpg")
        mask.save(mask_dir / f"img_{i:02d}.png")
    return tmp_path


@pytest.fixture
def processed_data(raw_data, tmp_path):
    out_dir = tmp_path / "processed"
    preprocess(raw_dir=raw_data, out_dir=out_dir, img_size=(64, 64))
    return out_dir


# ── Preprocessing ─────────────────────────────────────────────────────────────

def test_preprocess_creates_splits(processed_data):
    for split in ("train", "val", "test"):
        assert (processed_data / split / "images").exists()
        assert (processed_data / split / "masks").exists()


def test_preprocess_total_count(raw_data, processed_data):
    total = sum(
        len(list((processed_data / split / "images").glob("*.jpg")))
        for split in ("train", "val", "test")
    )
    assert total == 5


# ── Model forward pass ────────────────────────────────────────────────────────

def test_model_forward_pass():
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 64, 64), f"Unexpected output shape: {out.shape}"


def test_model_output_finite():
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all(), "Model output contains NaN or Inf"


# ── Training mini-run ─────────────────────────────────────────────────────────

def test_training_loss_decreases(processed_data):
    """One epoch on a tiny batch — loss must be finite and not explode."""
    ds     = PersonSegmentationDataset(processed_data / "train")
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    model  = smp.Unet("resnet34", encoder_weights=None, classes=1)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = smp.losses.DiceLoss(mode="binary")

    model.train()
    losses = []
    for x, y in loader:
        opt.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert all(np.isfinite(l) for l in losses), "Loss is NaN/Inf during training"


# ── Inference ─────────────────────────────────────────────────────────────────

def test_predict_mask_shape(tmp_path):
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    img_path = tmp_path / "test.jpg"
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_path)

    mask = predict_mask(model, img_path, img_size=(64, 64))
    assert mask.shape == (64, 64)


def test_predict_mask_binary(tmp_path):
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    img_path = tmp_path / "test.jpg"
    Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(img_path)

    mask = predict_mask(model, img_path, img_size=(64, 64))
    unique = np.unique(mask).tolist()
    assert set(unique).issubset({0, 1}), f"Mask not binary: {unique}"
