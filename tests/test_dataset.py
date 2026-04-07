"""
Unit tests for PersonSegmentationDataset.
"""
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader

from src.dataset import PersonSegmentationDataset


@pytest.fixture
def fake_dataset(tmp_path):
    """Create a minimal fake dataset (3 image/mask pairs)."""
    img_dir  = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    for i in range(3):
        img  = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        mask = Image.fromarray(np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255)
        img.save(img_dir / f"img_{i:02d}.jpg")
        mask.save(mask_dir / f"img_{i:02d}.png")

    return tmp_path


def test_dataset_length(fake_dataset):
    ds = PersonSegmentationDataset(fake_dataset)
    assert len(ds) == 3


def test_dataset_item_shapes(fake_dataset):
    ds = PersonSegmentationDataset(fake_dataset)
    image, mask = ds[0]
    assert image.shape == (3, 64, 64), f"Unexpected image shape: {image.shape}"
    assert mask.shape  == (1, 64, 64), f"Unexpected mask shape: {mask.shape}"


def test_mask_binary(fake_dataset):
    ds = PersonSegmentationDataset(fake_dataset)
    _, mask = ds[0]
    unique = mask.unique().tolist()
    assert all(v in [0.0, 1.0] for v in unique), f"Mask not binary: {unique}"


def test_image_range(fake_dataset):
    ds = PersonSegmentationDataset(fake_dataset)
    image, _ = ds[0]
    assert image.min() >= 0.0 and image.max() <= 1.0, "Image not in [0, 1]"


def test_dataloader_batch(fake_dataset):
    ds     = PersonSegmentationDataset(fake_dataset)
    loader = DataLoader(ds, batch_size=2)
    images, masks = next(iter(loader))
    assert images.shape[0] == 2
    assert masks.shape[0]  == 2
