"""
Carica coppie immagine/maschera dal dataset Supervisely.

Struttura attesa:
    <root>/images/*.{jpg,png}
    <root>/masks/*.png   (255 = persona, 0 = sfondo)
"""

import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class PersonSegmentationDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_dir = Path(root) / "images"
        self.mask_dir = Path(root) / "masks"
        self.transform = transform
        self.images = sorted(
            list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.float32)  # binarizza: 1 = persona

        if self.transform:
            out = self.transform(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask
