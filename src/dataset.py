from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PersonSegmentationDataset(Dataset):
    """
    Carica coppie immagine/maschera dal dataset Supervisely.
      data/raw/images/*.png
      data/raw/masks/*.png   (255=persona, 0=sfondo)
    """
    def __init__(self, root, transform=None):
        self.images    = sorted((Path(root) / "images").glob("*.png"))
        self.mask_dir  = Path(root) / "masks"
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")

        image = np.array(Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR))
        mask  = (np.array(Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST)) > 127).astype(np.float32)

        if self.transform:
            out         = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask  = torch.from_numpy(mask).unsqueeze(0)
        return image, mask