"""
Preprocessa il dataset: ridimensiona le immagini e divide in train/val/test.

Uso:
    python src/preprocess.py
"""

import random
from pathlib import Path
from PIL import Image

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
IMG_SIZE = (256, 256)
SPLIT = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 42


def preprocess(raw_dir=RAW_DIR, out_dir=OUT_DIR, img_size=IMG_SIZE, seed=SEED):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    images = sorted((raw_dir / "images").glob("*.png"))
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])
    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, files in splits.items():
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)
        for img_path in files:
            mask_path = raw_dir / "masks" / (img_path.stem + ".png")

            img = Image.open(img_path).convert("RGB").resize(img_size)
            mask = Image.open(mask_path).convert("L").resize(img_size, Image.NEAREST)

            img.save(out_dir / split / "images" / img_path.name)
            mask.save(out_dir / split / "masks" / (img_path.stem + ".png"))

    for split, files in splits.items():
        print(f"{split:5s}: {len(files)} campioni")
    print(f"Dimensione: {img_size} — salvati in {out_dir}/")


if __name__ == "__main__":
    preprocess()
