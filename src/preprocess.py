"""
preprocess.py

Legge le immagini grezze da data/raw/ e salva le versioni preprocessate
in data/processed/, pronte per il training.

Uso:
    python src/preprocess.py --input data/raw --output data/processed
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import albumentations as A


TRANSFORM = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
], additional_targets={"mask": "mask"})


def preprocess(input_dir: str, output_dir: str):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    image_paths = sorted((input_dir / "images").glob("*.png"))

    if not image_paths:
        print(f"Nessuna immagine trovata in {input_dir / 'images'}")
        return

    print(f"Preprocessing {len(image_paths)} immagini...")

    for img_path in image_paths:
        mask_path = input_dir / "masks" / (img_path.stem + ".png")

        if not mask_path.exists():
            print(f"  [SKIP] maschera mancante per {img_path.name}")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = (np.array(Image.open(mask_path).convert("L")) > 127).astype(np.uint8) * 255

        out   = TRANSFORM(image=image, mask=mask)

        Image.fromarray(out["image"]).save(output_dir / "images" / img_path.name)
        Image.fromarray(out["mask"]).save(output_dir / "masks"  / (img_path.stem + ".png"))

    print(f"Salvato in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Privacy Blurrer dataset")
    parser.add_argument("--input",  type=str, default="data/raw",       help="Cartella input con images/ e masks/")
    parser.add_argument("--output", type=str, default="data/processed", help="Cartella output")
    args = parser.parse_args()
    preprocess(args.input, args.output)
