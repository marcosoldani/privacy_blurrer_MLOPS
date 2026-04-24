"""
generate_manifest.py

Legge i file reali da data/raw/ e genera data/manifests/dataset_v1.0.json con i nomi corretti.
Split 80% train / 20% val.

Uso (dalla root del progetto):
    python scripts/generate_manifest.py
    python scripts/generate_manifest.py --data_dir data/raw --output data/manifests/dataset_v1.0.json --val_ratio 0.2
"""

import argparse
import json
from pathlib import Path
from datetime import date


def generate(data_dir: str, output: str, val_ratio: float):
    data_dir = Path(data_dir)
    images   = sorted((data_dir / "images").glob("*.png"))

    if not images:
        print(f"Nessuna immagine trovata in {data_dir / 'images'}")
        return

    # verifica che ogni immagine abbia la maschera corrispondente
    samples = []
    for img_path in images:
        mask_path = data_dir / "masks" / (img_path.stem + ".png")
        if mask_path.exists():
            samples.append({
                "image": f"images/{img_path.name}",
                "mask":  f"masks/{img_path.stem}.png",
            })
        else:
            print(f"  [WARN] maschera mancante per {img_path.name}, saltato")

    n_val   = max(1, int(len(samples) * val_ratio))
    n_train = len(samples) - n_val

    manifest = {
        "version":       "1.0",
        "created":       str(date.today()),
        "description":   "Supervisely Filtered Person Segmentation dataset - first release",
        "source":        "https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset",
        "total_samples": len(samples),
        "splits": {
            "train": samples[:n_train],
            "val":   samples[n_train:],
        }
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest generato: {output}")
    print(f"  Totale: {len(samples)} campioni")
    print(f"  Train:  {n_train}")
    print(f"  Val:    {n_val}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dataset_v1.0.json dai file reali")
    parser.add_argument("--data_dir",  type=str,   default="data/raw",          help="Cartella con images/ e masks/")
    parser.add_argument("--output",    type=str,   default="data/manifests/dataset_v1.0.json", help="Path output JSON")
    parser.add_argument("--val_ratio", type=float, default=0.2,                 help="Proporzione validation set")
    args = parser.parse_args()
    generate(args.data_dir, args.output, args.val_ratio)
