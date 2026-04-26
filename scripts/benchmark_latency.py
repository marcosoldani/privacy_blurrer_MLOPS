"""
benchmark_latency.py - Misura la latenza di inference del modello U-Net.

Uso (dalla root del progetto):
    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --runs 100 --warmup 10

Stampa P50/P95/P99 di n inference su CPU su un'immagine 256x256 reale.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torchvision import transforms

MODEL_PATH = Path("experiments/best.pt")
TEST_IMG_DIR = Path("data/processed/test/images")
IMG_SIZE = 256


def benchmark(runs: int = 50, warmup: int = 5):
    if not MODEL_PATH.exists():
        raise SystemExit(f"Checkpoint non trovato: {MODEL_PATH}")

    test_images = sorted(TEST_IMG_DIR.glob("*.png"))
    if not test_images:
        raise SystemExit(f"Nessuna immagine in {TEST_IMG_DIR}")

    device = torch.device("cpu")
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )

    img = Image.open(test_images[0]).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    print(f"Modello:  {MODEL_PATH}")
    print(f"Device:   {device}")
    print(f"Input:    {tuple(x.shape)}")
    print(f"Warmup:   {warmup} run")
    print(f"Misure:   {runs} run")
    print()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        times_ms = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(x)
            times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times_ms)
    print(f"Mean:     {arr.mean():.2f} ms")
    print(f"Std:      {arr.std():.2f} ms")
    print(f"Min:      {arr.min():.2f} ms")
    print(f"P50:      {np.percentile(arr, 50):.2f} ms")
    print(f"P95:      {np.percentile(arr, 95):.2f} ms")
    print(f"P99:      {np.percentile(arr, 99):.2f} ms")
    print(f"Max:      {arr.max():.2f} ms")
    print(f"Throughput: {1000 / arr.mean():.1f} req/s (single-thread)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark latenza di inference")
    parser.add_argument("--runs", type=int, default=50, help="Numero di run di misura")
    parser.add_argument("--warmup", type=int, default=5, help="Run di warmup")
    args = parser.parse_args()
    benchmark(runs=args.runs, warmup=args.warmup)
