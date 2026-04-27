"""
quantize_model.py - Quantizzazione del modello U-Net.

Esplora due strategie di compressione del checkpoint PyTorch:
  1. Dynamic INT8 quantization (Linear layers)
  2. FP16 conversion (tutti i layer, half precision)

Per ciascuna stampa: size su disco, latency P50 su CPU.

Uso (dalla root del progetto):
    python scripts/quantize_model.py
"""

import time
import warnings
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

warnings.filterwarnings("ignore", category=UserWarning)

ORIG_PATH = Path("experiments/best.pt")
INT8_PATH = Path("experiments/best_int8.pt")
FP16_PATH = Path("experiments/best_fp16.pt")
IMG_SIZE = 256
BENCH_RUNS = 30


def load_baseline_model() -> torch.nn.Module:
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    model.load_state_dict(torch.load(ORIG_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def benchmark(model: torch.nn.Module, x: torch.Tensor) -> float:
    """Ritorna la latency mediana in ms su BENCH_RUNS run."""
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        times = []
        for _ in range(BENCH_RUNS):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def main():
    if not ORIG_PATH.exists():
        raise SystemExit(f"Checkpoint non trovato: {ORIG_PATH}")

    x_fp32 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # ── Baseline FP32 ─────────────────────────────────────────────────────────
    base = load_baseline_model()
    base_size = file_size_mb(ORIG_PATH)
    base_latency = benchmark(base, x_fp32)
    print(f"FP32 baseline:  {base_size:6.1f} MB   latency P50 = {base_latency:6.1f} ms")

    # ── Dynamic INT8 (solo Linear layers) ─────────────────────────────────────
    int8_model = torch.quantization.quantize_dynamic(
        load_baseline_model(),
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(int8_model.state_dict(), INT8_PATH)
    int8_size = file_size_mb(INT8_PATH)
    int8_latency = benchmark(int8_model, x_fp32)
    print(
        f"Dynamic INT8:   {int8_size:6.1f} MB   latency P50 = {int8_latency:6.1f} ms   "
        f"(size {100*int8_size/base_size:5.1f}%, latency {100*int8_latency/base_latency:5.1f}%)"
    )

    # ── FP16 (half precision, tutti i layer) ──────────────────────────────────
    fp16_model = load_baseline_model().half()
    torch.save(fp16_model.state_dict(), FP16_PATH)
    fp16_size = file_size_mb(FP16_PATH)
    x_fp16 = x_fp32.half()
    fp16_latency = benchmark(fp16_model, x_fp16)
    print(
        f"FP16:           {fp16_size:6.1f} MB   latency P50 = {fp16_latency:6.1f} ms   "
        f"(size {100*fp16_size/base_size:5.1f}%, latency {100*fp16_latency/base_latency:5.1f}%)"
    )

    print()
    print("Note:")
    print("  - Dynamic INT8 quantizza solo i layer Linear: U-Net + ResNet34 e'")
    print("    quasi tutto Conv2D, quindi l'effetto sulla size e' minimo.")
    print("    Per guadagni reali serve static quantization (calibrazione richiesta)")
    print("    o ONNX Runtime quantization su tutti i Conv.")
    print("  - FP16 dimezza la size in modo affidabile. La latency su CPU puo'")
    print("    essere uguale o peggiore (kernel non ottimizzati); su GPU/MPS e' piu' veloce.")


if __name__ == "__main__":
    main()
