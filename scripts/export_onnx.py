"""
export_onnx.py - Esporta il checkpoint PyTorch in formato ONNX.

Uso (dalla root del progetto):
    python scripts/export_onnx.py

Output: experiments/best.onnx
Verifica: confronta l'output del modello PyTorch vs quello ONNX su un input
  di prova; deve combaciare entro tolleranza numerica.
"""

import warnings
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch

# Il legacy exporter di torch emette una DeprecationWarning innocua.
warnings.filterwarnings("ignore", category=DeprecationWarning)

PT_PATH = Path("experiments/best.pt")
ONNX_PATH = Path("experiments/best.onnx")
IMG_SIZE = 256
OPSET = 17
ATOL = 5e-4


def export():
    if not PT_PATH.exists():
        raise SystemExit(f"Checkpoint non trovato: {PT_PATH}")

    device = torch.device("cpu")
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    model.load_state_dict(torch.load(PT_PATH, map_location=device, weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print(f"Esporto {PT_PATH} -> {ONNX_PATH}  (opset {OPSET})")
    # dynamo=False usa il legacy TorchScript exporter: produce un singolo file
    # .onnx con i pesi inline (vs. il nuovo exporter che li mette in .onnx.data
    # esterno). Per un modello da ~24M parametri il singolo file e' piu'
    # comodo e portabile.
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )

    size_mb = ONNX_PATH.stat().st_size / 1e6
    print(f"OK: {ONNX_PATH} ({size_mb:.1f} MB)")
    return dummy, model


def verify(dummy: torch.Tensor, pt_model):
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n[skip verify] onnxruntime non installato.")
        print("Per verificare la consistenza output: pip install onnxruntime")
        return

    print("\nVerifica consistenza output PyTorch vs ONNX...")
    with torch.no_grad():
        pt_out = pt_model(dummy).numpy()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["mask"], {"input": dummy.numpy()})[0]

    diff = np.abs(pt_out - onnx_out).max()
    print(f"Max |PyTorch - ONNX|: {diff:.2e}  (tolleranza: {ATOL:.0e})")
    if diff < ATOL:
        print("PASS: output identici entro tolleranza.")
    else:
        print("WARN: differenza sopra tolleranza, controlla l'export.")


if __name__ == "__main__":
    dummy, model = export()
    verify(dummy, model)
