"""
Inference su una singola immagine: produce e salva la maschera binaria.

Uso:
    python src/predict.py --model experiments/best.pt --image path/to/image.jpg
"""

import argparse
import numpy as np
import torch
from PIL import Image

import segmentation_models_pytorch as smp

IMG_SIZE = (256, 256)
THRESHOLD = 0.5


def predict_mask(model, img_path, img_size=IMG_SIZE, threshold=THRESHOLD, device=None):
    """
    Esegue l'inference su una singola immagine e ritorna la maschera binaria.

    Args:
        model:      Modello PyTorch già caricato e in eval mode.
        img_path:   Path dell'immagine di input.
        img_size:   Tuple (h, w) per il resize.
        threshold:  Soglia per binarizzare la maschera (default 0.5).
        device:     torch.device (default: cpu).

    Returns:
        np.ndarray di shape (h, w) con valori 0/1 (uint8).
    """
    if device is None:
        device = torch.device("cpu")

    img = Image.open(img_path).convert("RGB").resize(img_size)
    x = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
    x = x.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        mask = (torch.sigmoid(logits) > threshold).squeeze().cpu().numpy().astype(np.uint8)

    return mask


def main():
    parser = argparse.ArgumentParser(description="Privacy Blurrer — inference")
    parser.add_argument("--model", required=True, help="Path al checkpoint .pt")
    parser.add_argument("--image", required=True, help="Path immagine di input")
    parser.add_argument("--output", default="mask_output.png", help="Path maschera output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet("resnet34", encoder_weights=None, classes=1).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    mask = predict_mask(model, args.image, device=device)
    Image.fromarray(mask * 255).save(args.output)
    print(f"Maschera salvata in: {args.output}")


if __name__ == "__main__":
    main()
