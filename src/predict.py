"""
predict.py

Carica un modello addestrato e produce la maschera binaria per una immagine.

Uso:
    python src/predict.py --image data/raw/images/foto.jpg --model experiments/best.pt
    python src/predict.py --image data/raw/images/foto.jpg --model experiments/best.pt --output mask.png
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp


IMAGE_SIZE = (256, 256)


def load_model(model_path: str) -> torch.nn.Module:
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    image = np.array(Image.open(image_path).convert("RGB").resize(IMAGE_SIZE))
    tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def predict(model: torch.nn.Module, image_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    with torch.no_grad():
        logits = model(image_tensor)
        mask   = (torch.sigmoid(logits) > threshold).squeeze().numpy()
    return (mask * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Run inference with Privacy Blurrer model")
    parser.add_argument("--image",     type=str, required=True,              help="Path immagine input")
    parser.add_argument("--model",     type=str, default="experiments/best.pt", help="Path modello .pt")
    parser.add_argument("--output",    type=str, default=None,               help="Path maschera output (opzionale)")
    parser.add_argument("--threshold", type=float, default=0.5,              help="Soglia binarizzazione")
    args = parser.parse_args()

    print(f"Carico modello da {args.model}...")
    model = load_model(args.model)

    print(f"Elaboro immagine {args.image}...")
    image_tensor = preprocess_image(args.image)
    mask = predict(model, image_tensor, args.threshold)

    if args.output:
        Image.fromarray(mask).save(args.output)
        print(f"Maschera salvata in {args.output}")
    else:
        output_path = Path(args.image).stem + "_mask.png"
        Image.fromarray(mask).save(output_path)
        print(f"Maschera salvata in {output_path}")

    person_pixels = (mask > 127).sum()
    total_pixels  = mask.size
    print(f"Persona rilevata: {person_pixels}/{total_pixels} pixel ({100*person_pixels/total_pixels:.1f}%)")


if __name__ == "__main__":
    main()
