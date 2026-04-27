"""
Test sulla CLI di src/predict.py.

Coprono il main() con argparse + caricamento modello + invocazione di
predict_mask + salvataggio della maschera output. Per evitare di dover
caricare un best.pt reale, salviamo un mini-checkpoint fake nel tmp_path.
"""

import sys

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from src.predict import main


@pytest.fixture
def fake_checkpoint_and_image(tmp_path):
    """Crea un mini-checkpoint U-Net e un'immagine fake in tmp_path."""
    # Mini-modello (architettura uguale a quella usata in main, encoder_weights=None)
    model = smp.Unet("resnet34", encoder_weights=None, classes=1)
    ckpt_path = tmp_path / "fake_best.pt"
    torch.save(model.state_dict(), ckpt_path)

    # Immagine fake 64x64
    img_path = tmp_path / "input.png"
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_path)

    output_path = tmp_path / "mask.png"
    return ckpt_path, img_path, output_path


def test_predict_cli_produces_mask(fake_checkpoint_and_image, monkeypatch, capsys):
    """main() carica il checkpoint, fa inferenza, salva la maschera."""
    ckpt_path, img_path, output_path = fake_checkpoint_and_image

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict.py",
            "--model",
            str(ckpt_path),
            "--image",
            str(img_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    # La maschera deve essere stata salvata
    assert output_path.exists()

    # Deve essere un PNG leggibile, dimensione 256x256 (IMG_SIZE in predict.py)
    mask = Image.open(output_path)
    assert mask.size == (256, 256)

    # Stdout deve contenere il messaggio di conferma
    captured = capsys.readouterr()
    assert "Maschera salvata" in captured.out


def test_predict_cli_requires_model_arg(monkeypatch):
    """argparse deve fallire (SystemExit) se manca --model."""
    monkeypatch.setattr(sys, "argv", ["predict.py", "--image", "foo.png"])

    with pytest.raises(SystemExit):
        main()
