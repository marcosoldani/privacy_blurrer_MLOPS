"""
Training loop con MLflow experiment tracking + Model Registry.

Uso:
    python src/train.py
    python src/train.py --augment   # abilita data augmentation albumentations

Backend MLflow: sqlite (`experiments/mlflow.db`). Necessario per il Model
Registry. Per ispezionare:
    mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
"""

import argparse

import albumentations as A
import mlflow
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.dataset import PersonSegmentationDataset

PARAMS = {
    "epochs": 10,
    "batch_size": 8,
    "lr": 1e-3,
    "image_size": 256,
    "encoder": "resnet34",
    "train_dir": "data/processed/train",
    "val_dir": "data/processed/val",
}

TRACKING_URI = "sqlite:///experiments/mlflow.db"
EXPERIMENT_NAME = "privacy_blurrer"
REGISTERED_MODEL_NAME = "privacy_blurrer_unet"


def build_train_transform() -> A.Compose:
    """Augmentation leggera per il training set: flip orizzontale + jitter
    di luminosita'/contrasto. Pensata per ridurre la varianza residua e
    aumentare la robustezza a condizioni di acquisizione diverse dal
    training set originale (cfr. docs/bias_variance_analysis.md)."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        ]
    )


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(0, 1).sum()
    return ((intersection + eps) / (union + eps)).item()


def run(use_augmentation: bool = False):
    # Usa gli split deterministici creati da preprocess.py.
    # Augmentation applicata solo al training set; val resta pulito.
    transform = build_train_transform() if use_augmentation else None
    train_ds = PersonSegmentationDataset(PARAMS["train_dir"], transform=transform)
    val_ds = PersonSegmentationDataset(PARAMS["val_dir"])

    train_loader = DataLoader(train_ds, batch_size=PARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=PARAMS["batch_size"])

    model = smp.Unet(PARAMS["encoder"], encoder_weights="imagenet", classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])
    loss_fn = smp.losses.DiceLoss(mode="binary")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run_ctx:
        mlflow.log_params({**PARAMS, "augmentation": use_augmentation})

        for epoch in range(1, PARAMS["epochs"] + 1):
            model.train()
            train_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            val_loss, val_iou = 0.0, 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    p = model(x)
                    val_loss += loss_fn(p, y).item()
                    val_iou += iou_score(p, y)
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)

            mlflow.log_metrics(
                {
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4),
                    "val_iou": round(val_iou, 4),
                },
                step=epoch,
            )
            print(f"[{epoch:02d}] loss={train_loss:.4f}  val_iou={val_iou:.4f}")

        torch.save(model.state_dict(), "experiments/best.pt")
        mlflow.log_artifact("experiments/best.pt")

        # Model Registry: ogni training run produce una nuova versione del
        # modello, promuovibile manualmente a Staging/Production via UI MLflow.
        result = mlflow.register_model(
            model_uri=f"runs:/{run_ctx.info.run_id}/best.pt",
            name=REGISTERED_MODEL_NAME,
        )
        print(f"Registered model: {result.name} v{result.version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Privacy Blurrer U-Net")
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Abilita data augmentation albumentations sul training set",
    )
    args = parser.parse_args()
    run(use_augmentation=args.augment)
