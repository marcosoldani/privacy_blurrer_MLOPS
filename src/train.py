"""
Training loop con MLflow experiment tracking.

Uso:
    python src/train.py
"""
import mlflow
import torch
from torch.utils.data import DataLoader, random_split

import segmentation_models_pytorch as smp
from src.dataset import PersonSegmentationDataset

PARAMS = {
    "epochs":     10,
    "batch_size": 8,
    "lr":         1e-3,
    "image_size": 256,
    "encoder":    "resnet34",
    "data_dir":   "data/processed/train",
}


def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred_bin     = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    union        = (pred_bin + target).clamp(0, 1).sum()
    return ((intersection + eps) / (union + eps)).item()


def run():
    ds    = PersonSegmentationDataset(PARAMS["data_dir"])
    n_val = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=PARAMS["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=PARAMS["batch_size"])

    model     = smp.Unet(PARAMS["encoder"], encoder_weights="imagenet", classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])
    loss_fn   = smp.losses.DiceLoss(mode="binary")

    mlflow.set_tracking_uri("experiments/mlruns")
    mlflow.set_experiment("privacy_blurrer")

    with mlflow.start_run():
        mlflow.log_params(PARAMS)

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
                    val_iou  += iou_score(p, y)
            val_loss /= len(val_loader)
            val_iou  /= len(val_loader)

            mlflow.log_metrics(
                {"train_loss": round(train_loss, 4),
                 "val_loss":   round(val_loss, 4),
                 "val_iou":    round(val_iou, 4)},
                step=epoch
            )
            print(f"[{epoch:02d}] loss={train_loss:.4f}  val_iou={val_iou:.4f}")

        torch.save(model.state_dict(), "experiments/best.pt")
        mlflow.log_artifact("experiments/best.pt")


if __name__ == "__main__":
    run()
