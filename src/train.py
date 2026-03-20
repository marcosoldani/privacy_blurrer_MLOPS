import mlflow
import torch
from torch.utils.data import DataLoader, random_split
from src.dataset import PersonSegmentationDataset
import segmentation_models_pytorch as smp

PARAMS = {
    "epochs":     10,
    "batch_size":  8,
    "lr":          1e-3,
    "encoder":    "resnet34",
    "data_dir":   "data/raw",
}


def iou(pred, mask, thr=0.5, eps=1e-6):
    p = (torch.sigmoid(pred) > thr).float()
    return ((p * mask).sum() + eps) / ((p + mask).clamp(0, 1).sum() + eps)


def run():
    ds    = PersonSegmentationDataset(PARAMS["data_dir"])
    n_val = max(1, int(len(ds) * 0.1))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=PARAMS["batch_size"], shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=PARAMS["batch_size"])

    model     = smp.Unet(PARAMS["encoder"], encoder_weights="imagenet", classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr"])
    loss_fn   = smp.losses.DiceLoss(mode="binary")

    mlflow.set_tracking_uri("experiments/mlruns")
    mlflow.set_experiment("privacy_blurrer")

    with mlflow.start_run():
        mlflow.log_params(PARAMS)

        for epoch in range(1, PARAMS["epochs"] + 1):
            # Training
            model.train()
            train_loss = 0.0
            for x, y in train_dl:
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dl)

            # Validation
            model.eval()
            val_loss, val_iou = 0.0, 0.0
            with torch.no_grad():
                for x, y in val_dl:
                    p         = model(x)
                    val_loss += loss_fn(p, y).item()
                    val_iou  += iou(p, y).item()
            val_loss /= len(val_dl)
            val_iou  /= len(val_dl)

            mlflow.log_metrics(
                {"train_loss": round(train_loss, 4),
                 "val_loss":   round(val_loss, 4),
                 "val_iou":    round(val_iou, 4)},
                step=epoch,
            )
            print(f"[{epoch:02d}] loss={train_loss:.4f}  val_iou={val_iou:.4f}")

        torch.save(model.state_dict(), "experiments/best.pt")
        mlflow.log_artifact("experiments/best.pt")


if __name__ == "__main__":
    run()
