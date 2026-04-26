"""
Registra il checkpoint corrente (`experiments/best.pt`) come prima versione
del modello "privacy_blurrer_unet" nel MLflow Model Registry.

Da eseguire una sola volta per popolare il registry con il modello in
produzione (utile dopo il primo training, prima di automatizzare la
registrazione in train.py).

Uso (dalla root del progetto):
    python scripts/register_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow  # noqa: E402

from src.train import REGISTERED_MODEL_NAME, TRACKING_URI  # noqa: E402

CHECKPOINT = Path("experiments/best.pt")
EXPERIMENT_NAME = "privacy_blurrer_registry"


def main():
    if not CHECKPOINT.exists():
        raise SystemExit(f"Checkpoint non trovato: {CHECKPOINT}")

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="manual_registration") as run:
        mlflow.log_artifact(str(CHECKPOINT))
        mlflow.set_tag("source", "manual registration of pre-existing best.pt")

        result = mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/{CHECKPOINT.name}",
            name=REGISTERED_MODEL_NAME,
        )
        print(f"Registered: {result.name} v{result.version} ({result.status})")
        print(f"Run ID:     {run.info.run_id}")
        print(f"MLflow UI:  mlflow ui --backend-store-uri {TRACKING_URI}")


if __name__ == "__main__":
    main()
