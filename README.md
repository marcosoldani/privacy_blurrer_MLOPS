# Privacy Blurrer

Servizio di segmentazione + anonimizzazione delle persone in un'immagine.
L'utente carica una foto; il backend restituisce la stessa foto con le persone
sfocate, pixelate o oscurate, oppure la sola maschera binaria.

Progetto MLOps del corso **ML Operation** (SUPSI).

---

## Goal

Rilevare tutte le persone in un'immagine e sostituirle con una versione
anonimizzata, lasciando inalterato il resto della scena. Caso d'uso tipico:
pubblicare foto senza esporre volti/identità, preservando il contesto.

Il sistema espone tre modalità di anonimizzazione:

| Modalità | Effetto |
|---|---|
| `gaussian` | blur gaussiano (kernel 51×51) sulle persone |
| `pixelate` | effetto mosaico (blocchi 20 px) sulle persone |
| `blackout` | persone sostituite con rettangoli neri |

In alternativa, l'endpoint `/predict` restituisce la sola maschera binaria
(255 = persona, 0 = sfondo).

---

## Data source

**Supervisely Filtered Person Segmentation** — Kaggle
([link](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset)).

- 2667 coppie immagine/maschera
- Formato PNG (varie risoluzioni, contesti outdoor/indoor)
- Maschera binaria: 255 = persona, 0 = sfondo

Versioning del dataset: `data/manifests/dataset_v1.0.json` (generato da
`scripts/generate_manifest.py`).

Split (gestito da `src/preprocess.py`, seed 42):
- train: 80% (2133 campioni)
- val:   10% (266)
- test:  10%

---

## Modello

**U-Net** con encoder **ResNet34** (pre-trained ImageNet), via
`segmentation-models-pytorch`.

- Input: 256×256 RGB
- Output: maschera binaria 256×256, soglia 0.5 su sigmoid
- Loss: Dice (binary)
- Optimizer: Adam, lr=1e-3
- Epoche: 10, batch size 8
- ~24.4M parametri, checkpoint ~93 MB

Training tracciato con **MLflow** (backend SQLite `experiments/mlflow.db`,
esperimento `privacy_blurrer`). Il checkpoint migliore è salvato in
`experiments/best.pt` ed è registrato come versione del modello
`privacy_blurrer_unet` nel **MLflow Model Registry**.

---

## Setup

Ricetta consigliata (conda + pip, riproducibile):

```bash
conda env create -f environment.yml
conda activate privacy_blurrer
```

In alternativa con solo pip:

```bash
pip install -r requirements.txt
```

Dipendenze frontend (opzionale):

```bash
cd frontend && npm install && cd ..
```

### Preparazione dati

```bash
# da data/raw/{images,masks} → data/processed/{train,val,test}
python src/preprocess.py

# manifest versionato del dataset (dopo preprocess)
python scripts/generate_manifest.py

# fit del drift detector sul training set
python scripts/fit_detector.py
```

### Training

```bash
python src/train.py                 # baseline
python src/train.py --augment       # con data augmentation albumentations
```

Output:
- `experiments/best.pt` — checkpoint del modello
- Run + metriche in `experiments/mlflow.db` (SQLite backend)
- Nuova versione registrata in `MLflow Model Registry`
  (model: `privacy_blurrer_unet`)

Per ispezionare run + registry:

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlflow.db
```

Per registrare un `best.pt` preesistente nel registry (one-shot):

```bash
python scripts/register_model.py
```

### Inference CLI

```bash
python src/predict.py --model experiments/best.pt --image foo.jpg --output mask.png
```

### Backend (API)

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` — liveness + stato modello
- `POST /predict` — maschera binaria PNG (multipart: `file`)
- `POST /blur?blur_type={gaussian|pixelate|blackout}` — immagine anonimizzata PNG

Validazione input:
- formati: JPEG, PNG, BMP, WEBP
- dimensioni: 32–4096 px per lato
- max 10 MB

### Frontend (React + Vite)

```bash
cd frontend
npm run dev
# apre http://localhost:5173
```

### Docker

```bash
docker build -t privacy-blurrer .
docker run -p 8000:8000 privacy-blurrer
```

Il container espone l'API su `http://localhost:8000`.

---

## Struttura del progetto

```
privacy_blurrer_MLOPS/
├── src/                    # codice applicativo
│   ├── app.py              # FastAPI: /health, /predict, /blur
│   ├── dataset.py          # PersonSegmentationDataset (PyTorch)
│   ├── preprocess.py       # resize + split train/val/test
│   ├── train.py            # training loop con MLflow + Model Registry (--augment opt)
│   ├── predict.py          # CLI inference one-shot
│   └── monitor.py          # drift detection (alibi-detect KSDrift)
├── scripts/                # utility one-shot
│   ├── fit_detector.py     # fit drift detector sul training
│   ├── generate_manifest.py# genera manifest versionato
│   ├── benchmark_latency.py# misura P50/P95/P99 inference
│   ├── export_onnx.py      # esporta best.pt → best.onnx
│   ├── quantize_model.py   # INT8 / FP16 quantization + size/latency report
│   └── register_model.py   # registra best.pt nel MLflow Model Registry
├── tests/                  # test PyTest (23 test, ~75% coverage)
│   ├── test_dataset.py     # unit test sul Dataset
│   ├── test_pipeline.py    # infrastructure tests (incl. overfit single batch)
│   ├── test_evaluation.py  # evaluation test con threshold IoU
│   ├── test_behavioral.py  # behavioral (flip) + perturbation (noise)
│   └── test_api.py         # API tests via FastAPI TestClient
├── data/
│   ├── raw/                # dataset grezzo (in .gitignore)
│   ├── processed/          # output di preprocess.py (in .gitignore)
│   └── manifests/          # dataset_v1.0.json
├── experiments/
│   ├── best.pt             # checkpoint modello PyTorch
│   ├── best.onnx           # export ONNX (in .gitignore, rigenerabile)
│   ├── detector.pkl        # drift detector fittato (alibi-detect KSDrift)
│   └── mlflow.db           # MLflow SQLite backend + Model Registry (in .gitignore)
├── logs/
│   └── predictions.jsonl   # log JSON strutturato runtime
├── frontend/               # React + Vite + Tailwind
├── docs/
│   ├── monitoring_report.md         # report drift detection
│   ├── bias_variance_analysis.md    # analisi metriche e prossimi passi
│   └── foundation_model_decision.md # giustificazione scelta no-FM
├── .github/workflows/
│   └── ci.yml              # Black + Flake8 + PyTest su push/PR (main, dev)
├── .pre-commit-config.yaml # hook locali Black + Flake8 + checks
├── Dockerfile
├── .dockerignore           # riduce build context da 3.4 GB a ~95 MB
├── environment.yml         # conda env (python 3.10 + pip:requirements.txt)
├── requirements.txt
└── README.md
```

---

## Test

```bash
pytest tests/ -v
```

23 test, **coverage ~75%** (`src/app.py` 89%, `src/dataset.py` 89%,
`src/preprocess.py` 97%):
- **dataset** (5) — unit test su `PersonSegmentationDataset`
- **pipeline** (8) — infrastructure tests, incluso overfit-single-batch
- **evaluation** (1) — regression guard con threshold IoU su val set
- **behavioral** (2) — flip invariance + gaussian noise robustness
- **api** (7) — FastAPI TestClient: `/health`, validation 400/413, happy paths

Coverage dettagliata:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Lint

```bash
black --check src/ tests/ scripts/ --line-length 100
flake8 src/ tests/ scripts/ --max-line-length=100 --extend-ignore=E203,W503
```

### Pre-commit hooks

Per applicare lint + format automaticamente prima di ogni commit:

```bash
pip install pre-commit
pre-commit install
```

I hook configurati in `.pre-commit-config.yaml` eseguono Black, Flake8,
trailing-whitespace, end-of-file-fixer, check-yaml e check-added-large-files
ad ogni `git commit`.

### CI

CI GitHub Actions (`.github/workflows/ci.yml`) esegue automaticamente
**Black + Flake8 + PyTest + coverage** su push/PR verso `main` e `dev`.
Il report di coverage `coverage.xml` viene caricato come artifact per la
visualizzazione.

---

## Monitoring

Ogni chiamata a `/predict` e `/blur` scrive una riga JSON in
`logs/predictions.jsonl` con: timestamp, endpoint, filename, dimensioni,
`mask_coverage`, latency (preprocess + inference), `validation_passed` e
`drift` (is_drift, drift_score, features mean RGB).

Dettagli e comportamento del drift detector: [docs/monitoring_report.md](docs/monitoring_report.md).
