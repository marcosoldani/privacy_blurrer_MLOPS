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

Training tracciato con **MLflow** (`experiments/mlruns/`, esperimento
`privacy_blurrer`). Il checkpoint migliore è salvato in `experiments/best.pt`.

---

## Setup

```bash
# dipendenze Python
pip install -r requirements.txt

# dipendenze frontend (opzionale)
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
python src/train.py
# output: experiments/best.pt + experiments/mlruns/<run_id>/
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

---

## Struttura del progetto

```
privacy_blurrer_MLOPS/
├── src/                    # codice applicativo
│   ├── app.py              # FastAPI: /health, /predict, /blur
│   ├── dataset.py          # PersonSegmentationDataset (PyTorch)
│   ├── preprocess.py       # resize + split train/val/test
│   ├── train.py            # training loop con MLflow
│   ├── predict.py          # CLI inference one-shot
│   └── monitor.py          # drift detection (numpy, percentile-based)
├── scripts/                # utility one-shot
│   ├── fit_detector.py     # fit drift detector sul training
│   └── generate_manifest.py# genera manifest versionato
├── tests/                  # test PyTest (12 test)
│   ├── test_dataset.py
│   └── test_pipeline.py
├── data/
│   ├── raw/                # dataset grezzo (in .gitignore)
│   ├── processed/          # output di preprocess.py (in .gitignore)
│   └── manifests/          # dataset_v1.0.json
├── experiments/
│   ├── best.pt             # checkpoint modello
│   ├── detector.json       # drift detector fittato
│   ├── mlflow.db           # MLflow tracking DB (in .gitignore)
│   └── mlruns/             # MLflow run artifacts (in .gitignore)
├── logs/
│   └── predictions.jsonl   # log JSON strutturato runtime
├── frontend/               # React + Vite + Tailwind
├── docs/
│   └── monitoring_report.md# report drift detection (Step 6)
├── .github/workflows/ci.yml# PyTest su push/PR (main, dev)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Test

```bash
pytest tests/ -v
```

12 test: dataset (5) + pipeline end-to-end (7) — preprocess, forward pass,
training mini-run, inference shape/binarietà.

La CI su GitHub Actions (`.github/workflows/ci.yml`) li esegue automaticamente
su push/PR verso `main` e `dev`.

---

## Monitoring

Ogni chiamata a `/predict` e `/blur` scrive una riga JSON in
`logs/predictions.jsonl` con: timestamp, endpoint, filename, dimensioni,
`mask_coverage`, latency (preprocess + inference), `validation_passed` e
`drift` (is_drift, drift_score, features mean RGB).

Dettagli e comportamento del drift detector: [docs/monitoring_report.md](docs/monitoring_report.md).
