# Monitoring Report — Step 6

## Overview

Il Privacy Blurrer integra un rilevatore di drift (`src/monitor.py`) basato su
**alibi-detect** (`alibi_detect.cd.KSDrift`). A ogni chiamata degli endpoint
`/predict` e `/blur` il detector valuta l'immagine ricevuta e il risultato
viene loggato insieme alle altre metriche di runtime in
`logs/predictions.jsonl`.

L'obiettivo è flaggare gli input la cui distribuzione di colore differisce
significativamente dal training set, in modo da identificare condizioni in
cui la predizione rischia di essere poco affidabile.

---

## Approccio

Il detector estrae da ogni immagine un vettore a **3 feature** — la media
normalizzata dei canali RGB — e applica un test di **Kolmogorov-Smirnov** per
canale, confrontando il singolo sample con la distribuzione di riferimento
calcolata sul training set.

Algoritmo:

1. **Fit** (offline, `scripts/fit_detector.py`): per ciascuna delle 2133
   immagini del training set si calcola `[mean_R, mean_G, mean_B]` e si
   passa la matrice di riferimento a
   `KSDrift(x_ref=..., p_val=0.05)`. Il detector fittato viene serializzato
   in `experiments/detector.pkl`.
2. **Check** (online, per ogni richiesta): si estrae il vettore dell'immagine
   corrente e si chiama `detector.predict(features)`. alibi-detect applica KS
   canale per canale e aggrega via **correzione di Bonferroni** (soglia per
   canale = `p_val / n_features`, cioè 0.0167 con 3 feature).
3. Nello schema di log esponiamo: `is_drift` (bool), `drift_score` (media
   delle KS distance sui 3 canali, in [0, 1]) e `features` (vettore
   `[mean_R, mean_G, mean_B]` dell'immagine).

Threshold: `P_VAL_THRESHOLD = 0.05` in `monitor.py`, che dopo correzione
Bonferroni diventa 0.0167 per canale.

---

## Reference distribution

Fittata su **2133 immagini** del training split del dataset Supervisely
Filtered Person Segmentation. File: `experiments/detector.pkl` (non tracciato
da git; rigenerabile con `python scripts/fit_detector.py`).

Statistiche della distribuzione di riferimento (calcolate sui feature vector
`[mean_R, mean_G, mean_B]` dei 2133 campioni):

| Canale | Mean | Std |
|---|---|---|
| R | 0.508 | 0.198 |
| G | 0.483 | 0.187 |
| B | 0.458 | 0.190 |

Le immagini vengono normalizzate in [0, 1] prima dell'estrazione feature.

---

## Log format

Ogni chiamata agli endpoint `/predict` o `/blur` produce una riga JSON in
`logs/predictions.jsonl`:

```json
{
  "timestamp": "2026-04-24T17:21:56.297432+00:00",
  "endpoint": "/blur",
  "filename": "ds10_pexels-photo-850708.png",
  "original_size": [256, 256],
  "mask_coverage": 0.1624,
  "preprocess_ms": 1.57,
  "inference_ms": 40.65,
  "latency_ms": 42.22,
  "validation_passed": true,
  "blur_type": "blackout",
  "drift": {
    "is_drift": false,
    "drift_score": 0.7482,
    "features": [0.6989, 0.6523, 0.5263]
  }
}
```

Le richieste rifiutate in validazione (formato, dimensioni, size) loggano
`validation_passed: false` e un campo `rejection_reason`, senza `drift` né
metriche di inferenza.

---

## Comportamenti osservati

Entrambi i casi sono stati prodotti chiamando `check_drift` sul detector
corrente (fit su 2133 campioni).

### Caso 1 — In-distribution (no drift)

Input: `ds10_pexels-photo-850708.png`, foto standard di una persona presa dal
test split del dataset.

```json
{
  "is_drift": false,
  "drift_score": 0.7482,
  "features": [0.6989, 0.6523, 0.5263]
}
```

**Osservazione:** le feature sono vicine alla zona densa della distribuzione
di riferimento; KSDrift non rileva alcuna deviazione significativa su
nessuno dei 3 canali. Nessun flag.

### Caso 2 — Out-of-distribution (drift detected)

Input: immagine monocromatica quasi nera (tutti i pixel impostati a 0.05),
scenario estremo utile per verificare che il detector reagisca a un input
chiaramente fuori dominio.

```json
{
  "is_drift": true,
  "drift_score": 0.9989,
  "features": [0.0500, 0.0500, 0.0500]
}
```

**Osservazione:** le feature 0.05 cadono ben sotto la coda inferiore della
distribuzione di riferimento su tutti e 3 i canali (il training set è
dominato da foto a illuminazione normale). La KS distance è prossima al
massimo teorico (≈ 1) su ogni canale e il p-value corretto supera la soglia,
triggerando il flag.

---

## Discussione

**Cosa il detector cattura.** Cambiamenti di illuminazione o palette globali
abbastanza marcati da spostare la statistica di primo ordine dei canali
chiaramente fuori dal supporto della reference: foto sovraesposte o
sottoesposte, immagini monocromatiche, filtri aggressivi, rendering
sintetici con palette innaturale.

**Limite del KS test in regime single-sample.** KSDrift, con `predict()` su
un singolo sample contro 2133 reference + correzione Bonferroni, ha
deliberatamente bassa sensibilità a variazioni moderate: serve un p-value
per canale sotto 0.0167 per triggerare il flag, il che in pratica richiede
che il sample cada molto vicino al limite della distribuzione di
riferimento. È la scelta corretta per evitare falsi positivi su ogni
richiesta, ma per catturare drift *di popolazione* sarebbe più robusto
accumulare predictions in batch e lanciare KS sul batch (ad esempio ogni
100 richieste).

**Cosa il detector NON cattura.** Drift semantico a statistica RGB invariata:
foto con soggetti molto diversi dal training (strumenti medici, satellitari,
rendering) che abbiano intensità medie "plausibili". Per quei casi
servirebbero feature deep (embedding di un encoder) o drift detection
supervisionato sulle predictions.

**Proxy utile per prediction drift.** `mask_coverage` nei log è un buon
indicatore secondario: un calo a quasi zero (il modello non vede nessuna
persona) o una saturazione verso 1 (falsi positivi diffusi) su input "in
distribuzione" sono segnali che il modello sta faticando anche senza
alterazioni evidenti della distribuzione di input.

**Azione consigliata in caso di drift frequente.** Rifittare il detector su
dati più recenti (`scripts/fit_detector.py`) oppure, se il drift riflette un
nuovo dominio di utilizzo, pianificare un re-training del modello includendo
campioni rappresentativi del nuovo dominio.
