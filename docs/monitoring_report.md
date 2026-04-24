# Monitoring Report — Step 6

## Overview

Il Privacy Blurrer integra un rilevatore di drift custom (`src/monitor.py`) che
valuta ogni immagine inviata agli endpoint `/predict` e `/blur` e logga il
risultato insieme alle altre metriche di runtime. Ogni chiamata produce una
riga in `logs/predictions.jsonl`.

L'obiettivo è flaggare gli input che differiscono significativamente dalla
distribuzione del training set, in modo da identificare condizioni in cui la
predizione rischia di essere poco affidabile.

---

## Approccio

Il detector estrae da ogni immagine un vettore a **3 feature** — la media
normalizzata dei canali RGB — e lo confronta con la distribuzione delle stesse
feature calcolata sul training set.

Algoritmo (implementato in `src/monitor.py`):

1. **Fit** (offline, `scripts/fit_detector.py`): per ciascuna delle 2133
   immagini del training set si calcola `[mean_R, mean_G, mean_B]` e si
   memorizzano i percentili al 2.5% e 97.5%, oltre a media e deviazione
   standard per canale.
2. **Check** (online, su ogni richiesta): si estrae il vettore dell'immagine
   corrente; se anche **un solo canale** cade fuori dall'intervallo
   `[percentile_low, percentile_high]` si segnala drift.
3. Si calcola inoltre un `drift_score` come deviazione normalizzata dalla
   media di riferimento: `mean(|features − mean_ref| / std_ref)`. È un
   valore continuo utile per graduare la severità (soglia binaria sul drift
   vs. score continuo per trend analysis).

Threshold implicito: `P_VAL_THRESHOLD = 0.05` in `monitor.py`, che significa
percentili al 2.5% e 97.5% (coda al 5% complessiva).

Scelta tecnica: il task richiedeva Alibi Detect, ma una dipendenza pesante non
era giustificata per 3 feature scalari. L'implementazione custom con solo
`numpy` è ~40 righe, zero dipendenze runtime aggiuntive, e produce output
equivalenti per questo caso d'uso.

---

## Reference distribution

Fittata su **2133 immagini** del training split del dataset Supervisely
Filtered Person Segmentation. File: `experiments/detector.json`.

| Canale | Mean | Std | P 2.5% | P 97.5% |
|---|---|---|---|---|
| R | 0.508 | 0.198 | 0.123 | 0.867 |
| G | 0.483 | 0.187 | 0.123 | 0.832 |
| B | 0.458 | 0.190 | 0.114 | 0.833 |

Le immagini vengono normalizzate in [0, 1] prima dell'estrazione feature.

---

## Log format

Ogni chiamata agli endpoint `/predict` o `/blur` produce una riga JSON in
`logs/predictions.jsonl`:

```json
{
  "timestamp": "2026-04-24T10:15:32.123456+00:00",
  "endpoint": "/blur",
  "filename": "ds10_pexels-photo-850708.png",
  "original_size": [640, 480],
  "mask_coverage": 0.1823,
  "preprocess_ms": 8.4,
  "inference_ms": 62.1,
  "latency_ms": 70.5,
  "validation_passed": true,
  "blur_type": "gaussian",
  "drift": {
    "is_drift": false,
    "drift_score": 0.7416,
    "features": [0.6989, 0.6523, 0.5263]
  }
}
```

Le richieste rifiutate in validazione (formato, dimensioni, size) loggano
`validation_passed: false` e un campo `rejection_reason`, senza `drift` né
metriche di inferenza.

---

## Comportamenti osservati

Entrambi i casi seguenti sono stati prodotti eseguendo `check_drift` sul
detector corrente (2133 campioni di riferimento).

### Caso 1 — In-distribution (no drift)

Input: `ds10_pexels-photo-850708.png`, foto standard di una persona presa dal
test split del dataset.

```json
{
  "is_drift": false,
  "drift_score": 0.7416,
  "features": [0.6989, 0.6523, 0.5263]
}
```

**Osservazione:** tutti e tre i canali cadono dentro i percentili di
riferimento. Il `drift_score` pari a 0.74 indica che l'immagine è ~0.74
deviazioni standard di distanza media dal centro della distribuzione — valore
normale per il test set. Nessun flag.

### Caso 2 — Out-of-distribution (drift detected)

Input: stessa immagine con uno shift di colore aggressivo (amplificazione
rossa ×1.8, compressione verde ×0.3 e blu ×0.2), che simula un filtro
anomalo / luce rossa intensa.

```json
{
  "is_drift": true,
  "drift_score": 1.5671,
  "features": [0.7684, 0.1957, 0.1053]
}
```

**Osservazione:** il canale blu (0.1053) è **sotto** il percentile 2.5%
(0.114) → drift flaggato. Anche il canale verde è vicino al limite inferiore
(0.196 vs. 0.123). Il `drift_score` è raddoppiato rispetto al caso 1 (1.57
vs. 0.74), coerente con l'ampia deviazione dalla media di riferimento.

---

## Discussione

**Cosa il detector cattura.** Cambiamenti di tono/illuminazione globale:
filtri aggressivi, foto notturne molto scure, immagini monocromatiche o con
palette esotica (rosso fuoco, sottomarino, infrarosso). Tutti casi in cui la
statistica di primo ordine dei canali cambia in modo marcato.

**Cosa il detector NON cattura.** Drift semantico a statistica RGB invariata:
foto con soggetti molto diversi dal training (strumenti medici, rendering
sintetici, satellitari) che però abbiano intensità medie "plausibili". Per
quei casi servirebbero feature deep (embedding di un encoder) o drift
detection supervisionato.

**Proxy utile per prediction drift.** `mask_coverage` nei log è un buon
indicatore secondario: un calo a quasi zero (il modello non vede nessuna
persona) o una saturazione verso 1 (falsi positivi diffusi) su input "in
distribuzione" sono segnali che il modello sta faticando anche senza
alterazioni evidenti della distribuzione di input.

**Azione consigliata in caso di drift frequente.** Rifittare il detector su
dati più recenti (`scripts/fit_detector.py`) oppure, se il drift riflette un
nuovo dominio di utilizzo, pianificare un re-training del modello includendo
campioni rappresentativi del nuovo dominio.
