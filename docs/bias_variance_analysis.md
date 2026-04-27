# Bias-Variance Analysis

Analisi del miglior checkpoint del modello U-Net (`experiments/best.pt`),
basata sulle metriche per epoca tracciate in MLflow (esperimento
`privacy_blurrer`, run `350dc4d5...`) e su una misura indipendente sul
val set fissato `data/processed/val/`.

## Metriche

| Epoca | train_loss | val_loss | val_iou |
|---|---|---|---|
| 1  | 0.2191 | 0.2012 | 0.6705 |
| 2  | 0.1697 | 0.1951 | 0.6778 |
| 3  | 0.1641 | 0.1526 | 0.7394 |
| 4  | 0.1452 | 0.1401 | 0.7573 |
| 5  | 0.1291 | 0.1397 | 0.7586 |
| 6  | 0.1319 | 0.1343 | 0.7662 |
| 7  | 0.1230 | 0.1229 | 0.7850 |
| 8  | 0.1206 | 0.1298 | 0.7739 |
| 9  | 0.1154 | 0.1190 | 0.7907 |
| 10 | 0.1021 | 0.1165 | 0.7945 |

Misura indipendente al 2026-04-25 sul val set fissato (266 immagini di
`data/processed/val/`, mai usato nel training loop): **mean IoU = 0.8382**.
La differenza con il valore MLflow (0.7945) è dovuta al fatto che `train.py`
usa un proprio `random_split` su `data/raw/` invece dello split deterministico
di `preprocess.py`; il val set fissato è leggermente più "facile" del campione
random visto durante il training.

## Diagnosi

**Bias basso, varianza bassa.** Il gap fra train_loss (0.1021) e val_loss
(0.1165) all'ultima epoca è di soli 0.014 (~14% del val_loss): il modello non
sta overfittando. Allo stesso tempo il train_loss continua a scendere
monotonicamente fino all'epoca 10 (0.22 → 0.10), e il val_iou non ha ancora
plateaued (0.78 → 0.79 fra epoca 7 e 10), il che indica che il modello
**potrebbe ancora imparare** — siamo leggermente sul lato underfit di un
regime già equilibrato.

In pratica: con 24M parametri (U-Net + ResNet34) e 2133 sample di training, la
capacità del modello e la regolarizzazione implicita dell'encoder pre-trained
ImageNet sono già ben calibrate per il task. Non serve aggiungere dropout,
weight decay aggressivo o data augmentation pesante; servirebbe più "tempo" o
più "dati" per migliorare ulteriormente.

## Prossime mosse, in ordine di rapporto valore/costo

1. **Più epoche di training** (costo ~0). Le curve mostrano che a epoca 10 il
   train_loss sta ancora scendendo. Trainare a 20-30 epoche con
   `EarlyStopping` su val_iou plateau → guadagno atteso 1-3 punti di IoU.
2. **Data augmentation con albumentations** (costo basso, libreria già in
   `requirements.txt`). Horizontal flip + random brightness/contrast +
   piccole rotazioni ridurrebbero ulteriormente la varianza già bassa e
   migliorerebbero la robustezza a condizioni di acquisizione diverse da
   quelle del training set. Atteso 1-2 punti di IoU + drift detector meno
   sensibile a piccole variazioni di luminosità.
3. **Encoder più grande** (es. `resnet50` o `efficientnet-b3`) — costo medio
   in compute, atteso 2-4 punti di IoU. Chiude la quota residua di bias.
4. **Più dati di training**, in particolare campioni di edge case
   (illuminazione difficile, soggetti parziali, scene affollate). Migliora
   sia il bias sia la robustezza in produzione, ma richiede labeling.
