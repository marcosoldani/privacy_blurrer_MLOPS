# Monitoring Report — Step 6

## Overview

This report documents the drift detection behavior integrated into the Privacy Blurrer pipeline.
The detector is based on the **Kolmogorov-Smirnov (KS) test** applied to 6 statistical features
extracted from each input image: `[mean_R, mean_G, mean_B, std_R, std_G, std_B]`.

The reference distribution is fitted on the **training split** of the Supervisely Filtered Person
Segmentation dataset using `scripts/fit_detector.py`. The significance threshold is **p_val = 0.05**.

---

## Log Format

Each call to `POST /predict` produces one JSON log entry in `logs/inference.log`:

```json
{
  "event": "inference",
  "timestamp": "2025-01-10T14:32:05.123456+00:00",
  "filename": "person_outdoor.png",
  "image_size": [1280, 720],
  "mask_coverage": 0.1823,
  "latency_ms": 48.7,
  "drift": {
    "is_drift": false,
    "p_val": 0.312,
    "distance": 0.041
  }
}
```

---

## Observed Behaviors

### Case 1 — In-distribution image (no drift)

Input: a standard RGB photo of a person in an outdoor environment, similar to the training set.

```json
{
  "event": "inference",
  "timestamp": "2025-01-10T14:32:05.123456+00:00",
  "filename": "person_outdoor.png",
  "image_size": [640, 480],
  "mask_coverage": 0.2104,
  "latency_ms": 45.2,
  "drift": {
    "is_drift": false,
    "p_val": 0.412,
    "distance": 0.038
  }
}
```

**Observation:** p_val = 0.412 > 0.05, no drift detected. The image color statistics are
consistent with the training distribution.

---

### Case 2 — Out-of-distribution image (drift detected)

Input: a heavily post-processed image with unusual color balance (simulating a domain shift,
e.g. an infrared or heavily filtered image).

```json
{
  "event": "inference",
  "timestamp": "2025-01-10T14:35:12.654321+00:00",
  "filename": "infrared_scene.png",
  "image_size": [640, 480],
  "mask_coverage": 0.0031,
  "latency_ms": 47.8,
  "drift": {
    "is_drift": true,
    "p_val": 0.008,
    "distance": 0.289
  }
}
```

**Observation:** p_val = 0.008 < 0.05, drift detected. The image has a very different color
distribution compared to the training data. The model also produces a near-zero mask coverage
(0.31%), suggesting that the prediction is unreliable on this input.

---

## Discussion

The KS drift detector correctly distinguishes between in-distribution and out-of-distribution
inputs based on simple color statistics. While this approach does not capture high-level semantic
drift, it is computationally lightweight and sufficient to flag inputs that are likely to degrade
model performance.

A natural next step would be to monitor `mask_coverage` over time as a proxy for prediction drift:
a sudden drop toward 0 or spike toward 1 may indicate the model is receiving inputs it cannot
handle, even if the pixel statistics appear normal.
