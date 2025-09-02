# Figure captions (Final Project Report alignment)

## Task 1 — Transit Detection (CNN variants)
- **ROC AUC comparison (V1–V4):** V3_SE highest (~0.987), confirming best ranking performance. :contentReference[oaicite:55]{index=55}
- **PR AUC comparison (V1–V4):** V3_SE ~0.962; maintains high precision at higher recall. :contentReference[oaicite:56]{index=56}
- **F1 comparison (V1–V4):** V3_SE ≈0.905 (abstract lists ~0.891). :contentReference[oaicite:57]{index=57}
- **Brier comparison:** V3_SE lowest (~0.039), showing best probability calibration. :contentReference[oaicite:58]{index=58}
- **Science recovery rate:** ~94% (196/208) of confirmed planets recovered. :contentReference[oaicite:59]{index=59}
- **Best-model panels (V3_SE):** CM/ROC/PR/Calibration illustrate strong separation and well-calibrated probabilities. :contentReference[oaicite:60]{index=60}

## Task 2 — Characterisation
- **Planet type confusion matrix (RF tuned):** Perfect diagonal; note potential feature leakage. :contentReference[oaicite:61]{index=61}
- **PR curves (planet type):** AP ~1.0 per class, consistent with perfect CM. :contentReference[oaicite:62]{index=62}
- **RF importances & SHAP:** Mass and equilibrium temperature dominate signal. :contentReference[oaicite:63]{index=63}
- **EDA (correlations & missingness):** Atmospheres table heavily missing; radius–mass ≈0.45, temp–insolation ≈0.58. :contentReference[oaicite:64]{index=64}
- **Molecule multilabel PR:** Strong for TiO (~0.97), H (~0.84), H₂O (~0.70). :contentReference[oaicite:65]{index=65}
- **Regression learning curves:** CV R² → ~0.999 (eq. temp) and ~0.985–0.99 (mass). :contentReference[oaicite:66]{index=66}

## Task 3 — Habitability
- **Confusion matrices (LGBM / RF):** (Near-)perfect diagonals; discuss class imbalance and label-rule overlap. :contentReference[oaicite:67]{index=67}
- **PR & ROC (LGBM):** Near-ideal separation; AUPRC/AUROC ~1.0. :contentReference[oaicite:68]{index=68}
- **SHAP & RF importances:** `pl_rade` and `pl_bmasse` principal drivers; insolation & eq. temp also important. :contentReference[oaicite:69]{index=69}

## Task 4 — Radio-Silence
- **ROC / PR:** AUROC ≈0.985, AUPR ≈0.956; operating threshold ≈0.66 with F1 ≈0.845. :contentReference[oaicite:70]{index=70}
- **Risk histogram & EDA:** Score distribution well separated; FPP/host-relative components show informative skew. :contentReference[oaicite:71]{index=71}
- **Fairness by Teff quartiles:** All groups maintain AUROC > 0.97. :contentReference[oaicite:72]{index=72}

## Task 5 — Probe Decision
- **Default vs comms-heavy:** Rankings robust; Top-50 Jaccard ≈0.786 under ±10% weight changes. :contentReference[oaicite:73]{index=73}
