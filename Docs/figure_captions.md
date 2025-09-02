# Figure captions (used in the Final Project Report)

## Task 1 — Transit Detection
- **ROC AUC comparison (V1–V4):** V3_SE leads with ≈0.987, confirming the best ranking performance.
- **PR AUC comparison (V1–V4):** V3_SE achieves ≈0.962; precision remains high even at higher recall.
- **F1 comparison (V1–V4):** V3_SE best (≈0.905); Residual (V2) worst (≈0.841).
- **Calibration/Brier comparison:** V3_SE lowest Brier (≈0.039), indicating well-calibrated probabilities.
- **Science recovery rate:** V3_SE recovered 196/208 known planets (≈94.2%); 12 misses (FN).
- **Best model panels (V3_SE):** CM/ROC/PR/Calibration/Learning curves show stable training and well-calibrated outputs.

## Task 2 — Characterisation
- **Planet type CM (RF tuned):** Perfect diagonal; extremely high performance — flagged with a leakage-audit note.
- **PR curves (planet type):** AP≈1.0 per class; consistent with the CM.
- **RF feature importances & SHAP:** Mass and equilibrium temperature dominate; photometry provides secondary signal.
- **Correlations & missingness:** Atmospheres tables have heavy missingness; planetary features show moderate correlations (e.g., eq. temp ↔ insolation).
- **Molecule multilabel PR:** AP very high for TiO (~0.97) and H (~0.84), moderate for H2O (~0.70).
- **Regression learning curves:** CV R² ~0.999 for eq. temperature and ~0.985–0.99 for mass, improving with data size.

## Task 3 — Habitability
- **Confusion matrices (LGBM/ RF best):** (Near-)perfect diagonals across three classes; class imbalance and potential leakage considered.
- **PR & ROC (LGBM):** Near-ideal separation; AUPRC/AUROC ≈1.0.
- **SHAP & RF importances:** `pl_rade` and `pl_bmasse` are principal drivers; insolation and orbital factors secondary.

## Task 4 — Radio Silence
- **ROC/PR:** AUROC≈0.98, AP≈0.96; operating threshold≈0.66 balances F1≈0.845.
- **Risk histogram & EDA:** Score distribution is well separated; FPP/host-relative components show informative skew.
- **Fairness by Teff:** Comparable risk across stellar-temperature quartiles (no obvious adverse pattern).

## Task 5 — Probe Decision
- **Pair scatter & priority histograms (default vs comms-heavy):** Rankings respond predictably to weight changes; default weights {0.35, 0.35, 0.15, 0.15}.
