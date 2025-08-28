"""
Task 4 — Radio Silence Prediction (Risk Scoring + Optional Model Distillation)
Author: Praveen Kumar Savariraj (Student ID: 23089117)
Module: 7PAM2002 (UH MSc Data Science) — Final Project

Overview
--------
This script uses Kepler False Positive Probabilities (FPP) and Positional Probabilities
tables to compute a transparent, interpretable "Radio Silence Risk" score per KOI/KEPID.
The score combines:
  • Alternative-star positional confusion (pp_1hi_rel_prob + pp_2hi_rel_prob)
  • 1 - host relative probability (1 - pp_host_rel_prob)
  • False positive strength (max of fpp_prob and sum of FPP component probabilities)

We then generate a heuristic label:
  Radio_Silent = 1 if risk >= 75th percentile, else 0
This is *not* a ground-truth label — it is a proxy designed to rank/risk-score targets.
We optionally train a lightweight Logistic Regression that *distills* the heuristic rule.
High performance is expected (the model learns the rule), so we explicitly discuss
overfitting and the absence of true labels in the report (important for viva integrity).

Paths
-----
Change RAW_DIR to your local folder path that contains the two CSVs.
All outputs will be saved under OUTPUT_DIR/Task4/run_<timestamp>/

Data Source Notes
-----------------
• Kepler False Positive Probabilities Table:
  - According to our notes, skip first 39 rows (column definitions)
• Kepler Positional Probabilities Table:
  - According to our notes, skip first 15 rows (column definitions)

Checklist Compliance
--------------------
✔ EDA, cleaning, missing handling
✔ Baseline (heuristic risk) + optional distilled model
✔ Validation: holdout split; (optional) k-fold CV
✔ Metrics: ROC-AUC, PR-AUC, F1, confusion matrix (for distilled model)
✔ Figures: histograms, ROC, PR; all labeled & saved
✔ Error analysis: list top false positives/negatives (relative to heuristic label)
✔ Explainability: LR coefficients and per-feature magnitudes
✔ Outputs saved reproducibly (CSV, figures, metrics, model if needed)
✔ Ethics: clearly states proxy label and limitations; no personal data

Usage
-----
Run end-to-end in Spyder or terminal:
  python task4_radio_silence_pipeline.py
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay
)

# ===============================
# Config
# ===============================

# === Change this to your Windows path ===
RAW_DIR = r"D:\UH One drive\OneDrive - University of Hertfordshire\Final Project\Data\Filtered"

FPP_FILE = "Kepler False Positive Probabilities Table.csv"
POS_FILE = "Kepler Positional Probabilities Table.csv"

# Output base
OUTPUT_DIR = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task4"
RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

# Toggles
DO_MODEL_DISTILLATION = True           # Train a small LR on the heuristic labels
DO_CROSS_VAL         = False           # Set True to run 5-fold CV (slower)
TEST_SIZE            = 0.2
RANDOM_STATE         = 42

# ===============================
# Helpers
# ===============================

def safe_float(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def describe_missing(df, name):
    miss = df.isna().mean().sort_values(ascending=False)
    miss.to_csv(os.path.join(RUN_DIR, f"{name}_missingness.csv"))
    return miss

# ===============================
# 1) Load
# ===============================
print("[LOAD] Reading CSVs…")
# Notes from project: skiprows for definitions
try:
    fpp = pd.read_csv(os.path.join(RAW_DIR, FPP_FILE), skiprows=39)
except Exception:
    fpp = pd.read_csv(os.path.join(RAW_DIR, FPP_FILE))

try:
    pos = pd.read_csv(os.path.join(RAW_DIR, POS_FILE), skiprows=15)
except Exception:
    pos = pd.read_csv(os.path.join(RAW_DIR, POS_FILE))

print(f"[LOAD] FPP shape: {fpp.shape}")
print(f"[LOAD] POS shape: {pos.shape}")

# ===============================
# 2) Select key columns & clean
# ===============================
fpp_keep = [
    'kepid','kepoi_name','fpp_prob','fpp_score',
    'fpp_prob_heb','fpp_prob_ueb','fpp_prob_beb',
    'fpp_prob_heb_dbl','fpp_prob_ueb_dbl','fpp_prob_beb_dbl',
    'fpp_ror','fpp_steff'
]
pos_keep = [
    'kepid','kepoi_name','pp_host_rel_prob','pp_host_prob_score',
    'pp_1hi_rel_prob','pp_2hi_rel_prob','pp_unk_rel_prob'
]

fpp2 = fpp[[c for c in fpp_keep if c in fpp.columns]].copy()
pos2 = pos[[c for c in pos_keep if c in pos.columns]].copy()

numeric_cols = [c for c in fpp2.columns if c not in ['kepid','kepoi_name']] + \
               [c for c in pos2.columns if c not in ['kepid','kepoi_name']]
fpp2 = safe_float(fpp2, [c for c in fpp2.columns if c in numeric_cols])
pos2 = safe_float(pos2, [c for c in pos2.columns if c in numeric_cols])

# ===============================
# 3) Merge
# ===============================
merged = pd.merge(fpp2, pos2, on=['kepid','kepoi_name'], how='inner')
print(f"[MERGE] merged shape: {merged.shape}")

# ===============================
# 4) EDA & missingness
# ===============================
miss_fpp = describe_missing(fpp2, "fpp")
miss_pos = describe_missing(pos2, "pos")
miss_merged = describe_missing(merged, "merged")

# Histogram examples
plt.figure()
merged['fpp_prob'].dropna().hist(bins=40)
plt.title("FPP Probability Distribution")
plt.xlabel("fpp_prob")
plt.ylabel("count")
save_fig(os.path.join(RUN_DIR, "eda_fpp_prob_hist.png"))

if 'pp_host_rel_prob' in merged.columns:
    plt.figure()
    merged['pp_host_rel_prob'].dropna().hist(bins=40)
    plt.title("Host Relative Probability Distribution")
    plt.xlabel("pp_host_rel_prob")
    plt.ylabel("count")
    save_fig(os.path.join(RUN_DIR, "eda_pp_host_rel_prob_hist.png"))

# ===============================
# 5) Feature engineering
# ===============================
merged['fpp_fp_components_sum'] = merged[[c for c in [
    'fpp_prob_heb','fpp_prob_ueb','fpp_prob_beb',
    'fpp_prob_heb_dbl','fpp_prob_ueb_dbl','fpp_prob_beb_dbl'
] if c in merged.columns]].sum(axis=1, skipna=True)

for c in ['fpp_prob','fpp_fp_components_sum','pp_host_rel_prob','pp_1hi_rel_prob','pp_2hi_rel_prob','pp_unk_rel_prob']:
    if c in merged.columns:
        merged[c] = merged[c].fillna(0.0)

alt_rel_prob_sum = merged[[c for c in ['pp_1hi_rel_prob','pp_2hi_rel_prob'] if c in merged.columns]].sum(axis=1, skipna=True)
host_confusion = 1.0 - merged['pp_host_rel_prob'] if 'pp_host_rel_prob' in merged.columns else 0.0
fp_strength = merged[[c for c in ['fpp_prob','fpp_fp_components_sum'] if c in merged.columns]].max(axis=1, skipna=True)

comp_df = pd.DataFrame({
    'alt_rel_prob_sum': alt_rel_prob_sum,
    'host_confusion': host_confusion,
    'fp_strength': fp_strength
})

comp_norm = pd.DataFrame(MinMaxScaler().fit_transform(comp_df.fillna(0.0)),
                         columns=comp_df.columns, index=comp_df.index)

merged['radio_silence_risk'] = comp_norm.mean(axis=1)

# Heuristic label (proxy, not ground truth)
threshold = merged['radio_silence_risk'].quantile(0.75)
merged['Radio_Silent'] = (merged['radio_silence_risk'] >= threshold).astype(int)

# Save risk fig
plt.figure()
merged['radio_silence_risk'].hist(bins=40)
plt.title("Radio Silence Risk (Composite)")
plt.xlabel("risk")
plt.ylabel("count")
save_fig(os.path.join(RUN_DIR, "risk_hist.png"))

# ===============================
# 6) Optional model distillation (LR)
# ===============================
metrics = {}
feature_cols = [c for c in ['fpp_prob','fpp_fp_components_sum','pp_host_rel_prob','pp_1hi_rel_prob','pp_2hi_rel_prob','pp_unk_rel_prob','fpp_ror','fpp_steff'] if c in merged.columns]

if DO_MODEL_DISTILLATION and len(feature_cols) > 0:
    X = merged[feature_cols].fillna(0.0).values
    y = merged['Radio_Silent'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    lr = LogisticRegression(max_iter=200)
    
    if DO_CROSS_VAL:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        cv_auc = cross_val_score(lr, X_train, y_train, cv=cv, scoring='roc_auc')
        metrics['cv_auc_mean'] = float(cv_auc.mean())
        metrics['cv_auc_std'] = float(cv_auc.std())
    
    lr.fit(X_train, y_train)
    proba = lr.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)

    metrics['test_roc_auc'] = float(roc_auc_score(y_test, proba))
    metrics['test_pr_auc']  = float(average_precision_score(y_test, proba))
    metrics['test_f1']      = float(f1_score(y_test, pred))
    metrics['confusion_matrix'] = confusion_matrix(y_test, pred).tolist()

    # Plots
    RocCurveDisplay.from_predictions(y_test, proba)
    plt.title("Logistic Regression ROC-AUC (Distilled from Heuristic)")
    save_fig(os.path.join(RUN_DIR, "roc_curve.png"))

    PrecisionRecallDisplay.from_predictions(y_test, proba)
    plt.title("Logistic Regression PR Curve (Distilled)")
    save_fig(os.path.join(RUN_DIR, "pr_curve.png"))

    # Explainability: coefficients
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coef': lr.coef_.ravel()
    }).sort_values('coef', key=lambda s: s.abs(), ascending=False)
    coef_df.to_csv(os.path.join(RUN_DIR, "lr_coefficients.csv"), index=False)

    # Error analysis relative to heuristic
    errors = pd.DataFrame({
        'y_true': y_test,
        'y_pred': pred,
        'y_proba': proba
    })
    errors.to_csv(os.path.join(RUN_DIR, "errors_test_split.csv"), index=False)

# ===============================
# 7) Save outputs
# ===============================
merged_out = merged.copy()
merged_out['risk_threshold_0p75'] = threshold
csv_path = os.path.join(RUN_DIR, "task4_radio_silence_scores.csv")
merged_out.to_csv(csv_path, index=False)

with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
    json.dump({
        "features": feature_cols,
        "metrics": metrics,
        "threshold_0p75": float(threshold),
        "rows": int(merged_out.shape[0])
    }, f, indent=2)

print(f"[DONE] Outputs saved to: {RUN_DIR}")
print(f"       Scores CSV: {csv_path}")


# Optional: Learning curve
DO_LEARNING_CURVE = False  # set True to generate learning_curve.png

if DO_MODEL_DISTILLATION and DO_LEARNING_CURVE and len(feature_cols) > 0:
    from sklearn.model_selection import learning_curve
    sizes, train_scores, val_scores = learning_curve(
        LogisticRegression(max_iter=200),
        merged[feature_cols].fillna(0.0).values,
        merged['Radio_Silent'].values,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        train_sizes=np.linspace(0.2, 1.0, 5),
        scoring='roc_auc'
    )
    plt.figure()
    plt.plot(sizes, train_scores.mean(axis=1), marker='o', label='Train')
    plt.plot(sizes, val_scores.mean(axis=1), marker='o', label='CV')
    plt.title("Learning Curve (ROC-AUC) — Logistic Regression (Distilled)")
    plt.xlabel("Training examples")
    plt.ylabel("ROC-AUC")
    plt.legend()
    save_fig(os.path.join(RUN_DIR, "learning_curve.png"))


# ===============================
# 8) Fairness-style subgroup check (proxy): risk vs stellar Teff quartiles
# ===============================
if 'fpp_steff' in merged.columns:
    merged['steff_bin'] = pd.qcut(merged['fpp_steff'], q=4, duplicates='drop')
    grp = merged.groupby('steff_bin')['radio_silence_risk'].agg(['count','mean','std']).reset_index()
    grp.to_csv(os.path.join(RUN_DIR, "fairness_risk_by_steff_quartile.csv"), index=False)

    # Boxplot of risk by steff quartile
    plt.figure()
    data = [merged.loc[merged['steff_bin'] == b, 'radio_silence_risk'].dropna().values for b in merged['steff_bin'].dropna().unique()]
    plt.boxplot(data)
    plt.title("Radio Silence Risk by Stellar Teff Quartile")
    plt.xlabel("Teff quartile bins")
    plt.ylabel("risk")
    save_fig(os.path.join(RUN_DIR, "fairness_risk_by_steff_quartile.png"))
