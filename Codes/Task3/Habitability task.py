r"""
Task 3 — Habitability Classification (Rebuild)
Datasets: Planetary Systems, Stellar Hosts, Atmospheres (20_08_25)
Author: Praveen Kumar Savariraj, 23089117
Module: 7PAM2002 (UH MSc Data Science) — Final Project

Purpose
=======
Rebuild Task 3 (Habitability Classification) using the three datasets in:
    D:\UH One drive\OneDrive - University of Hertfordshire\Final Project\Data\20_08_25

We produce a clear, viva‑ready pipeline with:
- EDA (distributions, missingness, correlations) and clean merges
- Rule‑based labels (Phase 1)
- Baseline → tuned ML models (Phase 2), with CV and test split
- Multiple metrics (F1, ROC‑AUC, PR‑AUC, confusion matrix)
- Learning curves and hyperparameter tuning impact
- Explainability (feature importances + SHAP for tree models)
- Saved outputs: CSVs, figures, and trained models (joblib)
- Reproducibility (fixed seeds, environment logging), and brief ethics notes

How to run (Spyder / Python)
============================
1) Set BASE_DATA_PATH and OUTPUT_BASE below to match your filesystem.
2) Run the whole script top‑to‑bottom.
3) All outputs will be saved under OUTPUT_BASE/run_YYYYMMDD_HHMMSS/.

Ethics & Data Provenance (for report)
====================================
- Data sources: NASA Exoplanet Archive tables (Planetary Systems, Stellar Hosts) and a derived
  atmospheres table (cleaned_exoplanet_atmospheres_with_molecules_cleaned.csv).
- No personal data involved; GDPR not applicable. We comply with UH Ethics Policy (UPR RE01).
- Cite NASA Exoplanet Archive per their guidelines in the report. Include license/usage terms.
- Computational sustainability: We prefer tree models with moderate complexity; avoid heavy deep learning here.

"""

# =========================
# Imports & configuration
# =========================
import os
import sys
import json
import time
import math
import glob
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional: LightGBM if available
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

# -------------------------
# Paths
# -------------------------
BASE_DATA_PATH = r"D:\UH One drive\OneDrive - University of Hertfordshire\Final Project\Data\20_08_25"
OUTPUT_BASE    = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task3"

FILES = {
    "planetary": "Planetary Systems Composite Data.csv",
    "stellar": "Stellar Hosts.csv",
    "atmo": "cleaned_exoplanet_atmospheres_with_molecules_cleaned.csv",
}

# Create run folder
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_BASE, f"run_{RUN_TS}")
os.makedirs(RUN_DIR, exist_ok=True)
FIG_DIR = os.path.join(RUN_DIR, "figures"); os.makedirs(FIG_DIR, exist_ok=True)
MODEL_DIR = os.path.join(RUN_DIR, "models"); os.makedirs(MODEL_DIR, exist_ok=True)
CSV_DIR = os.path.join(RUN_DIR, "csv"); os.makedirs(CSV_DIR, exist_ok=True)

# Seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================
# Utility helpers
# =========================

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()


def log_versions():
    info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__('sklearn').__version__,
        "seaborn": sns.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "lightgbm_available": LGBM_AVAILABLE
    }
    with open(os.path.join(RUN_DIR, "env_versions.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("[ENV] Versions logged →", os.path.join(RUN_DIR, "env_versions.json"))


# =========================
# Load data
# =========================
print("[LOAD] Reading datasets from:", BASE_DATA_PATH)
ps_path = os.path.join(BASE_DATA_PATH, FILES["planetary"])  # 5917 x 84
sh_path = os.path.join(BASE_DATA_PATH, FILES["stellar"])    # 46301 x 45
at_path = os.path.join(BASE_DATA_PATH, FILES["atmo"])       # 981 x 28

ps = pd.read_csv(ps_path)
sh = pd.read_csv(sh_path)
at = pd.read_csv(at_path)
print(f"[LOAD] planetary systems: {ps.shape} | stellar hosts: {sh.shape} | atmospheres: {at.shape}")

# =========================
# EDA — High level overview
# =========================
# Save head/tail/sample for quick inspection without opening CSVs.
ps.head(20).to_csv(os.path.join(CSV_DIR, "ps_head.csv"), index=False)
sh.head(20).to_csv(os.path.join(CSV_DIR, "sh_head.csv"), index=False)
at.head(20).to_csv(os.path.join(CSV_DIR, "at_head.csv"), index=False)

# Plot missingness (top 40 columns with most NaNs) for each table
for name, df in {"planetary": ps, "stellar": sh, "atmo": at}.items():
    null_pct = df.isna().mean().sort_values(ascending=False)[:40]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=null_pct.values, y=null_pct.index)
    plt.xlabel("Fraction Missing")
    plt.ylabel("Column")
    plt.title(f"Missingness Top 40 — {name}")
    savefig(os.path.join(FIG_DIR, f"eda_missingness_top40_{name}.png"))

# Key numeric distributions (subset of columns likely useful)
ps_numeric_cols = [
    c for c in ["pl_rade","pl_bmasse","pl_eqt","pl_insol","pl_orbsmax","pl_orbeccen"] if c in ps.columns
]
sh_numeric_cols = [
    c for c in ["st_teff","st_rad","st_mass","st_met","st_logg","sy_dist"] if c in sh.columns
]
plt.figure(figsize=(10, 6))
for i, c in enumerate(ps_numeric_cols):
    plt.subplot(math.ceil(len(ps_numeric_cols)/3), 3, i+1)
    sns.histplot(ps[c].dropna(), bins=40, kde=True)
    plt.xlabel(c)
plt.suptitle("Planetary — Numeric Distributions")
savefig(os.path.join(FIG_DIR, "eda_planetary_numeric_hist.png"))

plt.figure(figsize=(10, 6))
for i, c in enumerate(sh_numeric_cols):
    plt.subplot(math.ceil(len(sh_numeric_cols)/3), 3, i+1)
    sns.histplot(sh[c].dropna(), bins=40, kde=True)
    plt.xlabel(c)
plt.suptitle("Stellar — Numeric Distributions")
savefig(os.path.join(FIG_DIR, "eda_stellar_numeric_hist.png"))

# Correlation heatmap for planetary numerics
if len(ps_numeric_cols) >= 2:
    plt.figure(figsize=(8, 6))
    corr = ps[ps_numeric_cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", square=True)
    plt.title("Planetary numeric correlation")
    savefig(os.path.join(FIG_DIR, "eda_planetary_corr_heatmap.png"))

# =========================
# Merge tables (pl_name + hostname)
# =========================
# - ps has [pl_name, hostname, pl_*]
# - sh has [hostname, st_*]
# - at has [pl_name, molecule_list] (and other fields) — we only join molecule list as feature/context

stellar_keep = [
    c for c in ["hostname", "st_teff","st_rad","st_mass","st_met","st_logg","st_spectype","sy_dist","sy_vmag","sy_kmag","sy_gaiamag"]
    if c in sh.columns
]
ps_keep = [
    c for c in ["pl_name","hostname","discoverymethod","disc_year","pl_rade","pl_bmasse","pl_eqt","pl_insol","pl_orbsmax","pl_orbeccen"]
    if c in ps.columns
]

ps_sub = ps[ps_keep].copy()
sh_sub = sh[stellar_keep].copy()

merged = ps_sub.merge(sh_sub, on="hostname", how="left")
if "pl_name" in at.columns:
    at_sub = at[[c for c in ("pl_name","molecule_list") if c in at.columns]].copy()
    merged = merged.merge(at_sub, on="pl_name", how="left")

print("[MERGE] merged shape:", merged.shape)
merged.to_csv(os.path.join(CSV_DIR, "merged_raw.csv"), index=False)

# =========================
# Phase 1 — Rule-based labels
# =========================
# Heuristics (explainable in viva):
# - Rocky size: 0.5 ≤ pl_rade ≤ 1.6 (Earth radii)
# - Moderately massive: 0.5 ≤ pl_bmasse ≤ 5 (Earth masses)
# - Temperate band (either criterion ok):
#       0.35 ≤ pl_insol ≤ 1.5   OR   180 K ≤ pl_eqt ≤ 310 K
# - Low eccentricity preferred: pl_orbeccen ≤ 0.5 (if available)
# Labels:
#   Habitable (Potentially) : all 3 primary criteria (size + mass + temperate) met and (ecc ok or missing)
#   Partially Habitable     : exactly 2 of the 3 primary criteria met
#   Non-Habitable           : otherwise


def in_range(x, lo, hi):
    if pd.isna(x):
        return False
    return (x >= lo) and (x <= hi)


def compute_rule_label(row):
    size_ok = in_range(row.get("pl_rade", np.nan), 0.5, 1.6)
    mass_ok = in_range(row.get("pl_bmasse", np.nan), 0.5, 5.0)
    temp_ok = False
    if "pl_insol" in row and not pd.isna(row["pl_insol"]):
        temp_ok = (row["pl_insol"] >= 0.35) and (row["pl_insol"] <= 1.5)
    if (not temp_ok) and ("pl_eqt" in row) and (not pd.isna(row["pl_eqt"])):
        temp_ok = (row["pl_eqt"] >= 180) and (row["pl_eqt"] <= 310)

    primary_hits = sum([size_ok, mass_ok, temp_ok])

    ecc_ok = True
    if "pl_orbeccen" in row and not pd.isna(row["pl_orbeccen"]):
        ecc_ok = row["pl_orbeccen"] <= 0.5

    if primary_hits == 3 and ecc_ok:
        return "Habitable (Potentially)"
    elif primary_hits == 2:
        return "Partially Habitable"
    else:
        return "Non-Habitable"


merged["habitability_label_rule"] = merged.apply(compute_rule_label, axis=1)

# Save rule-based label distribution
label_counts = merged["habitability_label_rule"].value_counts(dropna=False)
label_counts.to_csv(os.path.join(CSV_DIR, "rule_label_distribution.csv"))
print("[RULE] Label distribution:\n", label_counts)

# =========================
# Phase 2 — ML classification
# =========================
# Target: habitability_label_rule
# Features: planetary + stellar numerics (robust to missing via median impute)

TARGET = "habitability_label_rule"
FEATS = [
    c for c in [
        "pl_rade","pl_bmasse","pl_eqt","pl_insol","pl_orbsmax","pl_orbeccen",
        "st_teff","st_rad","st_mass","st_met","st_logg","sy_dist","sy_vmag","sy_kmag","sy_gaiamag"
    ] if c in merged.columns
]

ml_df = merged.dropna(subset=[TARGET]).copy()
X = ml_df[FEATS]
y = ml_df[TARGET]

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Preprocess: median imputation for numerics (no scaling for trees; scaling for LR)
num_imputer = SimpleImputer(strategy='median')

# ---------- Baseline: Logistic Regression (multinomial) ----------
lr_clf = Pipeline([
    ('impute', num_imputer),
    ('scale', StandardScaler(with_mean=True)),
    ('clf', LogisticRegression(max_iter=2000, multi_class='auto', random_state=RANDOM_STATE))
])

print("[MODEL] Training baseline Logistic Regression…")
lr_clf.fit(X_train, y_train)

# Evaluate helper

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, labels):
    # Predictions
    y_pred_tr = model.predict(X_tr)
    y_pred_te = model.predict(X_te)

    # Metrics
    acc_tr = accuracy_score(y_tr, y_pred_tr)
    acc_te = accuracy_score(y_te, y_pred_te)

    precision, recall, f1, _ = precision_recall_fscore_support(y_te, y_pred_te, average='weighted', zero_division=0)

    # One-vs-rest ROC/PR
    classes_ = np.unique(y_te)
    y_bin_tr = label_binarize(y_tr, classes=classes_)
    y_bin_te = label_binarize(y_te, classes=classes_)

    # Ensure we have predict_proba
    if hasattr(model, "predict_proba"):
        proba_te = model.predict_proba(X_te)
    else:
        # fallback via decision_function if available
        if hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_te)
            # Convert to pseudo-prob via softmax
            e = np.exp(df_vals - df_vals.max(axis=1, keepdims=True))
            proba_te = e / e.sum(axis=1, keepdims=True)
        else:
            proba_te = None

    roc_auc_macro = np.nan
    pr_auc_macro = np.nan

    if proba_te is not None and y_bin_te.shape[1] > 1:
        roc_aucs, pr_aucs = [], []
        for i in range(y_bin_te.shape[1]):
            fpr, tpr, _ = roc_curve(y_bin_te[:, i], proba_te[:, i])
            roc_aucs.append(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(y_bin_te[:, i], proba_te[:, i])
            pr_aucs.append(auc(rec, prec))
        roc_auc_macro = float(np.nanmean(roc_aucs))
        pr_auc_macro = float(np.nanmean(pr_aucs))

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred_te, labels=labels)

    # Save metrics
    metrics_row = {
        "model": name,
        "acc_train": acc_tr,
        "acc_test": acc_te,
        "precision_w": precision,
        "recall_w": recall,
        "f1_w": f1,
        "roc_auc_macro": roc_auc_macro,
        "pr_auc_macro": pr_auc_macro,
    }
    return metrics_row, cm, y_pred_te, proba_te

labels_order = ["Non-Habitable", "Partially Habitable", "Habitable (Potentially)"]

results = []
cm_store = {}

m_lr, cm_lr, ypred_lr, proba_lr = evaluate_model("LogReg", lr_clf, X_train, y_train, X_test, y_test, labels_order)
results.append(m_lr); cm_store["LogReg"] = cm_lr

# Plot learning curve for LR
train_sizes, train_scores, test_scores = learning_curve(
    lr_clf, X_train, y_train, cv=5, scoring='f1_weighted', n_jobs=-1, train_sizes=np.linspace(0.2, 1.0, 5), random_state=RANDOM_STATE
)
plt.figure(figsize=(7,5))
plt.plot(train_sizes, np.mean(train_scores, axis=1), marker='o', label='Train')
plt.plot(train_sizes, np.mean(test_scores, axis=1), marker='s', label='CV')
plt.xlabel('Training examples'); plt.ylabel('F1 (weighted)'); plt.title('Learning Curve — Logistic Regression')
plt.legend()
savefig(os.path.join(FIG_DIR, "learning_curve_logreg.png"))

# ---------- Random Forest (baseline + tuning) ----------
rf_base = Pipeline([
    ('impute', num_imputer),
    ('clf', RandomForestClassifier(random_state=RANDOM_STATE))
])
print("[MODEL] Training RandomForest (baseline)…")
rf_base.fit(X_train, y_train)

m_rf_base, cm_rf_base, ypred_rf_base, proba_rf_base = evaluate_model("RF_base", rf_base, X_train, y_train, X_test, y_test, labels_order)
results.append(m_rf_base); cm_store["RF_base"] = cm_rf_base

# RF tuning
rf_param_grid = {
    'clf__n_estimators': [150, 300],
    'clf__max_depth': [None, 8, 16],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print("[TUNE] GridSearchCV for RandomForest…")
rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
print("[TUNE] RF best params:", rf_grid.best_params_)

# Save CV results
pd.DataFrame(rf_grid.cv_results_).to_csv(os.path.join(CSV_DIR, "cv_results_rf.csv"), index=False)

m_rf, cm_rf, ypred_rf, proba_rf = evaluate_model("RF_best", rf_best, X_train, y_train, X_test, y_test, labels_order)
results.append(m_rf); cm_store["RF_best"] = cm_rf

# Feature importances (RF)
rf_clf = rf_best.named_steps['clf']
if hasattr(rf_clf, 'feature_importances_'):
    importances = pd.Series(rf_clf.feature_importances_, index=FEATS).sort_values(ascending=False)
    importances.to_csv(os.path.join(CSV_DIR, "rf_feature_importances.csv"))
    plt.figure(figsize=(8,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.xlabel('Importance'); plt.ylabel('Feature'); plt.title('RandomForest Feature Importances')
    savefig(os.path.join(FIG_DIR, "rf_feature_importances.png"))

# ---------- LightGBM (baseline + tuning) ----------
if LGBM_AVAILABLE:
    lgb_base = Pipeline([
        ('impute', num_imputer),
        ('clf', LGBMClassifier(random_state=RANDOM_STATE, n_estimators=300))
    ])
    print("[MODEL] Training LightGBM (baseline)…")
    lgb_base.fit(X_train, y_train)

    m_lgb_base, cm_lgb_base, ypred_lgb_base, proba_lgb_base = evaluate_model("LGBM_base", lgb_base, X_train, y_train, X_test, y_test, labels_order)
    results.append(m_lgb_base); cm_store["LGBM_base"] = cm_lgb_base

    lgb_param_grid = {
        'clf__num_leaves': [15, 31, 63],
        'clf__max_depth': [-1, 8, 16],
        'clf__learning_rate': [0.05, 0.1],
        'clf__n_estimators': [300, 600]
    }
    print("[TUNE] GridSearchCV for LightGBM…")
    lgb_grid = GridSearchCV(lgb_base, lgb_param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)
    lgb_grid.fit(X_train, y_train)
    lgb_best = lgb_grid.best_estimator_
    print("[TUNE] LGBM best params:", lgb_grid.best_params_)
    pd.DataFrame(lgb_grid.cv_results_).to_csv(os.path.join(CSV_DIR, "cv_results_lgbm.csv"), index=False)

    m_lgb, cm_lgb, ypred_lgb, proba_lgb = evaluate_model("LGBM_best", lgb_best, X_train, y_train, X_test, y_test, labels_order)
    results.append(m_lgb); cm_store["LGBM_best"] = cm_lgb

    # SHAP for LGBM
    try:
        # Fit an imputer and transform training data for explainer
        X_train_imp = num_imputer.fit_transform(X_train)
        lgb_final = lgb_best.named_steps['clf']
        explainer = shap.TreeExplainer(lgb_final)
        shap_values = explainer.shap_values(X_train_imp)
        shap_sum = np.sum(np.abs(shap_values[0]), axis=0) if isinstance(shap_values, list) else np.sum(np.abs(shap_values), axis=0)
        # Save SHAP bar summary
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, X_train_imp, feature_names=FEATS, show=False)
        savefig(os.path.join(FIG_DIR, "lgbm_shap_summary.png"))
    except Exception as e:
        print("[SHAP] LightGBM SHAP failed:", e)

# =========================
# Plots: Confusion matrices, ROC/PR curves
# =========================

def plot_confusion_matrix(cm, labels, title, outpath):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    savefig(outpath)

for name, cm in cm_store.items():
    plot_confusion_matrix(cm, labels_order, f"Confusion Matrix — {name}", os.path.join(FIG_DIR, f"cm_{name}.png"))

# ROC/PR curves for the best performing model (by F1_w on test)
res_df = pd.DataFrame(results).sort_values('f1_w', ascending=False)
res_df.to_csv(os.path.join(CSV_DIR, "metrics_summary.csv"), index=False)
print("[RESULTS] Summary (sorted by F1_w):\n", res_df)

best_name = res_df.iloc[0]['model']
print(f"[BEST] Using {best_name} for ROC/PR plots…")

prob_map = {"LogReg": proba_lr, "RF_base": proba_rf_base, "RF_best": proba_rf}
if LGBM_AVAILABLE:
    prob_map.update({"LGBM_base": proba_lgb_base, "LGBM_best": proba_lgb})

best_proba = prob_map.get(best_name, None)

if best_proba is not None:
    classes_ = np.unique(y_test)
    y_bin_te = label_binarize(y_test, classes=classes_)
    # ROC
    plt.figure(figsize=(7,5))
    for i, cls in enumerate(classes_):
        fpr, tpr, _ = roc_curve(y_bin_te[:, i], best_proba[:, i])
        plt.plot(fpr, tpr, label=f"{cls}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC — {best_name}')
    plt.legend()
    savefig(os.path.join(FIG_DIR, f"roc_{best_name}.png"))

    # PR
    plt.figure(figsize=(7,5))
    for i, cls in enumerate(classes_):
        prec, rec, _ = precision_recall_curve(y_bin_te[:, i], best_proba[:, i])
        plt.plot(rec, prec, label=f"{cls}")
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR — {best_name}')
    plt.legend()
    savefig(os.path.join(FIG_DIR, f"pr_{best_name}.png"))

# =========================
# Save predictions & models
# =========================
# Consolidate predictions for key models
pred_df = merged.copy()

# Helper to add predictions (probabilities optional)

def add_preds(prefix, model, X_in):
    try:
        pred = model.predict(X_in)
        pred_proba = model.predict_proba(X_in) if hasattr(model, 'predict_proba') else None
        pred_df[f"{prefix}_pred"] = np.nan
        pred_df.loc[X_in.index, f"{prefix}_pred"] = pred
        if pred_proba is not None:
            for i, cls in enumerate(np.unique(y)):
                pred_df.loc[X_in.index, f"{prefix}_proba_{cls}"] = pred_proba[:, i]
    except Exception as e:
        print(f"[WARN] Could not add predictions for {prefix}:", e)

# Fit on full data (X,y) for final predictions per model
lr_clf.fit(X, y); add_preds("lr", lr_clf, X)
rf_best.fit(X, y); add_preds("rf", rf_best, X)
if LGBM_AVAILABLE:
    lgb_best.fit(X, y); add_preds("lgbm", lgb_best, X)

# Save merged + predictions
out_pred_path = os.path.join(CSV_DIR, "task3_habitability_with_predictions.csv")
pred_df.to_csv(out_pred_path, index=False)
print("[SAVE] Predictions →", out_pred_path)

# Save models
joblib.dump(lr_clf, os.path.join(MODEL_DIR, "logreg.joblib"))
joblib.dump(rf_best, os.path.join(MODEL_DIR, "random_forest_best.joblib"))
if LGBM_AVAILABLE:
    joblib.dump(lgb_best, os.path.join(MODEL_DIR, "lightgbm_best.joblib"))

# =========================
# Fairness / subgroup analysis (by stellar spectral type initial)
# =========================
if "st_spectype" in merged.columns:
    def spec_initial(x):
        if isinstance(x, str) and len(x) > 0:
            return x[0].upper()
        return np.nan
    ml_df["spec_initial"] = merged.loc[ml_df.index, "st_spectype"].apply(spec_initial)
    groups = ml_df["spec_initial"].dropna().unique()

    subgroup_rows = []
    for g in sorted(groups):
        idx = ml_df.index[ml_df["spec_initial"] == g]
        if len(idx) < 15:
            continue
        y_true = y.loc[idx]
        y_pred = rf_best.predict(X.loc[idx])
        pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        subgroup_rows.append({"spec_type": g, "n": len(idx), "precision_w": pr, "recall_w": rc, "f1_w": f1})
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(os.path.join(CSV_DIR, "subgroup_performance_by_spec.csv"), index=False)

# =========================
# Final notes printed to console
# =========================
log_versions()
print("\n[DONE] Outputs saved under:", RUN_DIR)
print("- Figures →", FIG_DIR)
print("- CSVs    →", CSV_DIR)
print("- Models  →", MODEL_DIR)
print("\nIf you see unrealistically high metrics, check for leakage (e.g., duplicate targets, overlapping rows after split).")

