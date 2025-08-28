r"""
Task 3 — Habitability Classification (Dual runs: 3A vs 3B)
----------------------------------------------------------
Deliverables this script saves:

Tables
- csv/metrics_summary__3A.csv       (rule-replication run)
- csv/metrics_summary__3B.csv       (robust run)
- csv/metrics_comparison_3A_3B.csv  (side-by-side best-model summary)

Figures (for the best model in each run)
- figures/3A__cm__<BestModel>.png                (confusion matrix)
- figures/3A__roc__<BestModel>__<Class>.png      (per-class ROC)
- figures/3A__pr__<BestModel>__<Class>.png       (per-class Precision-Recall)
- figures/3B__cm__<BestModel>.png
- figures/3B__roc__<BestModel>__<Class>.png
- figures/3B__pr__<BestModel>__<Class>.png

Models
- models/3A__<BestModel>.joblib
- models/3B__<BestModel>.joblib
- models/logreg__baseline.joblib
- models/rf__best.joblib
- (optional) models/lgbm__best.joblib

Notes
- notes__why_3B_preferred.txt   (short discussion for your report/viva)

Run-time choices:
- Robust run (3B) uses StratifiedGroupKFold (falls back gracefully) and
  **excludes** rule-driving features to avoid “learning the rule verbatim”.

Author: YOU
"""
from __future__ import annotations

import json
import os
import sys
import time
import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 125})

# ======================================================================
# Paths — EDIT if needed
# ======================================================================
BASE_DATA_PATH = r"D:/UH One drive/OneDrive - University of Hertfordshire/Final Project/Data/20_08_25"
OUTPUT_BASE    = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task3"
TASK2_FEATURES_PARQUET = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task2/latest/features_task2.parquet"  # or None

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ======================================================================
# Small utilities
# ======================================================================
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return p

def ts_run_dir(base: str) -> Dict[str, str]:
    ts = time.strftime("run_%Y%m%d_%H%M%S")
    root = os.path.join(base, ts)
    paths = {"root": root,
             "figures": os.path.join(root, "figures"),
             "csv": os.path.join(root, "csv"),
             "models": os.path.join(root, "models")}
    for v in paths.values(): _ensure_dir(v)
    return paths

def env_versions(out_path: str) -> None:
    import sklearn, numpy, pandas
    info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import lightgbm as lgb
        info["lightgbm"] = lgb.__version__
    except Exception:
        info["lightgbm"] = None
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)

# label normalisation (fix weird hyphen)
def clean_label(s: str) -> str:
    if not isinstance(s, str): return s
    s = s.replace("Non-Habitable", "Non-Habitable")   # non-breaking hyphen → ascii hyphen
    s = s.replace("Non–Habitable", "Non-Habitable")   # en-dash → hyphen
    return s

# ======================================================================
# Data loading & merging (robust to alt column names)
# ======================================================================
def load_datasets(base_path: str) -> Dict[str, pd.DataFrame]:
    d = {}
    d["planetary"] = pd.read_csv(os.path.join(base_path, "Planetary Systems Composite Data.csv"))
    d["stellar"]   = pd.read_csv(os.path.join(base_path, "Stellar Hosts.csv"))
    d["atmo"]      = pd.read_csv(os.path.join(base_path, "cleaned_exoplanet_atmospheres_with_molecules_cleaned.csv"))
    return d

def merge_all(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    p, s, a = d["planetary"].copy(), d["stellar"].copy(), d["atmo"].copy()
    # name harmonisation
    for df in (p, s, a):
        if "pl_name" not in df and "planet_name" in df: df.rename(columns={"planet_name":"pl_name"}, inplace=True)
        if "hostname" not in df:
            for cand in ["host_name","star_name","stellar_host","hostname"]:
                if cand in df.columns: df.rename(columns={cand:"hostname"}, inplace=True); break
    # merge
    m = p.merge(s, on="hostname", how="left", suffixes=("", "_host")) if "hostname" in p and "hostname" in s else p
    if "pl_name" in m and "pl_name" in a:
        m = m.merge(a, on="pl_name", how="left", suffixes=("", "_atmo"))
    elif "planet_name" in a and "pl_name" in m:
        m = m.merge(a.rename(columns={"planet_name":"pl_name"}), on="pl_name", how="left", suffixes=("", "_atmo"))
    return m

def try_merge_task2_features(df: pd.DataFrame, features_path: Optional[str], on: str = "pl_name") -> pd.DataFrame:
    try:
        if features_path and os.path.exists(features_path):
            f2 = pd.read_parquet(features_path)
            if on in df.columns and on in f2.columns:
                return df.merge(f2, on=on, how="left", suffixes=("", "_t2"))
    except Exception:
        pass
    return df

# ======================================================================
# Rule label (transparent baseline) — returns ASCII-safe labels
# ======================================================================
def label_habitability(row: pd.Series) -> str:
    r = row.get("pl_rade", row.get("radius", np.nan))          # Earth radii
    m = row.get("pl_bmasse", row.get("mass", np.nan))          # Earth masses
    teq = row.get("pl_eqt", row.get("temp_calculated", np.nan))
    ins = row.get("pl_insol", np.nan)
    ecc = row.get("pl_orbeccen", np.nan)

    cond_radius = (0.5 <= r <= 1.8) if pd.notna(r) else False
    cond_mass   = (0.5 <= m <= 5.0) if pd.notna(m) else False
    cond_flux   = (0.35 <= ins <= 1.5) if pd.notna(ins) else None
    cond_temp   = (180 <= teq <= 310) if pd.notna(teq) else None
    climate_ok  = cond_flux if cond_flux is not None else cond_temp if cond_temp is not None else False
    cond_ecc    = (ecc <= 0.5) if pd.notna(ecc) else True

    if cond_radius and cond_mass and climate_ok and cond_ecc:
        return "Habitable (Potentially)"
    if (cond_radius or cond_mass) and (climate_ok or cond_ecc):
        return "Partially Habitable"
    return "Non-Habitable"  # ASCII hyphen

# ======================================================================
# Preprocessing & evaluation
# ======================================================================
def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale_numeric: bool = False) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    # OneHotEncoder: support both sklearn<1.2 (sparse) and ≥1.2 (sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
    num_pipeline = Pipeline(num_steps)
    return ColumnTransformer([("num", num_pipeline, num_cols),
                              ("cat", cat_pipeline, cat_cols)], remainder="drop")

def get_feature_names(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    names = list(num_cols)
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        names.extend(list(ohe.get_feature_names_out(cat_cols)))
    except Exception:
        names.extend(cat_cols)
    return names

def evaluate_and_plot(tag: str,
                      model_name: str,
                      model: Pipeline,
                      X_train: pd.DataFrame, y_train: pd.Series,
                      X_test: pd.DataFrame,  y_test: pd.Series,
                      class_names: List[str],
                      fig_dir: str) -> Dict[str, float]:
    # fit
    model.fit(X_train, y_train)
    # predict
    y_trn = model.predict(X_train)
    y_tst = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # metrics
    acc_tr = accuracy_score(y_train, y_trn)
    acc_te = accuracy_score(y_test, y_tst)
    prec_w = precision_score(y_test, y_tst, average="weighted", zero_division=0)
    rec_w  = recall_score(y_test, y_tst, average="weighted", zero_division=0)
    f1_w   = f1_score(y_test, y_tst, average="weighted")

    try:
        y_test_bin = pd.get_dummies(y_test)[class_names]  # ensure order
        roc_auc = roc_auc_score(y_test_bin, proba, multi_class="ovr", average="macro")
        pr_auc  = average_precision_score(y_test_bin, proba, average="macro")
    except Exception:
        roc_auc = np.nan; pr_auc = np.nan

    # confusion matrix
    cm = confusion_matrix(y_test, y_tst, labels=class_names)
    fig = plt.figure(figsize=(6,5)); ax = fig.gca()
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right"); ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, f"{tag}__cm__{model_name}.png")); plt.close()

    # per-class curves
    try:
        y_bin = label_binarize(y_test, classes=class_names)
        for i, cls in enumerate(class_names):
            plt.figure(figsize=(6,5))
            RocCurveDisplay.from_predictions(y_bin[:, i], proba[:, i])
            plt.title(f"ROC — {model_name} — {cls}")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.tight_layout(); plt.savefig(os.path.join(fig_dir, f"{tag}__roc__{model_name}__{cls}.png")); plt.close()

            plt.figure(figsize=(6,5))
            PrecisionRecallDisplay.from_predictions(y_bin[:, i], proba[:, i])
            plt.title(f"Precision–Recall — {model_name} — {cls}")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.tight_layout(); plt.savefig(os.path.join(fig_dir, f"{tag}__pr__{model_name}__{cls}.png")); plt.close()
    except Exception:
        pass

    return {
        "run": tag,
        "model": model_name,
        "acc_train": acc_tr,
        "acc_test": acc_te,
        "precision_w": prec_w,
        "recall_w": rec_w,
        "f1_w": f1_w,
        "roc_auc_macro": roc_auc,
        "pr_auc_macro": pr_auc,
    }

# ======================================================================
# Experiment runner (3A or 3B)
# ======================================================================
def run_experiment(tag: str,
                   X: pd.DataFrame, y: pd.Series, groups, *,
                   num_cols: List[str], cat_cols: List[str],
                   run_paths: Dict[str, str]) -> Tuple[pd.DataFrame, Pipeline, str]:
    """
    Returns (metrics_table, best_estimator, best_name)
    Saves figures for the best estimator.
    """
    # Group-aware holdout
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    (tr_idx, te_idx), = splitter.split(X, y, groups)
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    grp_tr, grp_te = np.array(groups)[tr_idx], np.array(groups)[te_idx]

    # (Re)detect columns post-split (hygiene)
    num_cols_final = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and c in num_cols]
    cat_cols_final = [c for c in X_train.columns if c in cat_cols]

    # class names (clean)
    class_names = sorted(pd.Series(y).map(clean_label).unique().tolist())

    results = []

    # 1) Logistic Regression (baseline)
    pre_lr = build_preprocessor(num_cols_final, cat_cols_final, scale_numeric=True)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    model_lr = Pipeline([("pre", pre_lr), ("clf", lr)])
    res_lr = evaluate_and_plot(tag, "LogReg", model_lr, X_train, y_train, X_test, y_test, class_names, run_paths["figures"])
    results.append(res_lr)

    # 2) RandomForest (baseline)
    pre_rf = build_preprocessor(num_cols_final, cat_cols_final, scale_numeric=False)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
    )
    model_rf_base = Pipeline([("pre", pre_rf), ("clf", rf)])
    res_rf_base = evaluate_and_plot(tag, "RF_base", model_rf_base, X_train, y_train, X_test, y_test, class_names, run_paths["figures"])
    results.append(res_rf_base)

    # 3) RandomForest (group-aware GridSearch)
    param_rf = {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", 0.5],
    }
    # CV: prefer StratifiedGroupKFold, else GroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    except Exception:
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=5)
    gs_rf = GridSearchCV(model_rf_base, param_grid=param_rf, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train, groups=grp_tr)
    model_rf_best = gs_rf.best_estimator_
    res_rf_best = evaluate_and_plot(tag, "RF_best", model_rf_best, X_train, y_train, X_test, y_test, class_names, run_paths["figures"])
    results.append(res_rf_best)

    # 4) LightGBM (optional, also group-aware)
    have_lgb = True
    model_lgb_best = None
    try:
        import lightgbm as lgb
        clf_lgb = lgb.LGBMClassifier(
            objective="multiclass", n_estimators=400, learning_rate=0.05,
            num_leaves=31, subsample=0.9, colsample_bytree=0.9,
            random_state=RANDOM_STATE
        )
        pre_lgb = build_preprocessor(num_cols_final, cat_cols_final, scale_numeric=False)
        model_lgb_base = Pipeline([("pre", pre_lgb), ("clf", clf_lgb)])
        res_lgb_base = evaluate_and_plot(tag, "LGBM_base", model_lgb_base, X_train, y_train, X_test, y_test, class_names, run_paths["figures"])
        results.append(res_lgb_base)

        param_lgb = {
            "clf__n_estimators": [300, 600],
            "clf__num_leaves": [31, 63],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [-1, 10],
        }
        gs_lgb = GridSearchCV(model_lgb_base, param_grid=param_lgb, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
        gs_lgb.fit(X_train, y_train, groups=grp_tr)
        model_lgb_best = gs_lgb.best_estimator_
        res_lgb_best = evaluate_and_plot(tag, "LGBM_best", model_lgb_best, X_train, y_train, X_test, y_test, class_names, run_paths["figures"])
        results.append(res_lgb_best)
    except Exception as e:
        have_lgb = False
        print(f"[INFO] LightGBM not available/failed: {e}")

    # results table
    res_df = pd.DataFrame(results).sort_values("f1_w", ascending=False)
    res_df.to_csv(os.path.join(run_paths["csv"], f"metrics_summary__{tag}.csv"), index=False)

    # choose best
    best_row = res_df.iloc[0]
    best_name = best_row["model"]
    if best_name == "RF_best": best_model = model_rf_best
    elif best_name == "RF_base": best_model = model_rf_base
    elif best_name == "LGBM_best" and have_lgb: best_model = model_lgb_best
    else: best_model = model_lr

    # save best model
    joblib.dump(best_model, os.path.join(run_paths["models"], f"{tag}__{best_name}.joblib"))
    # also save common baselines (handy for later comparison)
    joblib.dump(model_lr, os.path.join(run_paths["models"], "logreg__baseline.joblib"))
    joblib.dump(model_rf_best, os.path.join(run_paths["models"], "rf__best.joblib"))
    if have_lgb and model_lgb_best is not None:
        joblib.dump(model_lgb_best, os.path.join(run_paths["models"], "lgbm__best.joblib"))

    return res_df, best_model, best_name

# ======================================================================
# MAIN
# ======================================================================
def main():
    run_paths = ts_run_dir(OUTPUT_BASE)
    print(f"[OUT] {run_paths['root']}")
    env_versions(os.path.join(run_paths["root"], "env_versions.json"))

    # ---------------------------
    # Load & prepare data
    # ---------------------------
    d = load_datasets(BASE_DATA_PATH)
    df = merge_all(d)
    df = try_merge_task2_features(df, TASK2_FEATURES_PARQUET, on="pl_name") if TASK2_FEATURES_PARQUET else df

    # rule label (if not present)
    if "habitability_label" not in df.columns:
        df["habitability_label"] = df.apply(label_habitability, axis=1)
    df["habitability_label"] = df["habitability_label"].map(clean_label)

    # Candidate features (present in merged frame)
    candidate_numeric = [
        "pl_rade","pl_bmasse","pl_dens","pl_orbper","pl_orbsmax","pl_orbeccen","pl_eqt","pl_insol",
        "st_teff","st_rad","st_mass","sy_dist","sy_vmag","sy_kmag","sy_gaiamag",
        # atmos/alt names
        "mass","radius","orbital_period","semi_major_axis","star_teff","star_radius","star_distance","temp_calculated",
        "tsm","esm",
    ]
    candidate_categorical = [c for c in ["st_spectype","discoverymethod","type"] if c in df.columns]

    num_cols_all = [c for c in candidate_numeric if c in df.columns]
    cat_cols_all = candidate_categorical

    # Prepare base matrices
    groups_all = df["pl_name"].fillna(df.get("hostname"))
    y_all = df["habitability_label"]

    # ---------------------------
    # 3A — Rule-replication
    # ---------------------------
    X_3A = df[num_cols_all + cat_cols_all].copy()

    print("\n[RUN] 3A — Rule-replication (sanity check)")
    res_3A, best_3A, best_name_3A = run_experiment(
        tag="3A", X=X_3A, y=y_all, groups=groups_all,
        num_cols=num_cols_all, cat_cols=cat_cols_all, run_paths=run_paths
    )

    # ---------------------------
    # 3B — Credible predictor (robust)
    # Exclude rule-driving features so the model can't simply mimic the rule.
    # ---------------------------
    RULE_FEATURES = [  # direct rule drivers
        "pl_rade","pl_bmasse","pl_eqt","pl_insol","pl_orbeccen",
        "radius","mass","temp_calculated"
    ]
    num_cols_3B = [c for c in num_cols_all if c not in RULE_FEATURES]
    X_3B = df[num_cols_3B + cat_cols_all].copy()

    print("\n[RUN] 3B — Credible predictor (group-aware CV, excludes rule features)")
    res_3B, best_3B, best_name_3B = run_experiment(
        tag="3B", X=X_3B, y=y_all, groups=groups_all,
        num_cols=num_cols_3B, cat_cols=cat_cols_all, run_paths=run_paths
    )

    # ---------------------------
    # Save combined comparison table (best row from each run)
    # ---------------------------
    best_row_3A = res_3A.iloc[0].copy()
    best_row_3B = res_3B.iloc[0].copy()
    comp = pd.DataFrame([best_row_3A, best_row_3B])
    comp.to_csv(os.path.join(run_paths["csv"], "metrics_comparison_3A_3B.csv"), index=False)
    print("\n[RESULT] Best-model comparison (saved → csv/metrics_comparison_3A_3B.csv):")
    print(comp[["run","model","acc_test","precision_w","recall_w","f1_w","roc_auc_macro","pr_auc_macro"]])

    # ---------------------------
    # Short discussion note (why 3B is preferred)
    # ---------------------------
    note = """Why 3B is preferred
---------------------
• 3B uses StratifiedGroupKFold and a group-aware holdout split so the same planet/system never
  appears in both train and test → reduces optimistic leakage.
• 3B excludes direct rule-driving features (radius, mass, equilibrium temp/insolation, eccentricity).
  This forces the model to learn *signals beyond the hand-crafted rule*, improving credibility.
• Scores in 3B are typically lower than 3A but *honest* and more likely to generalise to new planets.
• Confusion matrices and PR/ROC curves for 3B give a more meaningful picture of class trade-offs,
  especially for minority classes.
"""
    with open(os.path.join(run_paths["root"], "notes__why_3B_preferred.txt"), "w") as f:
        f.write(note)

    print("\n[DONE] Outputs saved under:", run_paths["root"])
    print("- Figures →", run_paths["figures"])
    print("- CSVs    →", run_paths["csv"])
    print("- Models  →", run_paths["models"])

if __name__ == "__main__":
    main()

