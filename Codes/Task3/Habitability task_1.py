r"""
Task 3 — Habitability Classification (using 20_08_25 datasets)
----------------------------------------------------------------
Paths:
- Data folder:       D:/UH One drive/OneDrive - University of Hertfordshire/Final Project/Data/20_08_25
- Output base dir:   D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task3

What this script does (overview):
1) Loads & merges the three datasets (planetary systems, stellar hosts, atmospheres)
2) Runs shared, human‑readable EDA (missingness, distributions, correlations)
3) Creates a rule‑based habitability label (Habitable (Potentially) / Partially Habitable / Non‑Habitable)
4) Trains baseline models (LogisticRegression, RandomForest) + tuned models (RF Grid, LightGBM Grid if installed)
5) Uses GROUP‑AWARE train/test split to prevent leakage (ensures same planet/system isn't in both sets)
6) Evaluates with multiple metrics (Accuracy, Precision_w, Recall_w, F1_w, ROC‑AUC(OVR macro), PR‑AUC macro) + plots
7) Exports predictions + probabilities and a Streamlit/Task‑5 ready dashboard pack
8) Saves EVERYTHING to a timestamped run folder for reproducibility, incl. environment versions


Author: Praveen Kumar Savariraj
"""
from __future__ import annotations

import json
import os
import sys
import math
import time
import joblib
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils import Bunch

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({"figure.dpi": 125})

# ------------------------------------------------------------------------------------
# Config — EDIT THESE TWO PATHS if your folders differ
# ------------------------------------------------------------------------------------
BASE_DATA_PATH = r"D:/UH One drive/OneDrive - University of Hertfordshire/Final Project/Data/20_08_25"
OUTPUT_BASE    = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task3"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20

# Optional: if you don't want Task‑2 features in Task‑3, set this to None
TASK2_FEATURES_PARQUET = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task2/latest/features_task2.parquet"

# ------------------------------------------------------------------------------------
# Try to import the shared utils we created earlier. If unavailable, fall back to
# local lightweight helpers so the script stays runnable.
# ------------------------------------------------------------------------------------
try:
    from shared_utils_task2_task3_dashboard import (
        HUMAN_LABEL, pretty,
        run_shared_eda, run_leakage_checks,
        group_stratified_split, drop_constant_and_duplicate_features,
        try_merge_task2_features, export_dashboard_pack, proba_from_model
    )
    HAVE_SHARED = True
except Exception:
    HAVE_SHARED = False

    HUMAN_LABEL: Dict[str, str] = {
        "pl_name": "Planet Name", "hostname": "Host Star Name",
        "pl_rade": "Planet Radius (Earth radii)",
        "pl_bmasse": "Planet Mass (Earth masses)",
        "pl_dens": "Planet Density (g/cm³)",
        "pl_orbper": "Orbital Period (days)",
        "pl_orbsmax": "Semi‑major Axis (AU)",
        "pl_orbeccen": "Orbital Eccentricity",
        "pl_eqt": "Equilibrium Temperature (K)",
        "pl_insol": "Insolation Flux (Earth=1)",
        "st_teff": "Stellar Effective Temperature (K)",
        "st_rad": "Stellar Radius (Solar radii)",
        "st_mass": "Stellar Mass (Solar masses)",
        "st_spectype": "Stellar Spectral Type",
        "sy_dist": "Distance from Earth (pc)",
        "sy_vmag": "V Magnitude", "sy_kmag": "K Magnitude", "sy_gaiamag": "Gaia G Magnitude",
        # Atmosphere alt names
        "mass": "Planet Mass (Earth masses)", "radius": "Planet Radius (Earth radii)",
        "orbital_period": "Orbital Period (days)", "semi_major_axis": "Semi‑major Axis (AU)",
        "star_teff": "Stellar Effective Temperature (K)", "star_radius": "Stellar Radius (Solar radii)",
        "star_distance": "Distance from Earth (pc)", "temp_calculated": "Estimated Equilibrium Temperature (K)",
        "tsm": "Transmission Spectroscopy Metric (TSM)", "esm": "Emission Spectroscopy Metric (ESM)",
    }

    def pretty(col: str) -> str:
        return HUMAN_LABEL.get(col, col.replace("_", " ").title())

    def _ensure_dir(path: str) -> str:
        os.makedirs(path, exist_ok=True); return path

    def plot_missingness(df: pd.DataFrame, out_path: str, top_k: int = 30):
        miss = df.isna().mean().sort_values(ascending=False).head(top_k) * 100
        fig = plt.figure(figsize=(10, 6)); ax = fig.gca()
        miss.plot(kind="bar", ax=ax)
        ax.set_title("Top Missing Data Columns (%)"); ax.set_ylabel("Missing (%)"); ax.set_xlabel("Feature")
        ax.set_xticklabels([pretty(c) for c in miss.index], rotation=60, ha="right")
        plt.tight_layout(); plt.savefig(out_path); plt.close()

    def plot_histograms(df: pd.DataFrame, cols: Sequence[str], out_path: str, bins: int = 30):
        nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]; n = len(nums)
        if n == 0: return
        ncols = 3; nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*3)); axes = np.array(axes).reshape(-1)
        for i, c in enumerate(nums):
            ax = axes[i]; ax.hist(df[c].dropna(), bins=bins)
            ax.set_title(pretty(c)); ax.set_xlabel(pretty(c)); ax.set_ylabel("Count")
        for j in range(i+1, len(axes)): axes[j].axis("off")
        plt.suptitle("Distribution of Key Numeric Features", y=1.02)
        plt.tight_layout(); plt.savefig(out_path); plt.close()

    def plot_correlation(df: pd.DataFrame, out_path: str, method: str = "spearman", max_cols: int = 30):
        num_df = df.select_dtypes(include=[np.number])
        nunique = num_df.nunique(); keep = nunique[nunique > 2].index.tolist()
        num_df = num_df[keep].iloc[:, :max_cols]
        if num_df.shape[1] < 2: return
        corr = num_df.corr(method=method)
        fig = plt.figure(figsize=(10, 8)); ax = fig.gca()
        cax = ax.imshow(corr.values, interpolation="nearest", aspect="auto"); fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(corr.shape[1])); ax.set_yticks(range(corr.shape[1]))
        labels = [pretty(c) for c in corr.columns]
        ax.set_xticklabels(labels, rotation=60, ha="right"); ax.set_yticklabels(labels)
        ax.set_title(f"{method.title()} Correlation Heatmap")
        plt.tight_layout(); plt.savefig(out_path); plt.close()

    def run_shared_eda(df: pd.DataFrame, out_dir: str, title_prefix: str = ""):
        _ensure_dir(out_dir)
        possible_cols = [
            "pl_rade","pl_bmasse","pl_dens","pl_orbper","pl_orbsmax","pl_orbeccen","pl_eqt","pl_insol",
            "st_teff","st_rad","st_mass","st_spectype","sy_dist","sy_vmag","sy_kmag","sy_gaiamag",
            "mass","radius","orbital_period","semi_major_axis","star_teff","star_radius","star_distance","temp_calculated",
        ]
        cols = [c for c in possible_cols if c in df.columns]
        plot_missingness(df, os.path.join(out_dir, f"{title_prefix}__missingness_top30.png"))
        plot_histograms(df, cols, os.path.join(out_dir, f"{title_prefix}__histograms.png"))
        plot_correlation(df, os.path.join(out_dir, f"{title_prefix}__correlation.png"))
        df[cols].describe(include="all").T.rename_axis("feature").reset_index().to_csv(
            os.path.join(out_dir, f"{title_prefix}__summary_stats.csv"), index=False)

    def run_leakage_checks(df: pd.DataFrame, y_col: str, id_cols: Sequence[str], out_dir: str):
        _ensure_dir(out_dir); issues = []
        feature_like = [c for c in df.columns if c != y_col]
        label_aliases = {"label","target","habitability","habitability_rule","rule_label","is_habitable"}
        label_cols = [c for c in feature_like if c.lower() in label_aliases]
        if label_cols: issues.append({"type":"target_leakage","found_columns":label_cols})
        dup_all = df.duplicated(subset=id_cols, keep=False).sum()
        issues.append({"type":"duplicates_by_id","count":int(dup_all)})
        const_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
        issues.append({"type":"constant_columns","count":len(const_cols),"columns":const_cols[:20]})
        pd.DataFrame(issues).to_csv(os.path.join(out_dir, "leakage_diagnostics.csv"), index=False)

    def drop_constant_and_duplicate_features(X: pd.DataFrame, out_dir: Optional[str] = None) -> pd.DataFrame:
        const_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        if const_cols: X = X.drop(columns=const_cols)
        dup_to_drop = []; seen = {}
        for c in X.columns:
            key = (tuple(pd.Series(X[c]).fillna(-999999).values[:1000]), str(X[c].dtype))
            if key in seen: dup_to_drop.append(c)
            else: seen[key] = c
        if dup_to_drop: X = X.drop(columns=dup_to_drop)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "feature_hygiene.json"), "w") as f:
                json.dump({"dropped_constant":const_cols,"dropped_duplicate":dup_to_drop,"remaining_features":X.columns.tolist()}, f, indent=2)
        return X

    def group_stratified_split(X: pd.DataFrame, y: pd.Series, groups: Sequence, test_size: float = 0.2, random_state: int = 42):
        from sklearn.model_selection import GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        (train_idx, test_idx), = splitter.split(X, y, groups)
        return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx], np.array(groups)[train_idx], np.array(groups)[test_idx]

    def try_merge_task2_features(df: pd.DataFrame, features_path: str, on: str = "pl_name") -> pd.DataFrame:
        try:
            if features_path and os.path.exists(features_path):
                f2 = pd.read_parquet(features_path)
                if on in df.columns and on in f2.columns:
                    return df.merge(f2, on=on, how="left", suffixes=("", "_t2"))
        except Exception: pass
        return df

    def export_dashboard_pack(base_df: pd.DataFrame, id_col: str, label_col: str, proba_df: pd.DataFrame,
                              shap_values_df: Optional[pd.DataFrame] = None, feature_importances: Optional[pd.DataFrame] = None,
                              out_dir: str = ".", top_k: int = 100) -> None:
        os.makedirs(out_dir, exist_ok=True)
        keep_cols = [c for c in [id_col, "hostname", "st_spectype", "st_teff", "st_rad", "st_mass",
                                 "pl_rade", "pl_bmasse", "pl_eqt", "pl_insol", "pl_orbsmax", "pl_orbper", "pl_orbeccen",
                                 "sy_dist", "discoverymethod", "disc_year"] if c in base_df.columns]
        meta = base_df[keep_cols].drop_duplicates(subset=[id_col])
        proba_df = proba_df.drop_duplicates(subset=[id_col])
        dash = meta.merge(proba_df, on=id_col, how="left")
        hab_cols = [c for c in proba_df.columns if "habitable" in c.lower() or "positive" in c.lower()]
        if hab_cols: dash["rank_score"] = dash[hab_cols[0]]
        else:
            proba_cols = [c for c in proba_df.columns if c != id_col]
            dash["rank_score"] = dash[proba_cols].max(axis=1)
        dash_sorted = dash.sort_values("rank_score", ascending=False).head(top_k)
        dash_sorted.to_parquet(os.path.join(out_dir, "dashboard_data.parquet"), index=False)
        dash_sorted.to_csv(os.path.join(out_dir, "dashboard_data.csv"), index=False)
        meta_info = {"id_col": id_col, "label_col": label_col, "columns_human_labels": {c: pretty(c) for c in dash_sorted.columns}, "n_rows": int(dash_sorted.shape[0])}
        if feature_importances is not None:
            fi = feature_importances.copy()
            if "feature" in fi.columns:
                fi["feature_pretty"] = fi["feature"].map(pretty).fillna(fi["feature"])
            fi.sort_values("importance", ascending=False).to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)
            meta_info["has_feature_importances"] = True
        if shap_values_df is not None:
            shap_values_df.to_parquet(os.path.join(out_dir, "shap_values.parquet"), index=False)
            meta_info["has_shap"] = True
        with open(os.path.join(out_dir, "dashboard_meta.json"), "w") as f:
            json.dump(meta_info, f, indent=2)

    def proba_from_model(model, X: pd.DataFrame, id_series: pd.Series, class_order: Optional[List[str]] = None, id_col: str = "pl_name") -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        probs = model.predict_proba(X)
        if class_order is None and hasattr(model, "classes_"): class_order = list(model.classes_)
        cols = [f"proba_{str(c).lower()}" for c in class_order]
        out = pd.DataFrame(probs, columns=cols)
        out[id_col] = id_series.values
        return out[[id_col] + cols]

# ------------------------------------------------------------------------------------
# Helpers (paths, IO, figures)
# ------------------------------------------------------------------------------------

def ts_run_dir(base: str) -> Dict[str, str]:
    ts = time.strftime("run_%Y%m%d_%H%M%S")
    root = os.path.join(base, ts)
    paths = {
        "root": root,
        "figures": os.path.join(root, "figures"),
        "csv": os.path.join(root, "csv"),
        "models": os.path.join(root, "models"),
    }
    for p in paths.values(): os.makedirs(p, exist_ok=True)
    return paths


def env_versions(out_path: str) -> None:
    import sklearn, numpy, pandas
    info = {
        "python": sys.version,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import lightgbm as lgb
        info["lightgbm"] = lgb.__version__
    except Exception:
        info["lightgbm"] = None
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)


# ------------------------------------------------------------------------------------
# Data loading & merging
# ------------------------------------------------------------------------------------

def load_datasets(base_path: str) -> Dict[str, pd.DataFrame]:
    # filenames expected in the provided folder
    f_planetary = os.path.join(base_path, "Planetary Systems Composite Data.csv")
    f_stellar   = os.path.join(base_path, "Stellar Hosts.csv")
    f_atmo      = os.path.join(base_path, "cleaned_exoplanet_atmospheres_with_molecules_cleaned.csv")

    d = {}
    d["planetary"] = pd.read_csv(f_planetary)
    d["stellar"]   = pd.read_csv(f_stellar)
    d["atmo"]      = pd.read_csv(f_atmo)
    return d


def merge_all(d: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Harmonise key columns
    p = d["planetary"].copy()
    s = d["stellar"].copy()
    a = d["atmo"].copy()

    # Normalise likely key names
    for df in (p, s, a):
        # Create standard columns if present with alternative names
        if "pl_name" not in df.columns and "planet_name" in df.columns:
            df.rename(columns={"planet_name":"pl_name"}, inplace=True)
        if "hostname" not in df.columns:
            for cand in ["host_name", "star_name", "stellar_host", "hostname"]:
                if cand in df.columns:
                    df.rename(columns={cand:"hostname"}, inplace=True)
                    break

    # Merge planetary + stellar on hostname (left)
    m = p.merge(s, on="hostname", how="left", suffixes=("", "_host")) if "hostname" in p.columns and "hostname" in s.columns else p

    # Atmospheres often include planet name — merge on pl_name if available
    if "pl_name" in m.columns and "pl_name" in a.columns:
        m = m.merge(a, on="pl_name", how="left", suffixes=("", "_atmo"))
    else:
        # Fall back: if atmo uses alternative key
        if "planet_name" in a.columns and "pl_name" in m.columns:
            a2 = a.rename(columns={"planet_name":"pl_name"})
            m = m.merge(a2, on="pl_name", how="left", suffixes=("", "_atmo"))

    return m


# ------------------------------------------------------------------------------------
# Rule‑based habitability labelling (transparent baseline)
# ------------------------------------------------------------------------------------

def label_habitability(row: pd.Series) -> str:
    """Simple, interpretable rules — conservative (tweak in viva if needed)."""
    r = row.get("pl_rade", row.get("radius", np.nan))          # Earth radii
    m = row.get("pl_bmasse", row.get("mass", np.nan))          # Earth masses
    teq = row.get("pl_eqt", row.get("temp_calculated", np.nan))
    ins = row.get("pl_insol", np.nan)
    ecc = row.get("pl_orbeccen", np.nan)

    # Core conditions
    cond_radius = (0.5 <= r <= 1.8) if pd.notna(r) else False
    cond_mass   = (0.5 <= m <= 5.0) if pd.notna(m) else False
    # Use either insolation or temperature window if available
    cond_flux   = (0.35 <= ins <= 1.5) if pd.notna(ins) else None
    cond_temp   = (180 <= teq <= 310) if pd.notna(teq) else None
    climate_ok  = cond_flux if cond_flux is not None else cond_temp if cond_temp is not None else False
    cond_ecc    = (ecc <= 0.5) if pd.notna(ecc) else True  # if unknown, don't punish harshly

    if cond_radius and cond_mass and climate_ok and cond_ecc:
        return "Habitable (Potentially)"
    if (cond_radius or cond_mass) and (climate_ok or cond_ecc):
        return "Partially Habitable"
    return "Non‑Habitable"


# ------------------------------------------------------------------------------------
# Preprocessing & evaluation utilities
# ------------------------------------------------------------------------------------

def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale_numeric: bool = False) -> ColumnTransformer:
    num_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))] + ([("scaler", StandardScaler())] if scale_numeric else []))
    cat_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer(transformers=[("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")
    return pre


def get_feature_names(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    names = []
    # numeric
    names += num_cols
    # categorical (after OHE)
    try:
        ohe = pre.named_transformers_["cat"].named_steps["ohe"]
        cat_expanded = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        cat_expanded = cat_cols
    names += cat_expanded
    return names


def evaluate_and_plot(model_name: str, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, class_names: List[str], out_fig_dir: str) -> Dict[str, float]:
    # Fit
    model.fit(X_train, y_train)

    # Predict
    y_trn = model.predict(X_train)
    y_tst = model.predict(X_test)
    proba_tst = model.predict_proba(X_test)

    # Metrics
    acc_tr = accuracy_score(y_train, y_trn)
    acc_te = accuracy_score(y_test, y_tst)
    prec_w = precision_score(y_test, y_tst, average="weighted", zero_division=0)
    rec_w  = recall_score(y_test, y_tst, average="weighted", zero_division=0)
    f1_w   = f1_score(y_test, y_tst, average="weighted")

    # ROC AUC (multiclass OVR macro) — requires probas
    try:
        roc_auc = roc_auc_score(pd.get_dummies(y_test, columns=class_names), proba_tst, multi_class="ovr", average="macro")
    except Exception:
        roc_auc = np.nan

    # PR AUC macro (Average Precision macro)
    try:
        y_test_bin = pd.get_dummies(y_test)
        pr_auc = average_precision_score(y_test_bin, proba_tst, average="macro")
    except Exception:
        pr_auc = np.nan

    # Confusion matrix
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
    plt.tight_layout(); plt.savefig(os.path.join(out_fig_dir, f"cm__{model_name}.png")); plt.close()

    # Simple ROC/PR plots (One‑vs‑Rest) for the chosen model
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
        y_bin = label_binarize(y_test, classes=class_names)
        for i, cls in enumerate(class_names):
            fig = plt.figure(figsize=(6,5))
            RocCurveDisplay.from_predictions(y_bin[:, i], proba_tst[:, i])
            plt.title(f"ROC Curve — {model_name} — Class: {cls}")
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.tight_layout(); plt.savefig(os.path.join(out_fig_dir, f"roc__{model_name}__{cls}.png")); plt.close()

            fig = plt.figure(figsize=(6,5))
            PrecisionRecallDisplay.from_predictions(y_bin[:, i], proba_tst[:, i])
            plt.title(f"Precision‑Recall — {model_name} — Class: {cls}")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.tight_layout(); plt.savefig(os.path.join(out_fig_dir, f"pr__{model_name}__{cls}.png")); plt.close()
    except Exception:
        pass

    return {
        "model": model_name,
        "acc_train": acc_tr,
        "acc_test": acc_te,
        "precision_w": prec_w,
        "recall_w": rec_w,
        "f1_w": f1_w,
        "roc_auc_macro": roc_auc,
        "pr_auc_macro": pr_auc,
    }


def learning_curve_plot(model: Pipeline, X: pd.DataFrame, y: pd.Series, out_path: str, cv_splits: int = 5):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv_splits, scoring="f1_weighted", n_jobs=None, random_state=RANDOM_STATE)
    tr_mean = train_scores.mean(axis=1); te_mean = test_scores.mean(axis=1)
    fig = plt.figure(figsize=(6,5))
    ax = fig.gca()
    ax.plot(train_sizes, tr_mean, marker="o", label="Train F1 (mean)")
    ax.plot(train_sizes, te_mean, marker="s", label="CV F1 (mean)")
    ax.set_title("Learning Curve (F1‑weighted)")
    ax.set_xlabel("Training Set Size"); ax.set_ylabel("F1‑weighted")
    ax.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()


# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------

def main():
    # Output run folder
    run_paths = ts_run_dir(OUTPUT_BASE)
    print(f"[OUT] {run_paths['root']}")

    # Save environment versions
    env_versions(os.path.join(run_paths["root"], "env_versions.json"))

    # Load & merge
    d = load_datasets(BASE_DATA_PATH)
    df = merge_all(d)

    # Optional: incorporate engineered features from Task‑2 (if file exists)
    df = try_merge_task2_features(df, TASK2_FEATURES_PARQUET, on="pl_name") if TASK2_FEATURES_PARQUET else df

    # Create rule‑based label if not present
    if "habitability_label" not in df.columns:
        df["habitability_label"] = df.apply(label_habitability, axis=1)

    # Shared EDA (human‑readable labels)
    run_shared_eda(df, out_dir=run_paths["figures"], title_prefix="Task3")

    # Choose features (only those that exist)
    candidate_numeric = [
        "pl_rade","pl_bmasse","pl_dens","pl_orbper","pl_orbsmax","pl_orbeccen","pl_eqt","pl_insol",
        "st_teff","st_rad","st_mass","sy_dist","sy_vmag","sy_kmag","sy_gaiamag",
        # Atmosphere alt names
        "mass","radius","orbital_period","semi_major_axis","star_teff","star_radius","star_distance","temp_calculated","tsm","esm",
    ]
    candidate_categorical = [c for c in ["st_spectype","discoverymethod","type"] if c in df.columns]

    num_cols = [c for c in candidate_numeric if c in df.columns]
    cat_cols = candidate_categorical

    # Avoid leakage: remove any target/rule columns from features
    drop_leaky = [c for c in df.columns if c.lower() in {"habitability_label","label","target","habitability","rule_label","is_habitable"}]

    feature_cols = [c for c in (num_cols + cat_cols) if c not in drop_leaky]

    # Leakage diagnostics report
    run_leakage_checks(df[feature_cols + ["habitability_label","pl_name","hostname"]].copy(), y_col="habitability_label", id_cols=["pl_name","hostname"], out_dir=run_paths["csv"])

    # Prepare X, y, groups (group by planet name where available; fallback to host star)
    X = df[feature_cols].copy()
    y = df["habitability_label"].copy()
    groups = df["pl_name"].fillna(df.get("hostname"))

    # Feature hygiene
    X = drop_constant_and_duplicate_features(X, out_dir=run_paths["csv"])

    # Group‑aware train/test split
    X_train, X_test, y_train, y_test, grp_tr, grp_te = group_stratified_split(X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Identify column types on the *post‑hygiene* set
    num_cols_final = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and c in num_cols]
    cat_cols_final = [c for c in X_train.columns if c in cat_cols]

    # ---------------------------------
    # Models
    # ---------------------------------

    results = []

    # 1) Logistic Regression (baseline)
    pre_lr = build_preprocessor(num_cols_final, cat_cols_final, scale_numeric=True)
    lr = LogisticRegression(max_iter=2000, multi_class="auto", class_weight="balanced", random_state=RANDOM_STATE)
    model_lr = Pipeline(steps=[("pre", pre_lr), ("clf", lr)])
    res_lr = evaluate_and_plot("LogReg", model_lr, X_train, y_train, X_test, y_test, class_names=sorted(y.unique()), out_fig_dir=run_paths["figures"]) 
    results.append(res_lr)

    # 2) RandomForest (baseline)
    pre_rf = build_preprocessor(num_cols_final, cat_cols_final, scale_numeric=False)
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample")
    model_rf_base = Pipeline(steps=[("pre", pre_rf), ("clf", rf)])
    res_rf_base = evaluate_and_plot("RF_base", model_rf_base, X_train, y_train, X_test, y_test, class_names=sorted(y.unique()), out_fig_dir=run_paths["figures"]) 
    results.append(res_rf_base)

    # 3) RandomForest (GridSearchCV)
    param_rf = {
        "clf__n_estimators": [300, 600],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", 0.5]
    }
    gs_rf = GridSearchCV(model_rf_base, param_grid=param_rf, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=1)
    gs_rf.fit(X_train, y_train)
    model_rf_best = gs_rf.best_estimator_
    res_rf_best = evaluate_and_plot("RF_best", model_rf_best, X_train, y_train, X_test, y_test, class_names=sorted(y.unique()), out_fig_dir=run_paths["figures"]) 
    results.append(res_rf_best)

    # 4) LightGBM (optional)
    have_lgb = True
    try:
        import lightgbm as lgb
        clf_lgb = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
        )
        pre_lgb = build_preprocessor(num_cols_final, cat_cols_final, scale_numeric=False)
        model_lgb_base = Pipeline(steps=[("pre", pre_lgb), ("clf", clf_lgb)])
        res_lgb_base = evaluate_and_plot("LGBM_base", model_lgb_base, X_train, y_train, X_test, y_test, class_names=sorted(y.unique()), out_fig_dir=run_paths["figures"]) 
        results.append(res_lgb_base)

        # Small grid to demonstrate tuning impact
        param_lgb = {
            "clf__n_estimators": [300, 600],
            "clf__num_leaves": [31, 63],
            "clf__learning_rate": [0.03, 0.05, 0.1],
            "clf__max_depth": [-1, 10],
        }
        gs_lgb = GridSearchCV(model_lgb_base, param_grid=param_lgb, scoring="f1_weighted", cv=5, n_jobs=-1, verbose=1)
        gs_lgb.fit(X_train, y_train)
        model_lgb_best = gs_lgb.best_estimator_
        res_lgb_best = evaluate_and_plot("LGBM_best", model_lgb_best, X_train, y_train, X_test, y_test, class_names=sorted(y.unique()), out_fig_dir=run_paths["figures"]) 
        results.append(res_lgb_best)
    except Exception as e:
        have_lgb = False
        model_lgb_base = None; model_lgb_best = None
        print(f"[INFO] LightGBM not available or failed: {e}")

    # ---------------------------------
    # Results summary
    # ---------------------------------
    res_df = pd.DataFrame(results).sort_values("f1_w", ascending=False)
    res_df.to_csv(os.path.join(run_paths["csv"], "metrics_summary.csv"), index=False)
    print("[RESULTS] Summary (sorted by F1_w):\n", res_df)

    # Choose best model by F1_w
    best_row = res_df.iloc[0]
    best_name = best_row["model"]
    if best_name == "RF_best": best_model = model_rf_best
    elif best_name == "RF_base": best_model = model_rf_base
    elif best_name == "LGBM_best" and have_lgb: best_model = model_lgb_best
    elif best_name == "LGBM_base" and have_lgb: best_model = model_lgb_base
    else: best_model = model_lr

    # Fit best model on full TRAIN set (not including test) for fair evaluation; then refit on ALL for dashboard
    best_model.fit(X_train, y_train)

    # Save trained models
    joblib.dump(model_lr,      os.path.join(run_paths["models"], "logreg.joblib"))
    joblib.dump(model_rf_best, os.path.join(run_paths["models"], "random_forest_best.joblib"))
    if have_lgb and model_lgb_best is not None:
        joblib.dump(model_lgb_best, os.path.join(run_paths["models"], "lightgbm_best.joblib"))

    # Predictions/probabilities on TEST set (for honest evaluation)
    class_order = sorted(y.unique())
    proba_test_df = proba_from_model(best_model, X_test, id_series=df.loc[X_test.index, "pl_name"], class_order=class_order, id_col="pl_name")
    preds_test = best_model.predict(X_test)

    # Save combined predictions (test only)
    out_test = df.loc[X_test.index, ["pl_name","hostname","habitability_label"]].copy()
    out_test["pred_label"] = preds_test
    out_test = out_test.merge(proba_test_df, on="pl_name", how="left")
    out_test.to_csv(os.path.join(run_paths["csv"], "task3_test_predictions.csv"), index=False)

    # Refit on ALL data for dashboard ranking (as is common in production)
    best_model.fit(X, y)
    proba_all_df = proba_from_model(best_model, X, id_series=df["pl_name"], class_order=class_order, id_col="pl_name")
    pred_all = best_model.predict(X)

    # Save full predictions with rule label
    out_all = df[["pl_name","hostname","habitability_label"]].copy()
    out_all["pred_label"] = pred_all
    out_all = out_all.merge(proba_all_df, on="pl_name", how="left")
    out_all.to_csv(os.path.join(run_paths["csv"], "task3_habitability_with_predictions.csv"), index=False)

    # Feature importances (for trees)
    fi_df = None
    try:
        if best_name.startswith("RF"):
            # Extract feature names post‑transform
            pre = best_model.named_steps["pre"]
            feat_names = get_feature_names(pre, num_cols_final, cat_cols_final)
            importances = best_model.named_steps["clf"].feature_importances_
            fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            fi_df.to_csv(os.path.join(run_paths["csv"], "rf_feature_importances.csv"), index=False)
        elif best_name.startswith("LGBM") and have_lgb:
            pre = best_model.named_steps["pre"]
            feat_names = get_feature_names(pre, num_cols_final, cat_cols_final)
            importances = best_model.named_steps["clf"].feature_importances_
            fi_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values("importance", ascending=False)
            fi_df.to_csv(os.path.join(run_paths["csv"], "lgbm_feature_importances.csv"), index=False)
    except Exception:
        pass

    # Optional SHAP (if LightGBM and shap are available)
    try:
        import shap
        if have_lgb and best_name.startswith("LGBM"):
            explainer = shap.TreeExplainer(best_model.named_steps["clf"])  # trees only, on transformed space
            # Use a small background sample for speed
            pre = best_model.named_steps["pre"]
            Xs = pre.fit_transform(X, y)
            shap_vals = explainer.shap_values(Xs[:1000])
            # Save a simple summary plot for the top class
            plt.figure(figsize=(8,6))
            shap.summary_plot(shap_vals[0], Xs[:1000], show=False)
            plt.tight_layout(); plt.savefig(os.path.join(run_paths["figures"], "shap_summary.png")); plt.close()
    except Exception as e:
        pass

    # Export Streamlit/Task‑5 pack from full‑data proba (ranked top‑K by habitable prob)
    export_dashboard_pack(
        base_df=df,
        id_col="pl_name",
        label_col="habitability_label",
        proba_df=proba_all_df,
        shap_values_df=None,
        feature_importances=fi_df,
        out_dir=run_paths["csv"],
    )

    print("\n[DONE] Outputs saved under:", run_paths["root"])
    print("- Figures →", run_paths["figures"]) 
    print("- CSVs    →", run_paths["csv"]) 
    print("- Models  →", run_paths["models"]) 


if __name__ == "__main__":
    main()

