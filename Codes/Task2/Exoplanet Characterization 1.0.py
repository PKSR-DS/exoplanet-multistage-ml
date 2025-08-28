# -*- coding: utf-8 -*-
"""
Task 2 — Exoplanet Characterization (UH MSc Final Project)
Author: Praveen Kumar Savariraj | Student ID: 23089117
"""

# ========= 1) Imports & Settings =========
import os, json, warnings, re
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     GridSearchCV, learning_curve)
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             average_precision_score, f1_score)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.multiclass import OneVsRestClassifier

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN = True
except Exception:
    SEABORN = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# clearer matplotlib defaults
plt.rcParams.update({
    "figure.dpi": 130, "axes.titlesize": 14, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10
})

# ========= 2) Paths =========
DATA_DIR = "D:/UH One drive/OneDrive - University of Hertfordshire/Final Project/Data/20_08_25"
OUT_ROOT = "D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task2"
os.makedirs(OUT_ROOT, exist_ok=True)
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(OUT_ROOT, f"run_{RUN_STAMP}")
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name, dpi=130, tight=True):
    path = os.path.join(OUT_DIR, name)
    if tight: plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    print(f"[FIG] Saved: {path}")

# ========= 3) Human-readable labels =========
HUMAN_NAMES = {
    'pl_rade': 'Planet radius (Earth radii)',
    'pl_bmasse': 'Planet mass (Earth masses)',
    'pl_bmasse_filled': 'Mass (filled, Earth masses)',
    'pl_dens': 'Density (g/cc)',
    'pl_dens_filled': 'Density (filled, g/cc)',
    'pl_eqt': 'Equilibrium temperature (K)',
    'pl_eqt_filled': 'Equilibrium temperature (filled, K)',
    'dens_from_mass_radius': 'Density from M,R (g/cc)',
    'dens_from_mass_radius_filled': 'Density from filled M,R (g/cc)',
    'st_teff': 'Stellar effective temperature (K)',
    'st_mass': 'Stellar mass (Solar masses)',
    'st_rad': 'Stellar radius (Solar radii)',
    'st_met': 'Stellar metallicity [dex]',
    'st_logg': 'Stellar log g (cgs)',
    'sy_dist': 'System distance (pc)',
    'sy_snum': 'Stars in system',
    'sy_pnum': 'Planets in system',
    'sy_vmag': 'V magnitude',
    'sy_jmag': 'J magnitude',
    'sy_kmag': 'K magnitude'
}
def pretty(name: str) -> str:
    return HUMAN_NAMES.get(name, name.replace('_', ' ').title())

# ========= 4) Load & Standardise =========
def load_csv_smart(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    df.columns = [re.sub(r'\s+', '_', c.strip().lower()) for c in df.columns]
    return df

ps_path  = os.path.join(DATA_DIR, "Planetary Systems Composite Data.csv")
sh_path  = os.path.join(DATA_DIR, "Stellar Hosts.csv")
atm_path = os.path.join(DATA_DIR, "cleaned_exoplanet_atmospheres_with_molecules_cleaned.csv")

ps  = load_csv_smart(ps_path)
sh  = load_csv_smart(sh_path)
atm = load_csv_smart(atm_path)

print(f"[LOAD] planetary systems: {ps.shape}  | stellar hosts: {sh.shape}  | atmospheres: {atm.shape}")

for df_ in (ps, sh, atm):
    if 'kepid' in df_.columns:
        df_['kepid'] = pd.to_numeric(df_['kepid'], errors='coerce')

def coalesce_name(df_, cols, out_col):
    for c in cols:
        if c in df_.columns:
            df_[out_col] = df_[c].astype(str).str.strip()
            return
    df_[out_col] = np.nan

coalesce_name(ps,  ['pl_name','planet_name','name'], 'pl_name_std')
coalesce_name(ps,  ['hostname','host_name','star_name'], 'host_name_std')
coalesce_name(sh,  ['hostname','host_name','star_name'], 'host_name_std')
coalesce_name(atm, ['pl_name','planet_name','name'], 'pl_name_std')
coalesce_name(atm, ['hostname','host_name','star_name'], 'host_name_std')

def smart_merge(ps, sh, atm):
    if 'kepid' in ps.columns and 'kepid' in sh.columns and ps['kepid'].notna().any() and sh['kepid'].notna().any():
        merged = pd.merge(ps, sh, on='kepid', how='left', suffixes=('', '_sh'))
    else:
        merged = pd.merge(ps, sh, on='host_name_std', how='left', suffixes=('', '_sh'))
    if 'kepid' in merged.columns and 'kepid' in atm.columns and merged['kepid'].notna().any() and atm['kepid'].notna().any():
        merged = pd.merge(merged, atm, on='kepid', how='left', suffixes=('', '_atm'))
    else:
        merged = pd.merge(merged, atm, on=['pl_name_std','host_name_std'], how='left', suffixes=('', '_atm'))
    return merged

df = smart_merge(ps, sh, atm)
print(f"[MERGE] merged shape: {df.shape}")
df.to_csv(os.path.join(OUT_DIR, "merged_raw.csv"), index=False)

def to_numeric_cols(dfin):
    dfout = dfin.copy()
    for c in dfout.columns:
        if dfout[c].dtype == object:
            dfout[c] = pd.to_numeric(dfout[c], errors='ignore')
    return dfout

df = to_numeric_cols(df)

# ========= 6) Molecules =========
KNOWN_MOLS = ["H2O","H","He","CO2","CO","CH4","NH3","Na","K","TiO","VO","HCN","C2H2","O3","O2"]
MOL_RE = re.compile(r'(' + '|'.join(KNOWN_MOLS) + r')', flags=re.IGNORECASE)
mol_cols = [c for c in df.columns if 'molecule' in c or 'molecules' in c]

def clean_mol_cell(x):
    if pd.isna(x): return []
    return sorted({m.upper() for m in MOL_RE.findall(str(x))})

df['molecules_list'] = df[mol_cols[0]].apply(clean_mol_cell) if mol_cols else [[] for _ in range(len(df))]

# ========= 7) Physics helper feature =========
if 'pl_bmasse' in df.columns and 'pl_rade' in df.columns:
    with np.errstate(divide='ignore', invalid='ignore'):
        df['dens_from_mass_radius'] = 5.51 * (df['pl_bmasse'] / (df['pl_rade']**3))
else:
    df['dens_from_mass_radius'] = np.nan

# ========= 8) EDA (human-readable) =========
# Missingness
miss = df.isna().mean().sort_values(ascending=False).head(40)
plt.figure(figsize=(14,7))
plt.bar(range(len(miss)), miss.values)
plt.ylabel("Fraction Missing"); plt.xlabel("Columns")
plt.title("Top Missingness by Column")
plt.xticks(range(len(miss)), [pretty(c) for c in miss.index], rotation=70, ha='right')
savefig("eda_missingness_top40.png"); plt.close()

# Numeric histograms (top 16 informative)
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
coverage = df[num_cols_all].notna().mean()
variance = df[num_cols_all].var(numeric_only=True, ddof=0).fillna(0)
score = (coverage * (variance.replace(0, np.nan))).sort_values(ascending=False).dropna()
top_cols = score.head(min(16, len(score))).index.tolist()
n = len(top_cols); rows = int(np.ceil(n/4)); cols = min(4, n)
plt.figure(figsize=(4.5*cols, 3.6*rows))
for i, c in enumerate(top_cols, 1):
    ax = plt.subplot(rows, cols, i)
    data = df[c].dropna()
    bins = 40 if data.nunique() > 20 else min(20, data.nunique())
    ax.hist(data, bins=bins)
    ax.set_title(pretty(c)); ax.set_xlabel(pretty(c)); ax.set_ylabel("Count")
plt.suptitle("Numeric Distributions (Top Well-Filled Features)", y=1.02)
plt.tight_layout()
savefig("eda_numeric_histograms.png"); plt.close()

# Correlation heatmap (triangular, top 30 by coverage)
num_cols = df.select_dtypes(include=[np.number]).columns
best = df[num_cols].notna().sum().sort_values(ascending=False).head(min(30, len(num_cols))).index
corr = df[best].corr(numeric_only=True)
plt.figure(figsize=(14,12))
if SEABORN:
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                xticklabels=[pretty(c) for c in corr.columns],
                yticklabels=[pretty(c) for c in corr.index])
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
else:
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1); plt.colorbar()
    plt.xticks(range(len(corr.columns)), [pretty(c) for c in corr.columns], rotation=45, ha='right', fontsize=9)
    plt.yticks(range(len(corr.index)), [pretty(c) for c in corr.index], rotation=0, fontsize=9)
plt.title("Correlation Heatmap (Sampled Numeric Features)")
plt.xlabel("Features"); plt.ylabel("Features")
savefig("eda_correlation_heatmap.png"); plt.close()

# Molecule frequency
mol_counter = Counter()
for lst in df['molecules_list']: mol_counter.update(lst)
mol_freq = pd.DataFrame(mol_counter.most_common(20), columns=['Molecule','Count'])
mol_freq.to_csv(os.path.join(OUT_DIR, "molecule_frequency_top20.csv"), index=False)
plt.figure(figsize=(12,8))
plt.barh(mol_freq['Molecule'][::-1], mol_freq['Count'][::-1])
plt.title("Top 20 Molecules Mentioned"); plt.xlabel("Count"); plt.ylabel("Molecule")
savefig("eda_molecules_top20.png"); plt.close()

# ========= 9) Regression imputation =========
TARGETS_REG = [('pl_bmasse','mass_earth'), ('pl_eqt','eq_temp'), ('pl_dens','density')]
for col,_ in TARGETS_REG:
    if col not in df.columns: df[col] = np.nan

numeric_cols_all = [c for c in df.select_dtypes(include=[np.number]).columns]
BASE_EXCLUDE = {'kepid'}

# helpers to identify planet radius & density columns (avoid catching stellar radius)
def is_planet_radius_col(c: str) -> bool:
    c = c.lower()
    return c.startswith('pl_rad') or c in {'pl_rade','pl_radj','pl_rads'} or 'planet_radius' in c

def is_density_col(c: str) -> bool:
    c = c.lower()
    return ('pl_dens' in c) or ('dens_from_mass_radius' in c) or (c.endswith('_dens') or 'density' in c)

def dynamic_cv_splits(n_samples, max_splits=5):
    if n_samples >= 5: return min(max_splits, 5)
    return max(0, n_samples)

def leakage_guard_feature_set(X, y, feature_cols, target_col, corr_thresh=0.995):
    safe_cols = [c for c in feature_cols if target_col not in c]
    drop_cols = set()
    for c in list(safe_cols):
        if c not in X.columns: continue
        s = pd.concat([X[c], y], axis=1).dropna()
        if len(s) >= 10:
            corr = s.corr().iloc[0,1]
            if pd.notna(corr) and abs(corr) > corr_thresh:
                drop_cols.add(c)
    safe_cols = [c for c in safe_cols if c not in drop_cols]
    return safe_cols, drop_cols

def run_regression_imputer(df, target_col, out_prefix):
    y = df[target_col]
    feature_cols = [c for c in numeric_cols_all if c not in BASE_EXCLUDE and c != target_col]

    # --- extra guard: when imputing MASS, never use planet-radius or density-like features
    if target_col == 'pl_bmasse':
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if not is_planet_radius_col(c) and not is_density_col(c)]
        removed = before - len(feature_cols)
        if removed > 0:
            print(f"[LEAK-GUARD mass-imputer] Removed {removed} radius/density features from mass model.")

    X = df[feature_cols]
    mask_train = y.notna()
    X_train, y_train = X[mask_train], y[mask_train]
    n_train = len(X_train)
    cv_splits = min(5, max(0, n_train // max(1, n_train)))  # keep function signature happy
    cv_splits = 5 if n_train >= 5 else n_train  # simple safe rule

    if target_col == 'pl_dens' and n_train < 2:
        print("[REG-pl_dens] No labeled samples. Using physics fallback ρ = 5.51 * (pl_bmasse_filled / pl_rade^3).")
        mass_f = df['pl_bmasse_filled'] if 'pl_bmasse_filled' in df.columns else df['pl_bmasse']
        y_filled = df[target_col].copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            dens_calc = 5.51 * (mass_f / (df['pl_rade']**3))
        y_filled.loc[df[target_col].isna()] = dens_calc[df[target_col].isna()]
        pd.DataFrame({'index': df.index, f'{target_col}_orig': df[target_col], f'{target_col}_imputed': y_filled})\
          .to_csv(os.path.join(OUT_DIR, f"reg_predictions_{out_prefix}.csv"), index=False)
        return y_filled, {'baseline': None, 'strong': None}

    baseline = Pipeline([('imp', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler(with_mean=False)),
                         ('model', LinearRegression())])
    if LGB_AVAILABLE:
        model = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_estimators=600, learning_rate=0.05, max_depth=-1)
    else:
        model = GradientBoostingRegressor(random_state=RANDOM_STATE, n_estimators=500, learning_rate=0.05, max_depth=3)
    strong = Pipeline([('imp', SimpleImputer(strategy='median')), ('model', model)])

    # leakage guard on train subset
    safe_cols, dropped = leakage_guard_feature_set(X_train, y_train, feature_cols, target_col)
    if dropped:
        print(f"[LEAK-GUARD {target_col}] Dropped near-duplicate/leaky cols: {sorted(list(dropped))[:8]}{' ...' if len(dropped)>8 else ''}")
    X_train = X_train[safe_cols]; X_full = X[safe_cols]

    metrics_bl = metrics_st = (np.nan, np.nan, np.nan)
    if len(X_train) >= 5:
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        def cv_reg_metrics(pipe, X_, y_):
            rmses, maes, r2s = [], [], []
            for tr, va in cv.split(X_, y_):
                Xtr, Xva = X_.iloc[tr], X_.iloc[va]
                ytr, yva = y_.iloc[tr], y_.iloc[va]
                pipe.fit(Xtr, ytr)
                p = pipe.predict(Xva)
                rmses.append(np.sqrt(mean_squared_error(yva, p)))
                maes.append(mean_absolute_error(yva, p))
                r2s.append(r2_score(yva, p))
            return float(np.mean(rmses)), float(np.mean(maes)), float(np.mean(r2s))
        metrics_bl = cv_reg_metrics(baseline, X_train, y_train)
        metrics_st = cv_reg_metrics(strong,   X_train, y_train)
        print(f"[REG-{target_col}] Baseline LinReg CV: RMSE={metrics_bl[0]:.3f} MAE={metrics_bl[1]:.3f} R2={metrics_bl[2]:.3f}")
        print(f"[REG-{target_col}] Strong Model   CV: RMSE={metrics_st[0]:.3f} MAE={metrics_st[1]:.3f} R2={metrics_st[2]:.3f}")
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                strong, X_train, y_train, cv=cv, scoring='r2',
                train_sizes=np.linspace(0.2, 1.0, 5)
            )
            plt.figure(figsize=(7,5))
            plt.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Train R2')
            plt.plot(train_sizes, val_scores.mean(axis=1), marker='s', label='CV R2')
            plt.xlabel("Training Size"); plt.ylabel("R2 Score"); plt.title(f"Learning Curve ({target_col})"); plt.legend()
            savefig(f"reg_learning_curve_{out_prefix}.png"); plt.close()
        except Exception:
            pass
    else:
        print(f"[REG-{target_col}] Too few labeled samples for CV (n={len(X_train)}). Fitting strong model without CV.")

    strong.fit(X_train, y_train)
    import joblib
    joblib.dump(strong, os.path.join(OUT_DIR, f"reg_model_{out_prefix}.joblib"))
    y_pred_all = strong.predict(X_full)
    y_filled = df[target_col].copy()
    y_filled.loc[df[target_col].isna()] = y_pred_all[df[target_col].isna()]
    pd.DataFrame({'index': df.index, f'{target_col}_orig': df[target_col], f'{target_col}_imputed': y_filled}) \
      .to_csv(os.path.join(OUT_DIR, f"reg_predictions_{out_prefix}.csv"), index=False)
    return y_filled, {'baseline': metrics_bl, 'strong': metrics_st}

metrics_reg_all = {}
for col, alias in TARGETS_REG:
    filled, metrics = run_regression_imputer(df, col, out_prefix=alias)
    df[col + "_filled"] = filled
    metrics_reg_all[col] = metrics

if 'pl_rade' in df.columns and 'pl_bmasse_filled' in df.columns:
    with np.errstate(divide='ignore', invalid='ignore'):
        df['dens_from_mass_radius_filled'] = 5.51 * (df['pl_bmasse_filled'] / (df['pl_rade']**3))

with open(os.path.join(OUT_DIR, "regression_cv_metrics.json"), "w") as f:
    json.dump(metrics_reg_all, f, indent=2)

# ========= 10) Planet type classification (with anti-leak) =========
def label_planet_type(row):
    r = row.get('pl_rade', np.nan)
    if pd.isna(r): return np.nan
    if r <= 1.5:       return 'Earth-like'
    elif r <= 2.5:     return 'Super-Earth'
    elif r <= 6.0:     return 'Neptune-like'
    else:              return 'Jupiter-like'

if 'pl_dens_filled' not in df.columns:
    df['pl_dens_filled'] = df.get('pl_dens', np.nan)

df['planet_type_rule'] = df.apply(label_planet_type, axis=1)

# strict ban list
LEAK_KEYS = ('pl_rade', 'rade', 'radj', 'radius')
BAN_FEATURES = {'pl_dens','pl_dens_filled','dens_from_mass_radius','dens_from_mass_radius_filled'}

CANDIDATE_FEATURES = []
for c in ['pl_bmasse_filled','pl_eqt_filled',
          'st_teff','st_mass','st_rad','st_met','st_logg',
          'sy_dist','sy_snum','sy_pnum','sy_vmag','sy_kmag','sy_jmag']:
    if c in df.columns and c not in BAN_FEATURES and not any(k in c for k in LEAK_KEYS):
        CANDIDATE_FEATURES.append(c)

# dynamic anti-leak: drop features that correlate too strongly with radius
if 'pl_rade' in df.columns:
    corr_warn = []
    for c in list(CANDIDATE_FEATURES):
        s = df[[c,'pl_rade']].dropna()
        if len(s) >= 50:
            rho = s.corr().iloc[0,1]
            if pd.notna(rho) and abs(rho) >= 0.98:
                CANDIDATE_FEATURES.remove(c)
                corr_warn.append((c, float(rho)))
    if corr_warn:
        dropped_cols = [f"{c} (corr={r:.3f})" for c,r in corr_warn]
        print("[LEAK-GUARD CLF] Dropped features highly correlated with radius:", dropped_cols)

if len(CANDIDATE_FEATURES) < 6:
    CANDIDATE_FEATURES = [c for c in df.select_dtypes(include=[np.number]).columns
                          if c not in {'kepid'} and c not in BAN_FEATURES
                          and not any(k in c for k in LEAK_KEYS)][:25]

clf_df = df[df['planet_type_rule'].notna()].copy()
X_clf = clf_df[CANDIDATE_FEATURES]
y_clf = clf_df['planet_type_rule'].astype('category')

num_imputer = SimpleImputer(strategy='median')

clf_baseline = Pipeline([
    ('imp', num_imputer),
    ('scaler', StandardScaler(with_mean=False)),
    ('model', LogisticRegression(max_iter=300, multi_class='multinomial', random_state=RANDOM_STATE))
])

clf_rf = Pipeline([
    ('imp', num_imputer),
    ('model', RandomForestClassifier(random_state=RANDOM_STATE))
])
param_grid = {
    'model__n_estimators': [300, 600],
    'model__max_depth': [None, 10, 20],
    'model__min_samples_split': [2, 5],
    'model__min_samples_leaf': [1, 2]
}

X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
)

clf_baseline.fit(X_train, y_train)
y_pred_base = clf_baseline.predict(X_test)

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(clf_rf, param_grid, cv=cv_strat, scoring='f1_macro', refit=True, verbose=0)
grid.fit(X_train, y_train)
best_clf = grid.best_estimator_
y_pred_rf = best_clf.predict(X_test)
print("[CLF] Best RF params:", grid.best_params_)

def save_report(name, y_true, y_pred, labels):
    rpt = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    rpt_df = pd.DataFrame(rpt).transpose()
    rpt_df.to_csv(os.path.join(OUT_DIR, f"classification_report_{name}.csv"))
    print(f"[CLF-{name}] macro-F1: {rpt_df.loc['macro avg','f1-score']:.3f}  accuracy: {rpt_df.loc['accuracy','precision']:.3f}")

labels_list = sorted(y_clf.unique().tolist())
save_report("baseline_logreg", y_test, y_pred_base, labels_list)
save_report("rf_tuned",       y_test, y_pred_rf,   labels_list)

cm = confusion_matrix(y_test, y_pred_rf, labels=labels_list)
plt.figure(figsize=(6,5))
if SEABORN:
    sns.heatmap(pd.DataFrame(cm, index=labels_list, columns=labels_list), annot=True, fmt='d', cmap='Blues')
else:
    plt.imshow(cm, cmap='Blues'); plt.colorbar()
    for (i,j),v in np.ndenumerate(cm): plt.text(j, i, int(v), ha='center', va='center')
    plt.xticks(range(len(labels_list)), labels_list, rotation=45, ha='right')
    plt.yticks(range(len(labels_list)), labels_list)
plt.ylabel("True"); plt.xlabel("Predicted"); plt.title("Confusion Matrix — RF (tuned)")
savefig("clf_confusion_matrix_rf.png"); plt.close()

if hasattr(best_clf.named_steps['model'], "predict_proba"):
    from sklearn.preprocessing import label_binarize
    X_test_imp = num_imputer.transform(X_test)
    y_score = best_clf.named_steps['model'].predict_proba(X_test_imp)
    y_test_bin = label_binarize(y_test, classes=labels_list)
    try:
        auc_macro = roc_auc_score(y_test_bin, y_score, average='macro', multi_class='ovr')
        print(f"[CLF-RF] ROC AUC (macro, OvR): {auc_macro:.3f}")
    except Exception:
        pass
    plt.figure(figsize=(7,5))
    for i, lab in enumerate(labels_list):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=f"{lab} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (OvR) — RF")
    plt.legend()
    savefig("clf_pr_curve_rf.png"); plt.close()

# Misclassifications
y_test_s = pd.Series(y_test, index=X_test.index, name="true_label")
y_pred_s = pd.Series(y_pred_rf, index=X_test.index, name="pred_label")
mis_idx = (y_test_s != y_pred_s)
cols_core = (['kepid'] if 'kepid' in clf_df.columns else []) + CANDIDATE_FEATURES
mis_table = clf_df.loc[X_test.index, cols_core].copy()
mis_table = mis_table.loc[mis_idx]
mis_table = pd.concat([mis_table, y_test_s.loc[mis_idx], y_pred_s.loc[mis_idx]], axis=1)
mis_table.to_csv(os.path.join(OUT_DIR, "misclassifications_rf.csv"), index=False)

# Feature importances (human-readable)
rf_model = best_clf.named_steps['model']
imp = pd.Series(rf_model.feature_importances_, index=CANDIDATE_FEATURES).sort_values(ascending=False)
imp.to_csv(os.path.join(OUT_DIR, "feature_importances_rf.csv"))
plt.figure(figsize=(8,6))
topn = imp.head(20)
plt.barh([pretty(n) for n in topn.index[::-1]], topn.values[::-1])
plt.title("Top Feature Importances — RF")
plt.xlabel("Importance"); plt.ylabel("Feature")
savefig("clf_feature_importances_rf.png"); plt.close()

if SHAP_AVAILABLE:
    try:
        explainer = shap.TreeExplainer(rf_model)
        X_imp_train = num_imputer.transform(X_train)
        shap_vals = explainer.shap_values(X_imp_train)
        plt.figure()
        shap.summary_plot(shap_vals, X_imp_train, feature_names=[pretty(n) for n in CANDIDATE_FEATURES], show=False)
        savefig("clf_shap_summary_rf.png"); plt.close()
    except Exception as e:
        print("[SHAP] Skipped due to error:", str(e))

# Apply RF to all feasible rows
df_pred = df.copy()
usable_mask = df_pred[CANDIDATE_FEATURES].notna().sum(axis=1) >= max(3, int(0.4*len(CANDIDATE_FEATURES)))
X_all = df_pred.loc[usable_mask, CANDIDATE_FEATURES]
X_all_imp = num_imputer.transform(X_all)
pred_all = best_clf.named_steps['model'].predict(X_all_imp)
df_pred.loc[usable_mask, 'planet_type_pred_rf'] = pred_all
df_pred['planet_type_pred_rf'] = df_pred['planet_type_pred_rf'].astype('category')

# ========= 11) Molecule multi-label =========
TOP_K = 10
top_molecules = [mol for mol,_ in mol_counter.most_common(TOP_K)]
mlb = MultiLabelBinarizer(classes=top_molecules)
Y_mol = mlb.fit_transform(df['molecules_list'].apply(lambda lst: [m for m in lst if m in top_molecules]))
mol_present_rows = Y_mol.sum(axis=1) > 0

MOL_FEATURES = list(dict.fromkeys(
    [c for c in ['pl_bmasse_filled','pl_dens_filled','pl_eqt_filled',
                 'dens_from_mass_radius','dens_from_mass_radius_filled',
                 'st_teff','st_mass','st_rad','sy_dist','st_met','st_logg',
                 'sy_snum','sy_pnum','sy_vmag','sy_kmag','sy_jmag'] if c in df.columns]
))
if len(MOL_FEATURES) < 6:
    MOL_FEATURES = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {'kepid'}][:25]

X_mfeat_all = df[MOL_FEATURES]
X_mfeat = X_mfeat_all[mol_present_rows]
Y_mol_used = Y_mol[mol_present_rows]

X_mtrain, X_mtest, Y_mtrain, Y_mtest = train_test_split(
    X_mfeat, Y_mol_used, test_size=0.2, random_state=RANDOM_STATE
)

mol_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('clf', OneVsRestClassifier(RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
        random_state=RANDOM_STATE
    )))
])
mol_pipe.fit(X_mtrain, Y_mtrain)
Y_pred = mol_pipe.predict(X_mtest)

report_multi = {
    "f1_micro": float(f1_score(Y_mtest, Y_pred, average='micro', zero_division=0)),
    "f1_macro": float(f1_score(Y_mtest, Y_pred, average='macro', zero_division=0)),
    "support": int(Y_mtest.shape[0])
}
with open(os.path.join(OUT_DIR, "molecule_multilabel_scores.json"), "w") as f:
    json.dump(report_multi, f, indent=2)
print(f"[MOLS] F1-micro={report_multi['f1_micro']:.3f}  F1-macro={report_multi['f1_macro']:.3f}")

try:
    Y_proba = mol_pipe.predict_proba(X_mtest)
    plt.figure(figsize=(8,6))
    for i, mol in enumerate(top_molecules):
        precision, recall, _ = precision_recall_curve(Y_mtest[:, i], Y_proba[:, i])
        ap = average_precision_score(Y_mtest[:, i], Y_proba[:, i])
        plt.plot(recall, precision, label=f"{mol} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curves — Molecule Multi-Label (Top-K)")
    plt.legend(fontsize=8); savefig("molecule_multilabel_pr_curves.png"); plt.close()
except Exception:
    pass

# ========= 12) Persist =========
import joblib
joblib.dump(mol_pipe, os.path.join(OUT_DIR, "molecule_multilabel_model.joblib"))
pd.DataFrame({"top_k_molecules": top_molecules}).to_csv(os.path.join(OUT_DIR, "molecule_topK_list.csv"), index=False)

keep_cols = [c for c in [
    'kepid','pl_name_std','host_name_std','pl_rade',
    'pl_bmasse','pl_bmasse_filled','pl_dens','pl_dens_filled','pl_eqt','pl_eqt_filled',
    'dens_from_mass_radius','dens_from_mass_radius_filled',
    'planet_type_rule','planet_type_pred_rf','predicted_molecules_topK'
] if c in df_pred.columns]

final_csv_path = os.path.join(OUT_DIR, "task2_characterization_with_kepid.csv")
df_pred[keep_cols].to_csv(final_csv_path, index=False)
print(f"[CSV] Saved: {final_csv_path}")

df_pred.to_csv(os.path.join(OUT_DIR, "merged_with_predictions_task2.csv"), index=False)

joblib.dump(best_clf, os.path.join(OUT_DIR, "clf_rf_tuned_model.joblib"))
joblib.dump(clf_baseline, os.path.join(OUT_DIR, "clf_logreg_baseline_model.joblib"))
joblib.dump(num_imputer, os.path.join(OUT_DIR, "clf_numeric_imputer.joblib"))

# ========= 13) Ethics & Reproducibility =========
ethics_note = """
Ethics & Reproducibility (Task 2):
• Data sources: NASA Exoplanet Archive derivatives (Planetary Systems, Stellar Hosts, Atmospheres).
• Non-personal astronomical data — GDPR not applicable to individuals; complies with UH ethics (no human data).
• Stored securely in OneDrive + versioned code on GitHub; outputs timestamped for reproducibility.
• Bias awareness: detection/confirmation biases in exoplanet catalogues can skew class distributions; macro-F1 used.
• Sustainability: classical ML (RF/GBM) keeps compute modest; report training time where relevant.
• Molecule predictions are exploratory proxies; real inference needs spectroscopic retrieval.
"""
with open(os.path.join(OUT_DIR, "ETHICS_AND_REPRO_NOTES.txt"), "w") as f:
    f.write(ethics_note.strip())
print(ethics_note)

print("\n[DONE] Task 2 completed with anti-leak fixes and readable EDA.")
print(f"Outputs folder: {OUT_DIR}")

