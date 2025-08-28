# Task 2 — Leakage sanity check (ablation) — safe + low-memory
# Save as: task2_ablation_leakage_check_minmem.py

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ========= Paths (edit if needed) =========
DATA = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task2\run_20250826_104658\merged_raw.csv"
PRED_PATH = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task2\run_20250826_104658\merged_with_predictions_task2.csv"  # set to None if not available

OUT_DIR = os.path.join(os.path.dirname(DATA), "ablation_check"); os.makedirs(OUT_DIR, exist_ok=True)

# ========= Helpers =========
def canonicalise_label(s: str) -> str | None:
    """Map many variants into four canonical classes."""
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.lower().replace("-", " ").strip()
    # buckets
    if "earth" in t and "super" not in t:
        return "Earth-like"
    if "super" in t and "earth" in t:
        return "Super-Earth"
    if "neptune" in t or "mini neptune" in t or "sub neptune" in t:
        return "Neptune-like"
    if "jupiter" in t or "giant" in t or "jovian" in t or "gas" in t:
        return "Jupiter-like"
    return None

def rule_label_from_features(r, m) -> str:
    """Very coarse fallback using radius/mass."""
    r = float(r) if pd.notna(r) else np.nan
    m = float(m) if pd.notna(m) else np.nan
    # broad, consistent with your report buckets
    if (pd.notna(r) and r >= 6.0) or (pd.notna(m) and m >= 50.0): return "Jupiter-like"
    if (pd.notna(r) and 2.5 <= r < 6.0) or (pd.notna(m) and 10.0 <= m < 50.0): return "Neptune-like"
    if (pd.notna(r) and 1.5 <= r < 2.5) or (pd.notna(m) and 2.0 <= m < 10.0): return "Super-Earth"
    return "Earth-like"

def safe_label_series(df: pd.DataFrame, pred_path: str | None) -> pd.Series:
    """Build a clean label series with mapping + fallback rules."""
    # 1) existing
    for c in ("planet_type", "planet_class"):
        if c in df.columns:
            y0 = df[c].astype(str).map(canonicalise_label)
            break
    else:
        y0 = pd.Series([None]*len(df), index=df.index)

    # 2) map from predictions by name (no heavy merge)
    if y0.isna().any() and pred_path and os.path.exists(pred_path):
        pred = pd.read_csv(pred_path)
        # choose a label col
        pred_label = None
        for c in ("planet_type", "planet_class", "pred_class", "pred_label"):
            if c in pred.columns:
                pred_label = c; break
        if pred_label is None:
            poss = [c for c in pred.columns if "class" in c.lower() or "type" in c.lower()]
            pred_label = poss[0] if poss else None

        # choose a name key
        join_key = "pl_name" if "pl_name" in df.columns and "pl_name" in pred.columns else None
        if join_key is None:
            for c in ("pl_name_std","host_name_std","hostname"):
                if c in df.columns and c in pred.columns:
                    join_key = c; break

        if pred_label and join_key:
            m = pred[[join_key, pred_label]].dropna().drop_duplicates(subset=[join_key], keep="last")
            m[pred_label] = m[pred_label].astype(str).map(canonicalise_label)
            map_dict = dict(zip(m[join_key], m[pred_label]))
            fill_from_pred = df[join_key].map(map_dict)
            y0 = y0.fillna(fill_from_pred)

    # 3) feature-based fallback
    need = y0.isna()
    if need.any():
        y0.loc[need] = [
            rule_label_from_features(r, m)
            for r, m in zip(df.get("pl_rade"), df.get("pl_bmasse"))
        ]
    return y0

def plot_cm(cm, labels, title, path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig(path, dpi=140); plt.close()

# ========= Load =========
print("[INFO] Loading:", DATA)
df = pd.read_csv(DATA)

# target
y = safe_label_series(df, PRED_PATH)
X = df.select_dtypes(include=[np.number]).copy()

# drop rows with NA target
mask = y.notna()
X, y = X.loc[mask], y.loc[mask]

# sanity on class counts
vc = y.value_counts()
print("[INFO] Class counts:", vc.to_dict())

# If the smallest class is 1, try to reassign those few using rule again; if still <2, we’ll drop stratify
min_class = vc.min()
use_stratify = True
if min_class < 2:
    print("[WARN] A class has <2 samples; falling back to non-stratified split.")
    use_stratify = False

# ========= Split =========
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if use_stratify else None
)

# ========= Train — FULL =========
rf_full = RandomForestClassifier(
    n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
)
rf_full.fit(Xtr, ytr)
yhat = rf_full.predict(Xte)
rep_full = classification_report(yte, yhat, digits=3)
print("\n== FULL FEATURES =="); print(rep_full)
with open(os.path.join(OUT_DIR, "report_full.txt"), "w", encoding="utf-8") as f: f.write(rep_full)
cm_full = confusion_matrix(yte, yhat, labels=rf_full.classes_)
plot_cm(cm_full, rf_full.classes_, "Confusion Matrix — Full", os.path.join(OUT_DIR, "cm_full.png"))

# ========= Train — ABLATION (drop mass/radius/eqt/insol) =========
drop_kw = ("bmasse","bmass","rade","radj","eqt","insol")
Xa = X.drop(columns=[c for c in X.columns if any(k in c.lower() for k in drop_kw)], errors="ignore")
Xtr_a, Xte_a, ytr_a, yte_a = train_test_split(
    Xa, y, test_size=0.2, random_state=42,
    stratify=y if use_stratify else None
)
rf_abl = RandomForestClassifier(
    n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced_subsample"
)
rf_abl.fit(Xtr_a, ytr_a)
yhat_a = rf_abl.predict(Xte_a)
rep_abl = classification_report(yte_a, yhat_a, digits=3)
print("\n== ABLATION (no mass/radius/eqt/insol) =="); print(rep_abl)
with open(os.path.join(OUT_DIR, "report_ablation.txt"), "w", encoding="utf-8") as f: f.write(rep_abl)
cm_abl = confusion_matrix(yte_a, yhat_a, labels=rf_abl.classes_)
plot_cm(cm_abl, rf_abl.classes_, "Confusion Matrix — Ablation", os.path.join(OUT_DIR, "cm_ablation.png"))

# ========= Summary =========
summary = {
    "rows_used": int(len(X)),
    "classes": y.value_counts().to_dict(),
    "used_stratify": bool(use_stratify),
    "n_features_full": int(X.shape[1]),
    "n_features_ablated": int(Xa.shape[1]),
    "dropped_cols_count": int(X.shape[1] - Xa.shape[1]),
}
with open(os.path.join(OUT_DIR, "ablation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n[SAVED] Reports & figures ->", OUT_DIR)
