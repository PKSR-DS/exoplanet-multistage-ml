# -*- coding: utf-8 -*-
"""
Task 5 — Weight Sensitivity (exact headers version)

Input CSV columns (confirmed):
  kepid, pl_name, ProbeScore,
  Score_Detection, Score_Habitability, Score_Type, Score_Radio,
  pl_rade, pl_bmasse, pl_eqt, pl_dens, rank
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CHANGE THIS PATH ONLY IF YOUR RUN FOLDER CHANGES ===
CSV = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task5\run_20250827_033026\task5_dashboard_data.csv"

OUT_DIR = os.path.dirname(CSV)
assert os.path.isfile(CSV), f"CSV not found:\n{CSV}"

print(f"[INFO] Loading: {CSV}")
df = pd.read_csv(CSV)

# ---- Required columns (exact names you gave) ----
REQ = [
    "kepid", "pl_name",
    "Score_Detection", "Score_Habitability", "Score_Type", "Score_Radio"
]
missing = [c for c in REQ if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Coerce scores to float; fill NAs safely if any
score_cols = ["Score_Detection", "Score_Habitability", "Score_Type", "Score_Radio"]
for c in score_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[c].isna().any():
        # fall back to column mean (or 0.5 if all-NA)
        df[c] = df[c].fillna(df[c].mean() if not df[c].isna().all() else 0.5)

# (Optional) clamp to [0,1] in case anything is slightly out-of-range
for c in score_cols:
    df[c] = df[c].clip(0, 1)

# ---- Define weight profiles to test ----
W = {
    "default":     dict(d=0.35, h=0.35, p=0.20, c=0.10),
    "comms_heavy": dict(d=0.30, h=0.30, p=0.15, c=0.25),
    # Add more if you want:
    # "detection_heavy": dict(d=0.50, h=0.25, p=0.15, c=0.10),
    # "habitability_heavy": dict(d=0.20, h=0.50, p=0.20, c=0.10),
}

def calc_priority(w: dict) -> pd.Series:
    """Linear composite using your four component scores."""
    w = dict(w)  # copy
    total = sum(w.values())
    if abs(total - 1.0) > 1e-9:
        # normalise if the user added a profile that doesn't sum to 1
        w = {k: v / total for k, v in w.items()}
    s = (
        w["d"] * df["Score_Detection"] +
        w["h"] * df["Score_Habitability"] +
        w["p"] * df["Score_Type"] +
        w["c"] * df["Score_Radio"]
    )
    return s

# Compute priorities for each profile and store in new columns
for name, w in W.items():
    col = f"Priority__{name}"
    df[col] = calc_priority(w)

# ---- Save weight-sensitivity results CSV ----
out_csv = os.path.join(OUT_DIR, "task5_weight_sensitivity_results.csv")
cols_to_save = (["kepid", "pl_name"] + score_cols +
                [f"Priority__{name}" for name in W.keys()])
df[cols_to_save].to_csv(out_csv, index=False)
print("[SAVED]", out_csv)

# ---- Figures: histograms and pair-scatter (Detection vs Habitability) ----
for name in W.keys():
    s = df[f"Priority__{name}"]

    # Histogram
    plt.figure()
    plt.hist(s, bins=40)
    plt.title(f"Target Priority — {name}")
    plt.xlabel("score"); plt.ylabel("count")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"priority_hist__{name}.png")
    plt.savefig(out_png, dpi=120); plt.close()
    print("[SAVED]", out_png)

    # Pair scatter (Detection vs Habitability, coloured by priority)
    plt.figure()
    sc = plt.scatter(
        df["Score_Detection"], df["Score_Habitability"],
        c=s, s=10, alpha=0.7
    )
    plt.colorbar(sc, label="priority")
    plt.title(f"Detection vs Habitability — coloured by {name}")
    plt.xlabel("Score_Detection"); plt.ylabel("Score_Habitability")
    plt.tight_layout()
    out_png2 = os.path.join(OUT_DIR, f"pair_scatter__{name}.png")
    plt.savefig(out_png2, dpi=120); plt.close()
    print("[SAVED]", out_png2)

# ---- Top-50 overlap (stability) between the first two profiles, if present ----
names = list(W.keys())
if len(names) >= 2:
    col0, col1 = f"Priority__{names[0]}", f"Priority__{names[1]}"
    K = 50

    # Rank by each profile
    top0_idx = np.argsort(-df[col0].values)[:K]
    top1_idx = np.argsort(-df[col1].values)[:K]

    top0 = set(df["kepid"].iloc[top0_idx])
    top1 = set(df["kepid"].iloc[top1_idx])

    jaccard = len(top0 & top1) / len(top0 | top1) if (top0 | top1) else 0.0
    print(f"[INFO] Top-{K} Jaccard({names[0]} vs {names[1]}): {jaccard:.3f}")

    # Save Top-50 tables for each profile (nice to attach to the report/appendix)
    top0_csv = os.path.join(OUT_DIR, f"top_{K}__{names[0]}.csv")
    top1_csv = os.path.join(OUT_DIR, f"top_{K}__{names[1]}.csv")

    df.loc[top0_idx, ["kepid","pl_name", col0] + score_cols] \
      .sort_values(col0, ascending=False) \
      .to_csv(top0_csv, index=False)
    df.loc[top1_idx, ["kepid","pl_name", col1] + score_cols] \
      .sort_values(col1, ascending=False) \
      .to_csv(top1_csv, index=False)

    print("[SAVED]", top0_csv)
    print("[SAVED]", top1_csv)

print("[DONE]")
