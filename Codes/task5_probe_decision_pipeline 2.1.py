# -*- coding: utf-8 -*-
"""
Task 5 — AI Space Probe Decision Simulation (Composite Target Ranking)
Author: Praveen Kumar Savariraj (Student ID: 23089117)
Module: 7PAM2002 — UH MSc Data Science Final Project

This script aggregates Task 1–4 outputs into an interpretable, tunable
Target Priority Score for probe target selection.

It auto-detects the most recent CSVs in your Output folders, is robust to
key naming, and prioritises calibrated detection probabilities if present.
"""

import os
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================== User config ========================

# Change if your base roots differ
BASE_DATA_DIR   = r"D:\UH One drive\OneDrive - University of Hertfordshire\Final Project\Data"
BASE_OUTPUT_DIR = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output"

# Where this run will be written
RUN_DIR = os.path.join(BASE_OUTPUT_DIR, "Task5", datetime.now().strftime("run_%Y%m%d_%H%M%S"))
FIG_DIR = os.path.join(RUN_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Weights (will be normalised to sum=1)
WEIGHTS = {
    "detection": 0.35,
    "habitability": 0.35,
    "physical": 0.20,
    "comms": 0.10,
}
TOP_N = 50  # Top-N export

# Filenames we try to auto-pick (latest by mtime)
PATTERNS = {
    "task1": ["task1_transit_predictions.csv", "*task1*predictions*.csv"],
    "task2": [
        "task2_characterization_with_kepid_kepid_mapped.csv",
        "task2_characterization_with_kepid.csv",
        "*task2*characterization*with*kepid*.csv",
        "merged_with_predictions_task2.csv",   # fallback
        "merged_raw.csv"                       # last-ditch (no labels)
    ],
    "task3": [
        "task3_habitability_with_predictions__with_kepid.csv",
        "task3_habitability_with_predictions.csv",
        "*task3*habitability*with*kepid*.csv"
    ],
    "task4": [
        "task4_radio_silence_scores.csv",
        "task4_radio_silence_predictions.csv",
        "*task4*radio*silence*score*.csv",
        "*task4*radio*silence*pred*.csv",
    ],
}

# ======================== Utilities ==========================

def _norm_name(col):
    return str(col).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def find_first(df, candidates):
    """
    Return the first existing column in df that matches any candidate name
    (allows fuzzy 'contains' match in a pinch).
    """
    if df is None or df.empty:
        return None
    exact = {_norm_name(c): c for c in df.columns}
    for c in candidates:
        key = _norm_name(c)
        if key in exact:
            return exact[key]
    # loose contains
    for col in df.columns:
        if any(_norm_name(c) in _norm_name(col) for c in candidates):
            return col
    return None

def pick_first_existing(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi - lo == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)

def safe_float(df, cols=None):
    if df is None:
        return df
    if cols is None:
        cols = [c for c in df.columns if c not in ["kepid", "pl_name"]]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def glob_latest(base_dir, patterns):
    """
    Search recursively under base_dir for files matching any of the patterns.
    Return the most recently modified one, else None.
    """
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(base_dir, "**", pat), recursive=True))
    if not hits:
        return None
    hits.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return hits[0]

def autopick(which):
    # Try Output first, then Data
    path = glob_latest(BASE_OUTPUT_DIR, PATTERNS[which]) or glob_latest(BASE_DATA_DIR, PATTERNS[which])
    if path:
        print(f"[AUTO] {which.title()} file: {path}")
    else:
        print(f"[AUTO] {which.title()} file: NOT FOUND")
    return path

def smart_read_csv(path):
    if not path or not os.path.exists(path):
        return None
    print(f"[LOAD] {path}")
    return pd.read_csv(path)

# ---------- kepid mapping helpers ----------

def attach_kepid(df, ref_t2=None):
    """
    Try hard to ensure a 'kepid' column exists.
    1) direct aliases (kepid/kepler_id/kic/kic_id/kicid…)
    2) map via pl_name (using Task2 if provided)
    3) map via kepoi_name (using Task2 if provided)
    Returns a copy (possibly merged) or None if not possible.
    """
    if df is None or df.empty:
        return None
    df = df.copy()

    # 1) Direct aliases
    aliases = ["kepid", "kepler_id", "kepid_num", "kic", "kic_id", "kicid", "kicnumber", "kic_number"]
    c = find_first(df, aliases)
    if c:
        if c != "kepid":
            df = df.rename(columns={c: "kepid"})
        df["kepid"] = pd.to_numeric(df["kepid"], errors="coerce")
        return df

    # 2) Map via pl_name
    pl_df = find_first(df, ["pl_name", "planet_name", "koi_name", "kepoi_name"])
    if ref_t2 is not None:
        pl_t2 = "pl_name" if "pl_name" in ref_t2.columns else find_first(ref_t2, ["pl_name", "planet_name"])
        if pl_df and pl_t2:
            m = ref_t2[[pl_t2, "kepid"]].drop_duplicates().rename(columns={pl_t2: "pl_name"})
            left = df.rename(columns={pl_df: "pl_name"}) if pl_df != "pl_name" else df
            out = left.merge(m, on="pl_name", how="left")
            if "kepid" in out.columns and out["kepid"].notna().any():
                return out

    # 3) Map via kepoi_name
    if ref_t2 is not None:
        koi_df = find_first(df, ["kepoi_name", "koi_name"])
        koi_t2 = find_first(ref_t2, ["kepoi_name", "koi_name"])
        if koi_df and koi_t2:
            m = ref_t2[[koi_t2, "kepid"]].drop_duplicates().rename(columns={koi_t2: "kepoi_name"})
            left = df.rename(columns={koi_df: "kepoi_name"}) if koi_df != "kepoi_name" else df
            out = left.merge(m, on="kepoi_name", how="left")
            if "kepid" in out.columns and out["kepid"].notna().any():
                return out

    return None

def std_keys(df, allow_missing=False, ref_for_mapping=None):
    if df is None:
        return None
    out = attach_kepid(df, ref_t2=ref_for_mapping)
    if out is None:
        if allow_missing:
            print("[WARN] No kepid-like key found; this dataset will be ignored and given neutral score.")
            return None
        raise ValueError("Expected a 'kepid' (or kic/kic_id/kicid, or a pl_name/kepoi_name map).")
    # standardise pl_name if present
    pl = find_first(out, ["pl_name", "planet_name"])
    if pl and pl != "pl_name":
        out = out.rename(columns={pl: "pl_name"})
    return out

# ====================== Component scorers ======================

def detection_component(t1):
    """
    Score in [0,1] from Task 1. Prioritise calibrated columns first.
    """
    if t1 is None or t1.empty:
        return pd.DataFrame(columns=["kepid", "detection_score"])

    det_priority = [
        # calibrated first
        "pred_prob_cal", "y_proba_cal", "transit_confidence_cal",
        # then raw
        "pred_prob", "y_proba", "transit_confidence", "detection_confidence",
        "cnn_score", "probability", "p_transit", "confidence",
    ]
    det_col = next((c for c in det_priority if c in t1.columns), None)

    if det_col is None:
        prob_like = [c for c in t1.columns if str(c).lower().endswith("_prob") or "conf" in str(c).lower()]
        det_col = prob_like[0] if prob_like else None

    t1 = t1.copy()
    if det_col is None:
        t1["detection_score"] = 0.5
    else:
        safe_float(t1, [det_col])
        t1["detection_score"] = minmax(t1[det_col])

    return t1[["kepid", "detection_score"]].dropna(subset=["kepid"])

def physical_component(t2):
    """
    Combine planet class (if any) and equilibrium temperature proximity to HZ into a physical score.
    """
    if t2 is None or t2.empty:
        return pd.DataFrame(columns=["kepid", "physical_score", "pl_name", "pl_rade", "pl_bmasse", "pl_eqt", "pl_dens"])

    t2 = t2.copy()

    # Planet class → numeric
    ptype_col = None
    for c in ["planet_type", "planet_class", "pl_class", "phys_class"]:
        if c in t2.columns:
            ptype_col = c
            break
    ptype_map = {
        "Earth-like": 1.0,
        "Super-Earth": 0.7,
        "Mini-Neptune": 0.4,
        "Neptune-like": 0.3,
        "Sub-Neptune": 0.4,
        "Jupiter-like": 0.1,
        "Gas Giant": 0.1,
    }

    def map_ptype(v):
        if pd.isna(v):
            return np.nan
        s = str(v).lower()
        for k, val in ptype_map.items():
            if k.lower() in s:
                return val
        return np.nan

    if ptype_col:
        t2["ptype_score"] = t2[ptype_col].apply(map_ptype)
    else:
        t2["ptype_score"] = np.nan

    # Temperature closeness 240–280K (taper to 0 at 180/310)
    eqt_col = pick_first_existing(t2, ["pl_eqt", "eqt", "equilibrium_temperature"])
    if eqt_col:
        safe_float(t2, [eqt_col])
        eqt = t2[eqt_col].astype(float)
        center_lo, center_hi = 240, 280
        min_k, max_k = 180, 310
        temp_score = np.where(eqt.between(center_lo, center_hi), 1.0,
                       np.where(eqt < center_lo, 1 - (center_lo - eqt) / (center_lo - min_k),
                       np.where(eqt > center_hi, 1 - (eqt - center_hi) / (max_k - center_hi), 0)))
        t2["temp_score"] = np.clip(temp_score, 0, 1)
    else:
        t2["temp_score"] = np.nan

    t2["physical_score"] = t2[["ptype_score", "temp_score"]].mean(axis=1, skipna=True)

    # Carry name + useful numeric columns
    pl_rade  = pick_first_existing(t2, ["pl_rade", "planet_radius_earth"])
    pl_bm    = pick_first_existing(t2, ["pl_bmasse", "planet_mass_earth"])
    pl_eqt   = eqt_col
    pl_dens  = pick_first_existing(t2, ["pl_dens", "density_from_mr", "density_from_m.r.(g/cc)", "density"])

    keep = ["kepid", "pl_name", "physical_score"]
    for c in [pl_rade, pl_bm, pl_eqt, pl_dens]:
        if c and c not in keep:
            keep.append(c)

    out = t2[keep].drop_duplicates(subset=["kepid"]).copy()
    # rename those columns to dashboard-friendly names
    ren = {}
    if pl_rade: ren[pl_rade] = "pl_rade"
    if pl_bm:   ren[pl_bm]   = "pl_bmasse"
    if pl_eqt:  ren[pl_eqt]  = "pl_eqt"
    if pl_dens: ren[pl_dens] = "pl_dens"
    out = out.rename(columns=ren)

    return out

def habitability_component(t3):
    """
    Map Task 3 labels to numeric score in [0,1].
    """
    if t3 is None or t3.empty:
        return pd.DataFrame(columns=["kepid", "habitability_score"])

    t3 = t3.copy()
    hab_col = pick_first_existing(t3, ["habitability_pred_rf", "habitability_label", "habitability"])

    def map_hab(v):
        if pd.isna(v):
            return np.nan
        s = str(v).lower()
        if "habitable" in s and "potential" in s:
            return 1.0
        if "habitable" in s and "partial" in s:
            return 0.6
        if "non" in s:
            return 0.0
        # numeric?
        try:
            f = float(v)
            return float(np.clip(f, 0, 1))
        except:
            return np.nan

    t3["habitability_score"] = t3[hab_col].apply(map_hab) if hab_col else np.nan
    return t3[["kepid", "habitability_score"]].dropna(subset=["kepid"])

def comms_component(t4):
    """
    Convert radio silence risk → comms score = 1 - risk.
    """
    if t4 is None or t4.empty:
        return pd.DataFrame(columns=["kepid", "comms_score", "radio_silence_risk"])

    t4 = t4.copy()
    risk_col = pick_first_existing(t4, ["radio_silence_risk", "risk", "radio_risk"])
    if risk_col is None and "Radio_Silent" in t4.columns:
        safe_float(t4, ["Radio_Silent"])
        t4["radio_silence_risk"] = np.where(t4["Radio_Silent"] == 1, 0.8, 0.2)
        risk_col = "radio_silence_risk"
    elif risk_col is None:
        t4["radio_silence_risk"] = 0.0
        risk_col = "radio_silence_risk"

    safe_float(t4, [risk_col])
    t4["comms_score"] = 1.0 - t4[risk_col].astype(float).clip(0, 1)
    return t4[["kepid", "comms_score", risk_col]].dropna(subset=["kepid"])

# =========================== Runner ===========================

def run_pipeline():
    print(f"[OUT] {RUN_DIR}")

    # --- Pick files ---
    t2_file = autopick("task2")
    t3_file = autopick("task3")
    t1_file = autopick("task1")
    t4_file = autopick("task4")

    # --- Load ---
    df_t2_raw = smart_read_csv(t2_file)
    df_t3_raw = smart_read_csv(t3_file)

    # Standardise keys for T2/T3 first (so T1/T4 can map via pl_name if needed)
    df_t2 = std_keys(df_t2_raw, allow_missing=False, ref_for_mapping=None) if df_t2_raw is not None else None
    df_t3 = std_keys(df_t3_raw, allow_missing=True, ref_for_mapping=df_t2) if df_t3_raw is not None else None

    # Now T1/T4 (allow missing and map via T2 where possible)
    df_t1 = std_keys(smart_read_csv(t1_file), allow_missing=True, ref_for_mapping=df_t2) if t1_file else None
    df_t4 = std_keys(smart_read_csv(t4_file), allow_missing=True, ref_for_mapping=df_t2) if t4_file else None

    # --- Component scores ---
    comp_det  = detection_component(df_t1) if df_t1 is not None else pd.DataFrame(columns=["kepid","detection_score"])
    comp_phys = physical_component(df_t2) if df_t2 is not None else pd.DataFrame(columns=["kepid","physical_score","pl_name","pl_rade","pl_bmasse","pl_eqt","pl_dens"])
    comp_hab  = habitability_component(df_t3) if df_t3 is not None else pd.DataFrame(columns=["kepid","habitability_score"])
    comp_com  = comms_component(df_t4) if df_t4 is not None else pd.DataFrame(columns=["kepid","comms_score","radio_silence_risk"])

    # --- Merge ---
    merged = None
    for d in [comp_det, comp_phys, comp_hab, comp_com]:
        merged = d.copy() if merged is None else pd.merge(merged, d, on="kepid", how="outer")

    # Fill missing components with column mean (or 0.5 if all NaN)
    for col in ["detection_score", "physical_score", "habitability_score", "comms_score"]:
        if col not in merged.columns:
            merged[col] = np.nan
        if merged[col].isna().all():
            merged[col] = 0.5
        else:
            merged[col] = merged[col].fillna(merged[col].mean())

    # Re-normalise to [0,1]
    for col in ["detection_score", "physical_score", "habitability_score", "comms_score"]:
        merged[col] = minmax(merged[col])

    # Normalise weights and compute composite
    wsum = sum(WEIGHTS.values())
    W = {k: v / wsum for k, v in WEIGHTS.items()}

    merged["target_priority"] = (
        W["detection"]   * merged["detection_score"] +
        W["habitability"]* merged["habitability_score"] +
        W["physical"]    * merged["physical_score"] +
        W["comms"]       * merged["comms_score"]
    )

    # Rank high→low
    merged = merged.sort_values("target_priority", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)

    # --- Dashboard-friendly table ---
    dash_cols = ["kepid", "pl_name", "target_priority",
                 "detection_score", "habitability_score", "physical_score", "comms_score",
                 "pl_rade", "pl_bmasse", "pl_eqt", "pl_dens", "rank"]
    dash_ren  = {
        "target_priority": "ProbeScore",
        "detection_score": "Score_Detection",
        "habitability_score": "Score_Habitability",
        "physical_score": "Score_Type",
        "comms_score": "Score_Radio",
    }
    dash_df = merged.copy()
    for c in dash_cols:
        if c not in dash_df.columns:
            dash_df[c] = np.nan
    dash_df = dash_df[dash_cols].rename(columns=dash_ren)

    # --- Save CSVs ---
    agg_path  = os.path.join(RUN_DIR, "task5_aggregated_targets.csv")
    top_path  = os.path.join(RUN_DIR, "task5_top_targets.csv")
    dash_path = os.path.join(RUN_DIR, "task5_dashboard_data.csv")
    score_path= os.path.join(RUN_DIR, "task5_probe_scores.csv")
    topk_path = os.path.join(RUN_DIR, "task5_probe_topk.csv")
    cfg_path  = os.path.join(RUN_DIR, "task5_config.json")

    merged.to_csv(agg_path, index=False)
    merged.head(TOP_N).to_csv(top_path, index=False)
    dash_df.to_csv(dash_path, index=False)

    # Smaller “scores only” table for analysis
    merged[["kepid", "detection_score", "habitability_score", "physical_score", "comms_score", "target_priority"]]\
        .rename(columns={"target_priority":"ProbeScore"}).to_csv(score_path, index=False)
    dash_df.head(TOP_N).to_csv(topk_path, index=False)

    # --- Figures ---
    plt.figure()
    merged["target_priority"].hist(bins=40)
    plt.title("Target Priority Score Distribution")
    plt.xlabel("target_priority"); plt.ylabel("count")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "score_hist.png")); plt.close()

    plt.figure()
    plt.scatter(merged["detection_score"], merged["habitability_score"],
                s=8, alpha=0.6, c=merged["target_priority"])
    plt.title("Detection vs Habitability (colored by Priority)")
    plt.xlabel("detection_score"); plt.ylabel("habitability_score")
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "pair_scatter.png")); plt.close()

    # --- Config dump ---
    cfg = {
        "weights": W,
        "top_n": TOP_N,
        "inputs": {"task1": t1_file, "task2": t2_file, "task3": t3_file, "task4": t4_file},
        "run_dir": RUN_DIR,
        "columns_used": {
            "dash": list(dash_df.columns),
            "scores": ["detection_score", "habitability_score", "physical_score", "comms_score"],
        }
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print("[DONE]")
    print(f"  Aggregated : {agg_path}")
    print(f"  Top-{TOP_N} : {top_path}")
    print(f"  Dashboard  : {dash_path}")
    print(f"  Scores     : {score_path}")
    print(f"  TopK       : {topk_path}")
    print(f"  Figures    : {FIG_DIR}")
    print(f"  Config     : {cfg_path}")

# --------------------- Entry point ---------------------

if __name__ == "__main__":
    run_pipeline()
