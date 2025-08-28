# -*- coding: utf-8 -*-
"""
Task 5: Space Probe Decision (zero-arg, T1 scoped to multi_run folder)
- T2 & T3 pinned exact files
- T1 auto-picks ONLY from the given multi_run path
- T4 auto-picks from Output/Task4
- Fixed TypeError in score_planet_type (and safer fallback in habitability)
"""

import os, glob, json
from datetime import datetime
import pandas as pd
import numpy as np

# ---------------- User-specific constants ----------------
OUTPUT_ROOT = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output"

# Pin Task 2 & Task 3 EXACT files
DEFAULT_T2_FILE = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task2/run_20250826_104658/task2_characterization_with_kepid_kepid_mapped.csv"
DEFAULT_T3_FILE = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task3/run_20250826_125617/csv/task3_habitability_with_predictions__with_kepid.csv"

# Restrict Task 1 auto-pick to THIS folder
T1_SEARCH_ROOT = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task1/multi_run_20250825_094156"

# Weights and options
W_DET, W_HAB, W_TYP, W_RAD = 0.35, 0.35, 0.15, 0.15
TOPK = 50
DEDUP = "max"   # "none" | "mean" | "max"

# ---------------- Helpers ----------------
def ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def smart_read_csv(p): print(f"[LOAD] {p}"); return pd.read_csv(p, low_memory=False)

def norm_name(col):
    return str(col).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

def find_first(df, candidates):
    nm = {norm_name(c): c for c in df.columns}
    for cand in candidates:
        k = norm_name(cand)
        if k in nm: return nm[k]
    for c in df.columns:
        if any(norm_name(x) in norm_name(c) for x in candidates):
            return c
    return None

def to_num(s): return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
def minmax01(s):
    s = to_num(s)
    if s.notna().sum() == 0: return pd.Series(np.nan, index=s.index)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        out = pd.Series(0.5, index=s.index); out[s.isna()] = np.nan; return out
    return (s - lo) / (hi - lo)
def clip01(x): return np.clip(x, 0.0, 1.0)

def latest_file(search_dir, patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(search_dir, "**", pat), recursive=True))
    if not files: return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

# ---------------- Auto-pickers ----------------
def autopick_task1():
    """
    Search ONLY inside T1_SEARCH_ROOT for likely Task1 outputs.
    """
    base = T1_SEARCH_ROOT
    patterns = [
        "task1_transit_predictions.csv",
        "*task1*transit*pred*.csv",
        "*transit*pred*.csv",
        "*task1*pred*.csv",
        # add score-style names (common in multi_run outputs)
        "*task1*transit*scor*.csv",
        "*transit*scor*.csv",
        "*task1*scor*.csv",
        "*predictions*.csv",
        "*scores*.csv",
    ]
    f = latest_file(base, patterns) if base and os.path.isdir(base) else None
    print(f"[AUTO] Task1 file: {f}" if f else "[AUTO] Task1 not found in scoped folder -> neutral 0.5")
    return f

def autopick_task4():
    base = os.path.join(OUTPUT_ROOT, "Task4")
    patterns = [
        "task4_radio_silence_predictions.csv",
        "*task4*radio*silence*pred*.csv",
        "*radio*silence*pred*.csv",
        "*task4*radio*.csv",
        # accept score-style names
        "*radio*silence*scor*.csv",
        "*task4*scor*.csv",
        "*scores*.csv",
    ]
    f = latest_file(base, patterns)
    print(f"[AUTO] Task4 file: {f}" if f else "[AUTO] Task4 not found -> neutral 0.5")
    return f

# ---------------- Scorers ----------------
def score_detection(df):
    if df is None or len(df) == 0: return None, {"fallback": 0.5}
    conf = find_first(df, ["det_conf","detection_confidence","pred_proba","prob","confidence","score","transit_prob","transit_confidence"])
    if conf: return minmax01(df[conf]), {"source_conf": conf}
    bcol = find_first(df, ["prediction","pred","label","transit_detected","has_transit"])
    if bcol: return clip01(to_num(df[bcol]).fillna(0.0)), {"source_bin": bcol}
    return pd.Series(0.5, index=df.index), {"fallback": 0.5}

def score_habitability(df):
    if df is None or len(df) == 0: return None, {"fallback": 0.5}
    label = find_first(df, ["habitability_label","hab_label","label_habitability","habitabilityclass","habitability"])
    pred  = find_first(df, ["habitability_pred_rf","hab_pred","pred_habitability","pred_rf"])
    mapping = {
        "habitable (potentially)": 1.0, "habitable(potentially)": 1.0, "habitable": 1.0,
        "partially habitable": 0.6, "partial": 0.6,
        "non-habitable": 0.0, "nonhabitable": 0.0, "not habitable": 0.0
    }
    if label:
        lbl = df[label].astype(str).str.lower().str.strip()
        out = lbl.map(mapping)
        # build a Series fallback (no ndarray inside fillna)
        fb = pd.Series(0.5, index=lbl.index)
        fb[lbl.str.contains("partial", na=False)] = 0.6
        fb[lbl.str.contains("non", na=False)] = 0.0
        fb[lbl.str.contains("habit", na=False)] = 1.0
        out = out.fillna(fb).fillna(0.5)
        return out.astype(float), {"label_col": label}
    if pred:
        p = to_num(df[pred])
        uniq = sorted([u for u in p.dropna().unique().tolist()])
        if set(uniq).issubset({0,1,2}):
            s = p.map({0:0.0, 1:0.6, 2:1.0}).fillna(0.5).astype(float)
            return s, {"pred_col": pred, "mapping": "0->0.0, 1->0.6, 2->1.0"}
        return minmax01(p).fillna(0.5).astype(float), {"pred_col": pred, "mapping": "minmax"}
    return pd.Series(0.5, index=df.index), {"fallback": 0.5}

def score_planet_type(df):
    if df is None or len(df) == 0: return None, {"fallback": 0.5}
    t = find_first(df, ["planet_type","type","predicted_type","class","category"])
    if not t: return pd.Series(0.5, index=df.index), {"fallback": 0.5}
    s = df[t].astype(str).str.lower().str.strip()
    base_map = {
        "earth-like": 1.0, "earthlike": 1.0, "terrestrial": 1.0,
        "super-earth": 0.8, "superearth": 0.8,
        "mini-neptune": 0.6, "sub-neptune": 0.6, "neptune-like": 0.5, "neptune": 0.5,
        "gas giant": 0.3, "jupiter-like": 0.3, "jupiter": 0.3, "unknown": 0.5
    }
    out = s.map(base_map)
    # build a Series fallback (avoid ndarray in fillna)
    fb = pd.Series(0.5, index=s.index)
    fb[s.str.contains("earth", na=False)] = 1.0
    fb[s.str.contains("super", na=False)] = 0.8
    fb[s.str.contains("mini", na=False) | s.str.contains("sub", na=False)] = 0.6
    fb[s.str.contains("neptune", na=False)] = 0.5
    fb[s.str.contains("jupiter", na=False) | s.str.contains("giant", na=False)] = 0.3
    out = out.fillna(fb).fillna(0.5)
    return out.astype(float), {"type_col": t}

def score_radio_visibility(df):
    if df is None or len(df) == 0: return None, {"fallback": 0.5}
    pcol = find_first(df, ["proba_silent","prob_silent","p_silent","silent_prob"])
    bcol = find_first(df, ["radio_silent","is_silent","silent"])
    if pcol:
        p = clip01(to_num(df[pcol]).fillna(0.5))
        return (1.0 - p), {"prob_col": pcol}
    if bcol:
        f = clip01(to_num(df[bcol]).fillna(1.0))
        return (1.0 - f), {"bin_col": bcol}
    return pd.Series(0.5, index=df.index), {"fallback": 0.5}

# ---------------- Pipeline ----------------
def std_keys(df):
    if df is None: return None
    df = df.copy()
    kep = find_first(df, ["kepid","kepler_id","kepid_num"])
    if not kep: raise ValueError("Expected a 'kepid' column (or equivalent) in input.")
    if kep != "kepid": df = df.rename(columns={kep: "kepid"})
    df["kepid"] = pd.to_numeric(df["kepid"], errors="coerce")
    pln = find_first(df, ["pl_name","planet_name","koi_name","kepoi_name"])
    if pln and pln != "pl_name": df = df.rename(columns={pln: "pl_name"})
    return df

def run_pipeline():
    out_root = ensure_dir(os.path.join(OUTPUT_ROOT, "Task5", f"run_{ts()}"))
    print(f"[OUT] {out_root}")

    # Load T2/T3 (pinned)
    df_t2 = std_keys(smart_read_csv(DEFAULT_T2_FILE))
    df_t3 = std_keys(smart_read_csv(DEFAULT_T3_FILE))

    # Auto-pick T1 (scoped) & T4 (global)
    t1_file = autopick_task1()
    t4_file = autopick_task4()
    df_t1 = std_keys(smart_read_csv(t1_file)) if t1_file else None
    df_t4 = std_keys(smart_read_csv(t4_file)) if t4_file else None

    # Scores
    typ_s, meta_typ = score_planet_type(df_t2)
    hab_s, meta_hab = score_habitability(df_t3)
    det_s, meta_det = score_detection(df_t1) if df_t1 is not None else (None, {"auto":True,"fallback":0.5})
    rad_s, meta_rad = score_radio_visibility(df_t4) if df_t4 is not None else (None, {"auto":True,"fallback":0.5})

    # Base (planet-level from T2)
    base = df_t2[["kepid"] + (["pl_name"] if "pl_name" in df_t2.columns else [])].drop_duplicates().copy()

    # Attach component scores
    df_typ = df_t2[["kepid"] + (["pl_name"] if "pl_name" in df_t2.columns else [])].copy(); df_typ["Score_Type"] = typ_s.fillna(0.5)
    df_hab = df_t3[["kepid"] + (["pl_name"] if "pl_name" in df_t3.columns else [])].copy(); df_hab["Score_Habitability"] = hab_s.fillna(0.5)
    if df_t1 is not None:
        df_det = df_t1[["kepid"]].copy(); df_det["Score_Detection"] = det_s.fillna(0.5); df_det = df_det.groupby("kepid", as_index=False)["Score_Detection"].mean()
    else:
        df_det = pd.DataFrame({"kepid":[], "Score_Detection":[]})
    if df_t4 is not None:
        df_rad = df_t4[["kepid"]].copy(); df_rad["Score_Radio"] = rad_s.fillna(0.5); df_rad = df_rad.groupby("kepid", as_index=False)["Score_Radio"].mean()
    else:
        df_rad = pd.DataFrame({"kepid":[], "Score_Radio":[]})

    # Merge
    on_pl = "pl_name" in base.columns and "pl_name" in df_hab.columns and "pl_name" in df_typ.columns
    keys = ["kepid"] + (["pl_name"] if on_pl else [])
    merged = base.merge(df_hab, on=keys, how="left")\
                 .merge(df_typ, on=keys, how="left")\
                 .merge(df_det, on=["kepid"], how="left")\
                 .merge(df_rad, on=["kepid"], how="left")

    # Fill missing with neutral
    for c in ["Score_Detection","Score_Habitability","Score_Type","Score_Radio"]:
        if c not in merged.columns: merged[c] = 0.5
        merged[c] = merged[c].fillna(0.5)

    # Weights
    w = np.array([W_DET, W_HAB, W_TYP, W_RAD], dtype=float)
    if not np.isfinite(w).all() or w.sum() <= 0: w = np.array([0.35,0.35,0.15,0.15])
    w = w / w.sum()

    merged["ProbeScore"] = (
        w[0]*merged["Score_Detection"] +
        w[1]*merged["Score_Habitability"] +
        w[2]*merged["Score_Type"] +
        w[3]*merged["Score_Radio"]
    )

    # Dedup options
    if DEDUP.lower() == "max":
        merged = merged.sort_values(["kepid","ProbeScore"], ascending=[True, False])\
                       .drop_duplicates(subset=["kepid"], keep="first")
    elif DEDUP.lower() == "mean":
        agg = merged.groupby("kepid", as_index=False).agg({
            "ProbeScore":"mean",
            "Score_Detection":"mean",
            "Score_Habitability":"mean",
            "Score_Type":"mean",
            "Score_Radio":"mean",
        })
        if "pl_name" in merged.columns:
            topname = merged.sort_values("ProbeScore", ascending=False)\
                            .drop_duplicates("kepid")[["kepid","pl_name"]]
            merged = agg.merge(topname, on="kepid", how="left")
        else:
            merged = agg

    # Rank
    merged = merged.sort_values("ProbeScore", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged)+1)

    # Bring a few useful T2 cols if present
    useful = ["pl_rade","pl_bmasse","pl_eqt","pl_dens","pl_orbper","st_teff","st_rad","st_mass","st_logg",
              "planet_type","type","predicted_type","class","category"]
    for u in useful:
        col = find_first(df_t2, [u])
        if col and col not in merged.columns:
            add_keys = ["kepid"] + (["pl_name"] if on_pl else [])
            merged = merged.merge(df_t2[add_keys + [col]].drop_duplicates(), on=add_keys, how="left")

    # Save
    out_dir = out_root
    scores_fp = os.path.join(out_dir, "task5_probe_scores.csv")
    top_fp    = os.path.join(out_dir, "task5_probe_topk.csv")
    dash_fp   = os.path.join(out_dir, "task5_dashboard_data.csv")

    merged.to_csv(scores_fp, index=False)
    merged.head(int(TOPK)).to_csv(top_fp, index=False)

    dash_cols = [c for c in ["kepid","pl_name","ProbeScore","Score_Detection","Score_Habitability","Score_Type","Score_Radio",
                             "pl_rade","pl_bmasse","pl_eqt","pl_dens","pl_orbper","st_teff","st_rad","st_mass","rank"] if c in merged.columns]
    merged[dash_cols].to_csv(dash_fp, index=False)

    cfg = {
        "t2_file": DEFAULT_T2_FILE,
        "t3_file": DEFAULT_T3_FILE,
        "t1_file_used": t1_file,
        "t4_file_used": t4_file,
        "weights": {"w_det": float(w[0]), "w_hab": float(w[1]), "w_typ": float(w[2]), "w_rad": float(w[3])},
        "topk": int(TOPK), "dedup": DEDUP,
        "meta": {"type": meta_typ, "habitability": meta_hab, "detection": meta_det, "radio": meta_rad}
    }
    with open(os.path.join(out_dir, "task5_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print(f"[SAVE] {scores_fp}")
    print(f"[SAVE] {top_fp}")
    print(f"[SAVE] {dash_fp}")
    print("[DONE] Probe scoring complete.")

# Entry point (no argparse)
if __name__ == "__main__":
    out_root = ensure_dir(os.path.join(OUTPUT_ROOT, "Task5", f"run_{ts()}"))
    run_pipeline()

