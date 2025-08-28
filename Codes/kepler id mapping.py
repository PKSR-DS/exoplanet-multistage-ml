# -*- coding: utf-8 -*-
"""
Add Kepler ID (kepid) to Task 3 files using KOI Cumulative CSV (NO retraining).
- Uses kepid + (kepler_name / kepoi_name / koi_name) from KOI CSV
- Matches to Task 3 by planet name (pl_name / planet_name / kepler_name)
- Writes <name>__with_kepid.csv next to each source file + an unmatched list.

Edit the paths below and run.
"""

import os
import pandas as pd
from typing import Optional, Tuple

# ============================
# CONFIG — EDIT PATHS
# ============================
KOI_CSV = r"D:/UH One drive/cumulative_2025.07.16_10.28.33.csv"

TARGETS = [
    r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task3\run_20250826_125617\csv\task3_habitability_with_predictions.csv"
]

# Choose where to save outputs:
OUTPUT_MODE = "sibling"   # "sibling" to save next to each source, or "folder" to save all into OUTPUT_DIR
OUTPUT_DIR  = r"D:/UH One drive/OneDrive - University of Hertfordshire/Output/Task3/kepid_mapped"

# ============================
# HELPERS
# ============================
def _norm_name(x: str) -> str:
    """Normalize planet/KOI names for matching: lowercase, keep only [a-z0-9]."""
    if not isinstance(x, str):
        return ""
    return "".join(ch.lower() for ch in x if ch.isalnum())

def _pick_first(df: pd.DataFrame, candidates) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _standardize_koi_map(df_koi: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Build a mapping DF with columns: ["__norm_key", "kepid", "name_src"].
    Tries name columns in order: kepler_name, kepoi_name, koi_name, pl_name, planet_name.
    """
    # kepid column in KOI cumulative is usually 'kepid'
    kepid_col = _pick_first(df_koi, ["kepid", "KepID"])
    if kepid_col is None:
        raise SystemExit("[FATAL] KOI CSV must have a 'kepid' column.")

    # name columns to try (KOI has kepler_name for confirmed; kepoi_name/koi_name for candidates)
    name_col = _pick_first(df_koi, ["kepler_name", "kepoi_name", "koi_name", "pl_name", "planet_name"])
    if name_col is None:
        raise SystemExit("[FATAL] KOI CSV must have a name column "
                         "(kepler_name / kepoi_name / koi_name / pl_name / planet_name).")

    out = df_koi.copy()
    out["__norm_key"] = out[name_col].map(_norm_name)
    # Keep only necessary fields; drop duplicates on normalized name (first wins)
    out = out[[name_col, kepid_col, "__norm_key"]].dropna(subset=["__norm_key"]).drop_duplicates("__norm_key")
    out.rename(columns={kepid_col: "kepid", name_col: "name_src"}, inplace=True)
    return out, "__norm_key"

def _make_out_path(src: str, suffix: str) -> str:
    root, ext = os.path.splitext(os.path.basename(src))
    if OUTPUT_MODE == "folder":
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return os.path.join(OUTPUT_DIR, root + suffix)
    return os.path.join(os.path.dirname(src), root + suffix)

def add_kepid_with_koi(target_csv: str, koi_map_df: pd.DataFrame) -> None:
    if not os.path.exists(target_csv):
        print(f"[WARN] Not found: {target_csv}")
        return

    df = pd.read_csv(target_csv)

    # Find the planet name column in the Task 3 file
    name_col = _pick_first(df, ["pl_name", "planet_name", "kepler_name", "pl_name_clean"])
    if name_col is None:
        print(f"[WARN] {os.path.basename(target_csv)}: no planet-name column; skipping.")
        return

    # Merge by normalized name
    df["__norm_key"] = df[name_col].map(_norm_name)
    merged = df.merge(koi_map_df[["__norm_key", "kepid"]], on="__norm_key", how="left")
    merged.drop(columns=["__norm_key"], inplace=True)

    # If target already has kepid, fill only missing values from KOI
    if "kepid_x" in merged.columns and "kepid_y" in merged.columns:
        merged["kepid"] = merged["kepid_x"].where(merged["kepid_x"].notna(), merged["kepid_y"])
        merged.drop(columns=["kepid_x", "kepid_y"], inplace=True)

    # Save outputs + summary
    total = len(merged)
    mapped = int(merged["kepid"].notna().sum()) if "kepid" in merged.columns else 0
    print(f"[INFO] {os.path.basename(target_csv)} → mapped {mapped}/{total} rows ({(mapped/total*100 if total else 0):.1f}%).")

    out_csv = _make_out_path(target_csv, "__with_kepid.csv")
    merged.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")

    # Unmatched names (to fix manually if needed)
    if "kepid" in merged.columns and merged["kepid"].isna().any():
        um = merged.loc[merged["kepid"].isna(), [name_col]].drop_duplicates()
        um_path = _make_out_path(target_csv, "__unmatched_pl_names.csv")
        um.to_csv(um_path, index=False)
        print(f"[INFO] Unmatched names written to: {um_path}")

def main():
    if OUTPUT_MODE == "folder":
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(KOI_CSV):
        raise SystemExit(f"[FATAL] KOI CSV not found:\n{KOI_CSV}")

    koi = pd.read_csv(KOI_CSV)
    koi_map, key = _standardize_koi_map(koi)

    for t in TARGETS:
        try:
            add_kepid_with_koi(t, koi_map)
        except Exception as e:
            print(f"[WARN] Failed on {t}: {e}")

if __name__ == "__main__":
    main()
