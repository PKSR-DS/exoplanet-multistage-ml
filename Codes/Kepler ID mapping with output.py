import os, re
import pandas as pd
import numpy as np

# --- paths (yours) ---
t2_path  = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task2\run_20250826_104658\task2_characterization_with_kepid.csv"
koi_path = r"D:\UH One drive\cumulative_2025.07.16_10.28.33.csv"

# --- helpers ---
def nk(s: str) -> str: return re.sub(r"[^a-z0-9]", "", s.lower())
def normkey_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)

def find_col(df, *cands):
    keys = {nk(c): c for c in df.columns}
    for cand in cands:
        k = nk(cand)
        if k in keys: return keys[k]
        for kk, orig in keys.items():
            if k in kk: return orig
    return None

# --- load files ---
t2  = pd.read_csv(t2_path, low_memory=True)
t2  = t2.loc[:, ~t2.columns.str.contains('^Unnamed', case=False)]

koi = pd.read_csv(koi_path, low_memory=True)
koi = koi.loc[:, ~koi.columns.str.contains('^Unnamed', case=False)]

# --- pick the planet-name column in your Task-2 file (use what's already there) ---
name_col = find_col(t2, "planet_name","pl_name","kepler_name")
if name_col is None:
    raise ValueError("Couldn't find a planet name column (expected planet_name/pl_name/kepler_name).")

# --- build mapping from KOI: kepler_name/kepoi_name -> kepid ---
kepid_col = find_col(koi, "kepid","kic","kic_id")
kname_col = find_col(koi, "kepler_name","pl_name")
koi_col   = find_col(koi, "kepoi_name","koi_name","koi")
if kepid_col is None:
    raise ValueError("KOI file is missing KepID/KIC column.")

pieces = []
if kname_col is not None:
    tmp = koi[[kname_col, kepid_col]].dropna().rename(columns={kname_col:"planet_label", kepid_col:"kepid"})
    tmp["pl_key"] = normkey_series(tmp["planet_label"])
    pieces.append(tmp[["pl_key","kepid"]])
if koi_col is not None:
    tmp = koi[[koi_col, kepid_col]].dropna().rename(columns={koi_col:"planet_label", kepid_col:"kepid"})
    tmp["pl_key"] = normkey_series(tmp["planet_label"])
    pieces.append(tmp[["pl_key","kepid"]])

cross = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["pl_key","kepid"])
cross["kepid"] = pd.to_numeric(cross["kepid"], errors="coerce").astype("Int64")
cross = cross.dropna(subset=["pl_key","kepid"]).drop_duplicates("pl_key")

# --- attach kepid to your Task-2 rows (no other columns changed) ---
t2["pl_key"] = normkey_series(t2[name_col])
out = t2.merge(cross, on="pl_key", how="left", suffixes=("", "_map"))

# if a kepid column already exists, only fill missing values from the map
if "kepid" in t2.columns:
    keep = pd.to_numeric(out["kepid"], errors="coerce").astype("Int64")
    fill = pd.to_numeric(out["kepid_map"], errors="coerce").astype("Int64") if "kepid_map" in out.columns else pd.Series(pd.NA, dtype="Int64", index=out.index)
    out["kepid"] = keep.fillna(fill)
    out.drop(columns=[c for c in ["kepid_map"] if c in out.columns], inplace=True)
else:
    out["kepid"] = pd.to_numeric(out.get("kepid", out.get("kepid_map", pd.Series(pd.NA, index=out.index))), errors="coerce").astype("Int64")
    out.drop(columns=[c for c in ["kepid_map"] if c in out.columns], inplace=True)

out.drop(columns=["pl_key"], inplace=True)

# --- save next to input (won't overwrite your original) ---
root, ext = os.path.splitext(t2_path)
out_path  = root + "_kepid_mapped.csv"
out.to_csv(out_path, index=False)

print(f"[OK] Wrote: {out_path}")
mapped = int(out["kepid"].notna().sum()) if "kepid" in out.columns else 0
print(f"[INFO] Rows: {len(out)} | rows with kepid: {mapped} | unique kepid: {out['kepid'].nunique(dropna=True)}")

