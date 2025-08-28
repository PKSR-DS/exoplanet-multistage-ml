# streamlit_app_mission_control_pro_v3.py
# Mission Control PRO — v3.5
# - Prevents MemoryError by disabling inline HTML embeds by default.
# - Optional HTML embed with size guard (user toggle + MB cap).
# - Keeps: 3D Plots tab removed; Highlights for Tasks 1–5; cleaned molecule parsing;
#          duplicate-safe joins; safe sliders; robust Sky View.

import os, glob, re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Optional deps ----------
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

try:
    import pydeck as pdk
    HAS_PYDECK = True
except Exception:
    HAS_PYDECK = False

# ---------- Page ----------
st.set_page_config(page_title="Mission Control PRO — Exoplanets (v3.5)", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
      .muted { color:#8a8a8a; font-size:0.9rem }
      .hero { padding:18px 22px; border-radius:18px; background:linear-gradient(135deg,#0b1220,#111a2e);
              border:1px solid #1f2a44; box-shadow:0 0 24px rgba(0,0,0,.45); }
      .hero h1{ margin:0; font-size:1.9rem } .hero p{ margin-top:6px; color:#b8c7ff }
      .kcard { padding:14px 16px; border-radius:14px; border:1px solid #222a3a; background:#0f1625;
               box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02); }
      .pill { padding:2px 8px; background:#1e293b; border-radius:999px; font-size:.8rem }
      .foot { color:#6b7280; font-size:.85rem }
      .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utils ----------
def ts(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def norm(s: str) -> str:
    s = str(s).lower()
    return re.sub(r'[^a-z0-9]+', '', s)

@st.cache_data(show_spinner=False)
def read_csv_smart(path: str):
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None

def safe_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = [c.strip() for c in df.columns]
    return df

def find_first(df: pd.DataFrame, candidates):
    if df is None:
        return None
    lut = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lut:
            return lut[norm(c)]
    for c in df.columns:
        for cand in candidates:
            if norm(cand) in norm(c):
                return c
    return None

def latest_file(search_dir: str, patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(search_dir, "**", pat), recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def ensure_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)

def minmax01(s: pd.Series) -> pd.Series:
    s = ensure_numeric(s)
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        out = pd.Series(0.5, index=s.index); out[s.isna()] = np.nan; return out
    return (s - lo) / (hi - lo)

def clip01(x: pd.Series) -> pd.Series:
    return np.clip(ensure_numeric(x), 0.0, 1.0)

def safe_slider(label: str, total: int, *, default: int = 27, cap: int = 300, key: str | None = None) -> int:
    if total <= 1:
        st.caption(f"Showing {total} item(s).")
        return max(total, 0)
    max_value = min(cap, total)
    default_value = min(default, max_value)
    return st.slider(label, 1, max_value, default_value, 1, key=key)

# ---------- Defaults (your paths) ----------
DEFAULT_OUTPUT_ROOT = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output"
DEFAULT_T1_ROOT     = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task1"
DEFAULT_T1_SEARCH_ROOT = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task1\multi_run_20250825_094156"
DEFAULT_T2_FILE     = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task2\run_20250826_104658\task2_characterization_with_kepid_kepid_mapped.csv"
DEFAULT_T3_FILE     = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task3\run_20250826_125617\csv\task3_habitability_with_predictions__with_kepid.csv"
DEFAULT_T4_ROOT     = os.path.join(DEFAULT_OUTPUT_ROOT, "Task4")
DEFAULT_T5_ROOT     = os.path.join(DEFAULT_OUTPUT_ROOT, "Task5")

DEFAULT_EDA_PATH    = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\EDA"

DEFAULT_DATA_ROOT   = r"D:\UH One drive\OneDrive - University of Hertfordshire\Final Project\Data\20_08_25"
DEFAULT_PLANET_FILE = os.path.join(DEFAULT_DATA_ROOT, "Planetary Systems Composite Data.csv")
DEFAULT_ATMOS_FILE  = os.path.join(DEFAULT_DATA_ROOT, "cleaned_exoplanet_atmospheres_with_molecules_cleaned.csv")
DEFAULT_STAR_FILE   = os.path.join(DEFAULT_DATA_ROOT, "Stellar Hosts.csv")

# ---------- Hero ----------
st.markdown(
    f"""
<div class="hero">
  <h1>🛰️ Mission Control PRO (v3.5)</h1>
  <p>HTML embeds are off by default to avoid MemoryError. Enable below if needed.</p>
  <div class="muted">Build: {ts()}</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.title("📚 Data Catalog")
t1_dir = st.sidebar.text_input("Task 1 root", DEFAULT_T1_ROOT)
t1_search = st.sidebar.text_input("Task 1 multi-run (auto-pick)", DEFAULT_T1_SEARCH_ROOT)
t1_file_override = st.sidebar.text_input("Task 1 file override (CSV)", "")

t2_file = st.sidebar.text_input("Task 2 mapped (with KEPID)", DEFAULT_T2_FILE)
t3_file = st.sidebar.text_input("Task 3 mapped (with KEPID)", DEFAULT_T3_FILE)
t4_root = st.sidebar.text_input("Task 4 root", DEFAULT_T4_ROOT)
t5_root = st.sidebar.text_input("Task 5 root (optional)", DEFAULT_T5_ROOT)

st.sidebar.markdown("---")
eda_root = st.sidebar.text_input("EDA folder", DEFAULT_EDA_PATH)
EMBED_HTML = st.sidebar.toggle("Embed HTML files inside app (may be heavy)", value=False)
MAX_HTML_MB = st.sidebar.slider("Max HTML size to embed (MB)", 1, 20, 5) if EMBED_HTML else 5

st.sidebar.markdown("---")
data_root = st.sidebar.text_input("Science data root", DEFAULT_DATA_ROOT)
planet_file = st.sidebar.text_input("Planetary systems CSV", DEFAULT_PLANET_FILE)
atmos_file  = st.sidebar.text_input("Atmospheres + molecules CSV", DEFAULT_ATMOS_FILE)
star_file   = st.sidebar.text_input("Stellar hosts CSV", DEFAULT_STAR_FILE)

st.sidebar.markdown("---")
with st.sidebar.expander("🎛️ Weights & Aggregation"):
    W_DET = st.slider("Detection (Task 1)", 0.0, 1.0, 0.35, 0.05)
    W_HAB = st.slider("Habitability (Task 3)", 0.0, 1.0, 0.35, 0.05)
    W_TYP = st.slider("Planet Type (Task 2)", 0.0, 1.0, 0.15, 0.05)
    W_RAD = st.slider("Radio Visibility (Task 4)", 0.0, 1.0, 0.15, 0.05)
    DEDUP = st.selectbox("Duplicate KEPID aggregation", ["max", "mean"], index=0)
    st.caption("Weights are normalized if they don't sum to 1.")

# ---------- Catalog discovery ----------
@st.cache_data(show_spinner=False)
def discover_catalog(t1_root, t1_search_root, t4_root, t5_root, t1_override):
    def pick(root, pats):
        if not os.path.isdir(root):
            return None
        return latest_file(root, pats)
    if t1_override and os.path.isfile(t1_override):
        t1_path = t1_override
    else:
        t1_pats = [
            "*task1*transit*pred*.csv", "*transit*pred*.csv", "*task1*pred*.csv",
            "*task1*transit*scor*.csv", "*transit*scor*.csv", "*task1*scor*.csv",
            "*predictions*.csv", "*scores*.csv", "*results*.csv", "*inference*.csv",
            "*task1*.csv", "*cnn*pred*.csv", "*multimodel*pred*.csv"
        ]
        t1_path = pick(t1_search_root, t1_pats) or pick(t1_root, t1_pats)
    return {
        "task1": t1_path,
        "task4": pick(t4_root, ["task4_radio_silence_predictions.csv", "*radio*silence*pred*.csv", "*radio*.csv", "*scores*.csv"]),
        "task5": pick(t5_root, ["*probe*rank*.csv", "*final*ranking*.csv", "*probe*score*.csv"]),
    }

catalog = discover_catalog(t1_dir, t1_search, t4_root, t5_root, t1_file_override)

# ---------- Load outputs ----------
t1_path = catalog["task1"]
t4_path = catalog["task4"]
t5_path = catalog["task5"]

t2_df = safe_df(read_csv_smart(t2_file) if os.path.isfile(t2_file) else None)
t3_df = safe_df(read_csv_smart(t3_file) if os.path.isfile(t3_file) else None)
t1_df = safe_df(read_csv_smart(t1_path) if t1_path else None)
t4_df = safe_df(read_csv_smart(t4_path) if t4_path else None)
t5_df = safe_df(read_csv_smart(t5_path) if t5_path else None)

# ---------- Status cards ----------
c1, c2, c3, c4 = st.columns(4)
def status_card(col, title, present, path_or_rows):
    with col:
        st.markdown(
            f"<div class='kcard'><div class='pill'>{'Connected' if present else 'Missing'}</div>"
            f"<h4>{title}</h4><div class='muted mono'>{path_or_rows}</div></div>",
            unsafe_allow_html=True,
        )
status_card(c1, "Task 1 — Detection", t1_df is not None, (t1_path or "—"))
status_card(c2, "Task 2 — Characterization (mapped)", t2_df is not None, f"rows: {len(t2_df):,}" if t2_df is not None else "—")
status_card(c3, "Task 3 — Habitability (mapped)", t3_df is not None, f"rows: {len(t3_df):,}" if t3_df is not None else "—")
status_card(c4, "Task 4 — Radio", t4_df is not None, (t4_path or "—"))

# ---------- Scorers (same as v3.4) ----------
def sc_detection(df):
    if df is None or len(df) == 0: return None
    conf = find_first(df, ["det_conf","detection_confidence","pred_proba","prob","confidence","score","transit_prob","transit_confidence"])
    if conf: return minmax01(df[conf])
    bcol = find_first(df, ["prediction","pred","label","transit_detected","has_transit","is_transit"])
    if bcol: return clip01(df[bcol]).fillna(0.0)
    return pd.Series(0.5, index=df.index)

def sc_habitability(df):
    if df is None or len(df) == 0: return None
    label = find_first(df, ["habitability_label","hab_label","label_habitability","habitabilityclass","habitability"])
    pred  = find_first(df, ["habitability_pred_rf","hab_pred","pred_habitability","pred_rf","habitability_pred"])
    mapping = {"habitable (potentially)":1.0,"habitable(potentially)":1.0,"habitable":1.0,
               "partially habitable":0.6,"partial":0.6,"non-habitable":0.0,"nonhabitable":0.0,"not habitable":0.0}
    if label:
        lbl = df[label].astype(str).str.lower().str.strip()
        out = lbl.map(mapping)
        fb = pd.Series(0.5, index=lbl.index)
        fb[lbl.str.contains("partial", na=False)] = 0.6
        fb[lbl.str.contains("non", na=False)] = 0.0
        fb[lbl.str.contains("habit", na=False)] = 1.0
        out = out.fillna(fb).fillna(0.5)
        return out.astype(float)
    if pred:
        p = ensure_numeric(df[pred])
        uniq = sorted([u for u in p.dropna().unique().tolist() if pd.notna(u)])
        if set(uniq).issubset({0,1,2}):
            return p.map({0:0.0,1:0.6,2:1.0}).fillna(0.5).astype(float)
        return minmax01(p).fillna(0.5).astype(float)
    return pd.Series(0.5, index=df.index)

def sc_planet_type(df):
    if df is None or len(df) == 0: return None
    t = find_first(df, ["planet_type","type","predicted_type","class","category"])
    if not t: return pd.Series(0.5, index=df.index)
    s = df[t].astype(str).str.lower().str.strip()
    base = {"earth-like":1.0,"earthlike":1.0,"terrestrial":1.0,"super-earth":0.8,"superearth":0.8,
            "mini-neptune":0.6,"sub-neptune":0.6,"neptune-like":0.5,"neptune":0.5,
            "gas giant":0.3,"jupiter-like":0.3,"jupiter":0.3,"unknown":0.5}
    out = s.map(base)
    fb = pd.Series(0.5, index=s.index)
    fb[s.str.contains("earth", na=False)] = 1.0
    fb[s.str.contains("super", na=False)] = 0.8
    fb[s.str.contains("mini", na=False) | s.str.contains("sub", na=False)] = 0.6
    fb[s.str.contains("neptune", na=False)] = 0.5
    fb[s.str.contains("jupiter", na=False) | s.str.contains("giant", na=False)] = 0.3
    out = out.fillna(fb).fillna(0.5)
    return out.astype(float)

def sc_radio(df):
    if df is None or len(df) == 0: return None
    pcol = find_first(df, ["proba_silent","prob_silent","p_silent","silent_prob"])
    scol = find_first(df, ["radio_silence_risk","risk","radio_silence_score"])
    bcol = find_first(df, ["radio_silent","silent_label","label_silent"])
    if pcol: return minmax01(df[pcol]).fillna(0.5)
    if scol: return minmax01(df[scol]).fillna(0.5)
    if bcol: return df[bcol].map({1:1.0,0:0.0}).fillna(0.5).astype(float)
    return pd.Series(0.5, index=df.index)

# ---------- Build unified ----------
def build_unified(weights=(0.35,0.35,0.15,0.15), dedup="max"):
    kepids = set()
    for df in [t2_df, t3_df, t1_df, t4_df]:
        if isinstance(df, pd.DataFrame) and len(df):
            kcol = find_first(df, ["kepid","keplerid","kid","kic"])
            if kcol:
                kepids.update(df[kcol].astype(str).str.lstrip("0").dropna().tolist())
    if not kepids:
        return pd.DataFrame()
    uni = pd.DataFrame({"kepid": sorted(kepids)})

    def agg_series(series: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
        kcol = find_first(df, ["kepid","keplerid","kid","kic"])
        tmp = pd.DataFrame({"kepid": df[kcol].astype(str).str.lstrip("0"),
                            "val": pd.to_numeric(series, errors="coerce")})
        g = tmp.groupby("kepid", as_index=False)["val"].mean() if dedup == "mean" else \
            tmp.groupby("kepid", as_index=False)["val"].max()
        return g

    det = sc_detection(t1_df); hab = sc_habitability(t3_df); typ = sc_planet_type(t2_df); rad = sc_radio(t4_df)
    if det is not None and t1_df is not None and len(t1_df):
        uni = uni.merge(agg_series(det, t1_df).rename(columns={"val":"score_det"}), on="kepid", how="left")
    else:
        uni["score_det"] = np.nan
    if hab is not None and t3_df is not None and len(t3_df):
        uni = uni.merge(agg_series(hab, t3_df).rename(columns={"val":"score_hab"}), on="kepid", how="left")
    else:
        uni["score_hab"] = np.nan
    if typ is not None and t2_df is not None and len(t2_df):
        uni = uni.merge(agg_series(typ, t2_df).rename(columns={"val":"score_typ"}), on="kepid", how="left")
    else:
        uni["score_typ"] = np.nan
    if rad is not None and t4_df is not None and len(t4_df):
        uni = uni.merge(agg_series(rad, t4_df).rename(columns={"val":"score_rad"}), on="kepid", how="left")
    else:
        uni["score_rad"] = np.nan

    w = np.array(weights, dtype=float); w = w / (w.sum() if w.sum() != 0 else 1.0)
    uni["probe_score"] = (
        w[0] * uni["score_det"].fillna(0.5) +
        w[1] * uni["score_hab"].fillna(0.5) +
        w[2] * uni["score_typ"].fillna(0.5) +
        w[3] * uni["score_rad"].fillna(0.5)
    )

    def add_field(df, src, cols):
        if df is None or not len(df): return
        c = find_first(df, cols); kcol = find_first(df, ["kepid","keplerid","kid","kic"])
        if not c or not kcol: return
        tmp = df[[kcol, c]].copy()
        tmp.columns = ["kepid", "val"]
        tmp["kepid"] = tmp["kepid"].astype(str).str.lstrip("0")
        first = tmp.dropna(subset=["val"]).groupby("kepid", as_index=False).first()
        uni_loc = uni.merge(first.rename(columns={"val": f"{src}__{c}"}), on="kepid", how="left")
        for col in uni_loc.columns:
            if col not in uni.columns: uni[col] = uni_loc[col]
        for col in uni.columns:
            if col in uni_loc.columns: uni[col] = uni_loc[col]

    add_field(t2_df, "t2", ["pl_name","planet_name","name","pl_name_std"])
    add_field(t2_df, "t2", ["planet_type","type","predicted_type","class","planet_type_rule"])
    add_field(t3_df, "t3", ["habitability_label","habitability"])
    return uni

# ---------- Live Top Targets ----------
st.markdown("### 🔝 Top Targets (Live)")
weights = (W_DET, W_HAB, W_TYP, W_RAD)
uni = build_unified(weights, DEDUP)

if uni is None or uni.empty:
    st.info("No KEPIDs found. Check your T2/T3 mapped files and paths in the sidebar.")
else:
    cA, cB, cC, cD = st.columns([1, 1, 1, 1])
    with cA: topk = st.number_input("Show Top", 5, 500, 25, 5)
    with cB: search = st.text_input("Filter KEPID(s)", placeholder="e.g. 00869227, 01156428")
    with cC: min_probe = st.slider("Min probe", 0.0, 1.0, 0.0, 0.05)
    with cD: show_human = st.toggle("Human-readable view", value=True)

    dfv = uni.copy()
    if search.strip():
        ks = [k.strip().lstrip("0") for k in search.replace(",", " ").split() if k.strip()]
        dfv = dfv[dfv["kepid"].isin(ks)]
    if min_probe > 0:
        dfv = dfv[dfv["probe_score"] >= min_probe]
    dfv = dfv.sort_values("probe_score", ascending=False).head(int(topk))

    def reason_row(r):
        parts = []
        if pd.notna(r["score_hab"]) and r["score_hab"] >= 0.8: parts.append("High habitability")
        if pd.notna(r["score_typ"]) and r["score_typ"] >= 0.8: parts.append("Earth-like/Super-Earth")
        if pd.notna(r["score_det"]) and r["score_det"] >= 0.7: parts.append("Strong detection")
        if pd.notna(r["score_rad"]) and r["score_rad"] <= 0.3: parts.append("Lower radio-silence risk")
        return ", ".join(parts) if parts else "Balanced signals"
    dfv["reason"] = dfv.apply(reason_row, axis=1)

    nice_cols = {
        "probe_score": st.column_config.ProgressColumn("Probe Score", format="%.2f", min_value=0.0, max_value=1.0),
        "score_det":  st.column_config.ProgressColumn("Detection",    format="%.2f", min_value=0.0, max_value=1.0),
        "score_hab":  st.column_config.ProgressColumn("Habitability", format="%.2f", min_value=0.0, max_value=1.0),
        "score_typ":  st.column_config.ProgressColumn("Planet Type",  format="%.2f", min_value=0.0, max_value=1.0),
        "score_rad":  st.column_config.ProgressColumn("Radio",        format="%.2f", min_value=0.0, max_value=1.0),
        "reason":     st.column_config.TextColumn("Why it ranks")
    }
    base_cols = ["kepid","probe_score","score_det","score_hab","score_typ","score_rad","reason"]
    extras = [c for c in dfv.columns if c.startswith("t2__pl_name") or "habitability" in c or "planet_type" in c]
    st.dataframe(dfv[base_cols + extras].reset_index(drop=True), use_container_width=True, height=420,
                 column_config=nice_cols if show_human else None)
    st.download_button("⬇️ Download Top (CSV)", dfv.to_csv(index=False).encode("utf-8"),
                       file_name="mission_control_top_targets.csv", mime="text/csv")

# ---------- Tabs (3D Plots removed) ----------
st.markdown("### 🧭 Navigation")
tab1, tab2, tab3, tabEDA, tab4, tab5, tab6 = st.tabs([
    "🔎 KEPID Explorer", "⚖️ Compare", "📁 Data Catalog", "📊 EDA", "🖼️ Gallery", "⭐ Highlights", "🌌 Sky View"
])

# ---------- Explorer ----------
with tab1:
    st.markdown("#### 🔎 Explorer")
    if uni is None or uni.empty:
        st.info("Load T2/T3 mapped files to enable the explorer.")
    else:
        sel = st.text_input("Enter a single KEPID", placeholder="e.g., 00869227")
        if sel.strip():
            k = sel.strip().lstrip("0")
            row = uni[uni["kepid"] == k]
            if row.empty:
                st.warning("KEPID not found in unified table.")
            else:
                st.write("**Unified signals**")
                st.table(row[["probe_score","score_det","score_hab","score_typ","score_rad"]])

                def show_src(df, title):
                    if df is None or not len(df): return
                    kcol = find_first(df, ["kepid","keplerid","kid","kic"])
                    if not kcol: return
                    r = df[df[kcol].astype(str).str.lstrip("0") == k]
                    if len(r):
                        r = r.loc[:, ~r.columns.duplicated()]
                        st.markdown(f"**{title}**")
                        st.dataframe(r.head(200), use_container_width=True, height=240)
                show_src(t1_df, "Task 1 — Detection")
                show_src(t2_df, "Task 2 — Characterization")
                show_src(t3_df, "Task 3 — Habitability")
                show_src(t4_df, "Task 4 — Radio")

# ---------- Compare ----------
with tab2:
    st.markdown("#### ⚖️ Compare KEPIDs")
    if uni is None or uni.empty:
        st.info("Load T2/T3 mapped files first.")
    else:
        ks = st.text_input("List KEPIDs (comma/space)", placeholder="e.g., 00869227 01156428 01234567")
        if ks.strip():
            picks = [p.strip().lstrip("0") for p in ks.replace(",", " ").split() if p.strip()]
            comp = uni[uni["kepid"].isin(picks)].copy().sort_values("probe_score", ascending=False)
            if comp.empty:
                st.warning("None of those KEPIDs are present.")
            else:
                comp["rank"] = range(1, len(comp) + 1)
                st.dataframe(comp[["rank","kepid","probe_score","score_det","score_hab","score_typ","score_rad"]],
                             use_container_width=True, height=380)
                st.bar_chart(comp.set_index("kepid")[["probe_score","score_det","score_hab","score_typ","score_rad"]])

# ---------- Data Catalog ----------
with tab3:
    st.markdown("#### 📁 Data Catalog & Diagnostics")
    def diag_df(name, df):
        st.write(f"**{name}:** {'OK ✅' if (df is not None and len(df)) else 'Missing ⚠️'}")
        if df is not None and len(df):
            sample = df.head(10).loc[:, ~df.columns.duplicated()]
            st.caption(f"rows: {len(df):,} | cols: {len(sample.columns)}")
            kcol = find_first(sample, ["kepid","keplerid","kid","kic"])
            st.write(f"- KEPID column: `{kcol or '—'}`")
            st.write("- Sample:")
            st.dataframe(sample, use_container_width=True, height=220)
    diag_df("Task 1", t1_df); diag_df("Task 2 (mapped)", t2_df); diag_df("Task 3 (mapped)", t3_df); diag_df("Task 4", t4_df)
    if t5_df is not None and len(t5_df): diag_df("Task 5 (precomputed)", t5_df)

# ---------- EDA Tab ----------
def list_media(root):
    if not os.path.isdir(root): return [], []
    imgs = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True) +
                  glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True) +
                  glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True))
    htmls = sorted(glob.glob(os.path.join(root, "**", "*.html"), recursive=True))
    return imgs, htmls

def human_size(bytes_: int) -> str:
    mb = bytes_ / (1024*1024)
    return f"{mb:.1f} MB"

def show_html_safely(html_files, embed: bool, max_mb: int, label: str):
    if not html_files:
        return
    st.markdown(f"**{label}**")
    max_bytes = max_mb * 1024 * 1024
    for p in html_files[:8]:
        size = os.path.getsize(p)
        fname = os.path.basename(p)
        st.caption(f"{fname} — {human_size(size)}")
        if embed and size <= max_bytes:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)
            except MemoryError:
                st.warning("Skipped embedding due to memory limits. Showing path instead.")
                st.code(p)
        else:
            if not embed:
                st.caption("Embedding is OFF — showing path:")
            else:
                st.caption(f"Too large to embed (>{max_mb} MB) — showing path:")
            st.code(p)

with tabEDA:
    st.markdown("#### 📊 EDA (from your EDA folder)")
    st.caption(eda_root)
    if os.path.isdir(eda_root):
        imgs, htmls = list_media(eda_root)
        st.caption(f"Found {len(imgs)} images and {len(htmls)} HTML files in EDA.")
        total = len(imgs)
        n = safe_slider("Images to show (EDA)", total, default=27, cap=300, key="eda_n")
        if total:
            cols = st.columns(3)
            for i, p in enumerate(imgs[:n]):
                with cols[i % 3]:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)
        # HTML (safe)
        show_html_safely(htmls, EMBED_HTML, MAX_HTML_MB, "Interactive EDA (HTML)")
    else:
        st.warning("EDA folder not found.")

# ---------- General Gallery ----------
with tab4:
    st.markdown("#### 🖼️ General Gallery")
    root = st.text_input("Scan root", DEFAULT_OUTPUT_ROOT, key="gallery_root")
    if os.path.isdir(root):
        imgs, htmls = list_media(root)
        st.caption(f"Found {len(imgs)} images and {len(htmls)} HTML files.")
        total = len(imgs)
        n = safe_slider("Images to show", total, default=27, cap=300, key="gallery_n")
        if total:
            cols = st.columns(3)
            for i, p in enumerate(imgs[:n]):
                with cols[i % 3]:
                    st.image(p, caption=os.path.basename(p), use_column_width=True)
        show_html_safely(htmls, EMBED_HTML, MAX_HTML_MB, "Interactive (HTML)")
    else:
        st.warning("Folder not found.")

# ---------- Highlights (Tasks 1–5) ----------
def pick_highlights(root):
    if not os.path.isdir(root): return {}
    patterns = {
        "Task1": ["**/Task1/**/roc*.png","**/Task1/**/pr*.png","**/Task1/**/confusion*.png","**/Task1/**/acc*.png",
                  "*task1*roc*.png","*cnn*roc*.png","*multimodel*roc*.png","**/Task1/**/loss*.png","**/Task1/**/accuracy*.png"],
        "Task2": ["**/Task2/**/roc*.png","**/Task2/**/cm*.png","**/Task2/**/report*.png","*feature*importance*.png",
                  "*task2*roc*.png","*characterization*roc*.png","**/Task2/**/pr*.png"],
        "Task3": ["**/Task3/**/roc*.png","**/Task3/**/pr*.png","**/Task3/**/cm*.png",
                  "*task3*roc*.png","*habit*roc*.png","*habit*cm*.png","**/Task3/**/report*.png"],
        "Task4": ["**/Task4/**/roc*.png","**/Task4/**/cm*.png","**/Task4/**/pr*.png",
                  "*task4*roc*.png","*radio*roc*.png","*radio*cm*.png"],
        "Task5": ["**/Task5/**/roc*.png","**/Task5/**/cm*.png","**/Task5/**/pr*.png",
                  "*task5*roc*.png","*probe*score*curve*.png","*ranking*curve*.png"],
    }
    out = {}
    for sect, globs_ in patterns.items():
        files = []
        for pat in globs_:
            files += glob.glob(os.path.join(root, pat), recursive=True)
        files = sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True)
        if files: out[sect] = files[:8]
    return out

with tab5:
    st.markdown("#### ⭐ Highlights — key metrics & model performance (Tasks 1–5)")
    root = st.text_input("Output root (for highlights)", DEFAULT_OUTPUT_ROOT, key="hlroot")
    if os.path.isdir(root):
        groups = pick_highlights(root)
        if not groups:
            st.info("No highlight images found yet. Put ROC/PR/Confusion/Loss images in Output and they’ll show here.")
        else:
            for sect, files in groups.items():
                st.markdown(f"**{sect}**")
                cols = st.columns(4)
                for i, p in enumerate(files):
                    with cols[i % 4]:
                        st.image(p, caption=os.path.basename(p), use_column_width=True)
                st.markdown("---")
    else:
        st.warning("Folder not found.")

# ---------- Sky View + Molecules (same as v3.4) ----------
@st.cache_data(show_spinner=False)
def load_science_data(planet_path, atmos_path, star_path):
    def _read(path):
        return read_csv_smart(path) if os.path.isfile(path) else None
    def _dedup(df):
        if not isinstance(df, pd.DataFrame): return df
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = [c.strip() for c in df.columns]
        return df
    pl  = _dedup(_read(planet_path))
    at  = _dedup(_read(atmos_path))
    stl = _dedup(_read(star_path))
    return pl, at, stl

def detect_col(df, names): return find_first(df, names) if df is not None else None

def sky_dataframe(pl, at, stl):
    frames = []
    for name, df in [("planet", pl), ("atmos", at), ("star", stl)]:
        if df is None or not len(df): continue
        df = df.loc[:, ~df.columns.duplicated()]
        ra = detect_col(df, ["ra","ra_deg","radeg","ra2000","raj2000","rightascension","st_ra","ra[deg]"])
        dec = detect_col(df, ["dec","dec_deg","decdeg","dec2000","decj2000","declination","st_dec","dec[deg]"])
        dist = detect_col(df, ["st_dist","sy_dist","distance","dist","st_dist_pc","sy_dist_pc","dist_pc"])
        mag = detect_col(df, ["st_vmag","vmag","mag_v","stellar_mag","sy_vmag","kic_mag","vmagnitude"])
        namecol = detect_col(df, ["pl_name","planet_name","name","hostname","star_name","object_name"])
        kepidcol = detect_col(df, ["kepid","keplerid","kic","kid"])
        tmp = pd.DataFrame()
        if ra:   tmp["ra"]   = pd.to_numeric(df[ra], errors="coerce")
        if dec:  tmp["dec"]  = pd.to_numeric(df[dec], errors="coerce")
        if dist: tmp["dist"] = pd.to_numeric(df[dist], errors="coerce")
        if mag:  tmp["vmag"] = pd.to_numeric(df[mag], errors="coerce")
        tmp["name"]  = df[namecol].astype(str) if namecol else name
        tmp["kepid"] = df[kepidcol].astype(str).str.lstrip("0") if kepidcol else np.nan
        tmp["source"] = name
        if "ra" in tmp.columns and "dec" in tmp.columns:
            frames.append(tmp[["ra","dec","dist","vmag","name","kepid","source"]])
    if not frames:
        return pd.DataFrame(columns=["ra","dec","dist","vmag","name","kepid","source"])
    sky = pd.concat(frames, ignore_index=True).dropna(subset=["ra","dec"])
    if sky["ra"].max() <= 24.1:  # hours -> degrees
        sky["ra"] = sky["ra"] * 15.0
    sky["dist"] = pd.to_numeric(sky["dist"], errors="coerce")
    sky["vmag"] = pd.to_numeric(sky["vmag"], errors="coerce")
    return sky

def sph_to_xyz(ra_deg, dec_deg, r):
    ra = np.deg2rad(ra_deg); dec = np.deg2rad(dec_deg)
    x = r * np.cos(dec) * np.cos(ra)
    y = r * np.cos(dec) * np.sin(ra)
    z = r * np.sin(dec)
    return x, y, z

# Molecule parsing (from v3.4)
MOLECULE_MAP = {"H20":"H2O","H₂O":"H2O","WATER":"H2O","CO₂":"CO2","C02":"CO2","SI02":"SIO2","SILICA":"SIO2",
                "TIO":"TiO","VO":"VO","N20":"N2O","SO2":"SO2"}
DROP_TOKENS = {"UPPER","UPPERATMOSPHERE","NO","NONE","UNKNOWN","NULL","UNDETECTED","-","N/A","NA"}
CORE_MOLS = ["H2O","CH4","CO2","CO","O2","O3","NH3","N2O","HCN","Na","K","He","H","SiO2","TiO","VO","Mg","Fe","Ca","SiO","SO2","C/O"]

def normalize_molecule_token(tok: str) -> str | None:
    t = tok.strip().upper()
    if not t: return None
    t = t.replace(" OR ", ",").replace("AND", ",")
    t = t.replace(" ", "").replace("\u2009","")
    t = MOLECULE_MAP.get(t, t)
    if t in DROP_TOKENS: return None
    if t in {"C/O","C/0"}: return "C/O"
    if re.fullmatch(r'[A-Z][a-z]?', t): return t
    if re.fullmatch(r'[A-Z][a-zA-Z]?\d*[A-Z]?[a-z]?\d*', t): return t
    return None

def parse_molecule_list(raw: str) -> list[str]:
    if pd.isna(raw): return []
    s = str(raw).replace(";", ",").replace("/", ",").replace("|", ",")
    s = s.replace("\n", ",").replace("\r", ",")
    parts = [p for p in re.split(r',+', s) if p is not None]
    cleaned = []
    for p in parts:
        val = normalize_molecule_token(p)
        if val: cleaned.append(val)
    seen, out = set(), []
    for v in cleaned:
        if v not in seen: seen.add(v); out.append(v)
    return out

with tab6:
    st.markdown("#### 🌌 3D Sky View (Planets & Hosts) + Atmospheres")
    pl, at, stl = load_science_data(planet_file, atmos_file, star_file)
    if pl is None and stl is None and at is None:
        st.warning("Connect your science datasets in the sidebar.")
    else:
        sky = sky_dataframe(pl, at, stl)
        if sky.empty:
            st.info("Could not find usable RA/Dec columns in the provided files.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1: mag_max = st.slider("Max visual magnitude (brighter first)", 0.0, 20.0, 15.0, 0.5)
            with c2: max_pts = st.slider("Max points", 100, 10000, 3000, 100)
            with c3: scale = st.selectbox("Radius scaling", ["1/dist", "log(1+dist)"], index=0)

            df = sky.copy()
            if "vmag" in df.columns:
                df = df[df["vmag"].isna() | (df["vmag"] <= mag_max)]
            df = df.head(max_pts)

            d = df["dist"].fillna(df["dist"].median() if df["dist"].notna().sum() else 100.0).astype(float)
            r = 1.0/(1e-6 + d) if scale == "1/dist" else 1.0/np.log1p(1e-6 + d)
            x, y, z = sph_to_xyz(df["ra"].values, df["dec"].values, r.values)
            df["x"], df["y"], df["z"] = x, y, z

            if HAS_PLOTLY:
                fig = go.Figure(data=[go.Scatter3d(
                    x=df["x"], y=df["y"], z=df["z"],
                    mode="markers",
                    marker=dict(size=3, opacity=0.7),
                    text=[f"{n}<br>KEPID: {k}" for n, k in zip(df["name"].fillna('—'), df["kepid"].fillna('—'))],
                    hovertemplate="%{text}<br>RA %{x:.3f}, Dec %{y:.3f}, Z %{z:.3f}<extra></extra>"
                )])
                fig.update_layout(height=620, margin=dict(l=0,r=0,t=30,b=0),
                                  scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
                st.plotly_chart(fig, use_container_width=True)
            elif HAS_PYDECK:
                layer = pdk.Layer(
                    "PointCloudLayer",
                    data=df[["x","y","z","name","kepid"]].rename(columns={"x":"position_x","y":"position_y","z":"position_z"}),
                    get_position=["position_x","position_y","position_z"],
                    get_normal=[0,0,1], point_size=2
                )
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1, pitch=40)))
            else:
                st.info("Plotly/PyDeck not installed — showing 2D RA vs Dec scatter.")
                st.scatter_chart(df[["ra","dec"]].rename(columns={"ra":"RA","dec":"Dec"}))

    # Molecules (cleaned from molecule_list)
    if at is not None and len(at):
        at_u = at.loc[:, ~at.columns.duplicated()]
        namecol = find_first(at_u, ["pl_name","planet_name","name"])
        molcol  = find_first(at_u, ["molecule_list","molecules","molecule","molecular","atmosphere_composition"])
        if molcol:
            mol_series = at_u[molcol].apply(parse_molecule_list)
            mol_str = mol_series.apply(lambda L: ", ".join(L) if L else "—")
            out = pd.DataFrame({
                "pl_name": at_u[namecol] if namecol else np.arange(len(at_u)),
                "molecules": mol_str,
                "n_molecules": mol_series.apply(len)
            })
            CORE_MOLS = ["H2O","CH4","CO2","CO","O2","NH3","N2O","HCN","Na","K","He","H","TiO","VO","SiO2","SO2","C/O"]
            for m in CORE_MOLS:
                out[m] = mol_series.apply(lambda L, mm=m: 1 if mm in L else 0)
            st.markdown("**Atmospheric molecules (cleaned)**")
            st.dataframe(out.head(60), use_container_width=True)
        else:
            st.info("Could not find a 'molecule_list' column in the atmospheres file.")

# ---------- Footer ----------
st.markdown("<div class='foot'>No personal data is processed. Radio-silence is a proxy score. Interpret responsibly.</div>", unsafe_allow_html=True)
