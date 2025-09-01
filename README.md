# Exoplanet Multi-Stage ML (Kepler) — Tasks 1–5

**Goal.** Build an auditable ML pipeline to (T1) detect transits from Kepler light curves, (T2) characterise planets, (T3) predict habitability, (T4) predict radio-silence risk, and (T5) rank targets for an autonomous probe; plus a Streamlit mission control dashboard.

**This repo includes all code versions** (baseline → tuned → final) and shows development over time via commits (a UH requirement).

## Folder map
- `Codes/`
  - `Task1/` CNN V1–V4 + calibration utilities
  - `Task2/` characterisation + KOI/KEPID mapping + leakage ablation
  - `Task3/` habitability (RF/LGBM) + calibration + ID mapping
  - `Task4/` radio-silence pipeline + threshold selection
  - `Task5/` probe decision pipeline + weight sensitivity
  - `Streamlit/` `streamlit_app_mission_control_pro_v3.5.py`
- `Output/` results per task (metrics, predictions, figures), plus `metrics_summary.csv` & `figure_manifest.csv`
- `Datasets/` data notes and tiny samples (no big raw data in repo)
- `Docs/` ethics note, figure gallery used in the FPR
- `configs/` JSON/YAML configs for paths and weights
- `scripts/` helper runners (e.g., `run_all_tasks.py`)

## Environment (Windows example)
```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -r requirements.txt
