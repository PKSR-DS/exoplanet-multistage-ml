# Exoplanet Multi-Stage ML (Kepler) – Tasks 1–5

**Research question.** To what extent can a multi-stage ML pipeline detect exoplanet transits from noisy light curves, characterise planets, predict habitability and radio-silence risk, and rank targets for an autonomous probe?

**Pipeline (Tasks).**
1) **Transit Detection** (Task 1): CNN V1–V4 (best = V3_SE). Outputs: ROC/PR, F1, calibration, science recovery.
2) **Characterisation** (Task 2): Stage-1 regressors (mass/eq. temp/density); Stage-2 planet-type classifier; molecules (multilabel).
3) **Habitability** (Task 3): RF/LGBM + SHAP; labels: Non-Habitable / Partially / **Habitable (Potentially)**.
4) **Radio Silence** (Task 4): lightweight logistic; fairness by stellar Teff quartiles.
5) **Probe Decision** (Task 5): deterministic scorer combining T1–T4 with weights; Streamlit dashboard.

> **Environment**: Python 3.12.3; TensorFlow 2.19.0; NumPy 1.26.4; pandas 2.2.3; scikit-learn; lightgbm; shap; matplotlib; plotly; streamlit.

## How to run

```bash
# 0) Create env
python -m venv .venv && . .venv/Scripts/activate  # Windows
pip install -r requirements.txt

# 1) Prepare data paths / small samples
python scripts/prepare_data.py --config configs/paths.json

# 2) Run tasks individually
python Codes/Task1/train_and_eval.py --config configs/task1.json
python Codes/Task2/run_characterisation.py --config configs/task2.json
python Codes/Task3/run_habitability.py --config configs/task3.json
python Codes/Task4/run_radio_silence.py --config configs/task4.json
python Codes/Task5/run_probe_decision.py --config configs/task5.json

# or all at once
python scripts/run_all_tasks.py --fast
