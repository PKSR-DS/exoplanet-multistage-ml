import subprocess, sys

CMDS = [
    ["python", "Codes/Task 1/Task 1 multimodel CNNV1-4.py"],
    ["python", "Codes/Task 1/task1_calibrate_threshold.py"],
    ["python", "Codes/Task 1/Task-1 CSV exposes both raw & calibrated.py"],
    ["python", "Codes/Task2/Exoplanet Characterization 1.0.py"],
    ["python", "Codes/Task2/Kepler ID mapping with output.py"],
    ["python", "Codes/Task2/Task 2 — Leakage sanity check (ablation).py"],
    ["python", "Codes/Task3/Habitability task final.py"],
    ["python", "Codes/Task3/calibration for RF_LGBM.py"],
    ["python", "Codes/Task3/kepler id mapping.py"],
    ["python", "Codes/Task4/task4_radio_silence_pipeline.py"],
    ["python", "Codes/Task4/Task 4 — Decision threshold.py"],
    ["python", "Codes/Task5/task5_probe_decision_pipeline 2.1.py"],
    ["python", "Codes/Task5/Task 5 — Weight Sensitivity.py"],
]

def main():
    for cmd in CMDS:
        print(">>", " ".join(cmd))
        code = subprocess.call(cmd)
        if code != 0:
            sys.exit(code)

if __name__ == "__main__":
    main()
