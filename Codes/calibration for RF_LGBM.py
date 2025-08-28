# task3_class_metrics_and_cal.py
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import calibration_curve

# Either (A) read existing classification report…
CR = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task2\run_20250826_104658\classification_report_rf_tuned.csv"
if os.path.exists(CR):
    cr = pd.read_csv(CR)
    # Make macro summary for the report
    macro_f1 = cr[cr['precision'].notna()].set_index('Unnamed: 0').loc['macro avg','f1-score']
    print("macro-F1:", macro_f1)
else:
    # Or (B) recompute quickly if you have preds
    PRED = r"...\task3_predictions.csv"  # fallback if available
    df = pd.read_csv(PRED)
    y_true = df['y_true']; y_pred = df['y_pred']
    print(classification_report(y_true, y_pred, digits=3))

# Optional: quick calibration curve for probe per class (e.g., 'Habitable (Potentially)')
# df_proba['p_hab'] = probability of positive class
# prob_true, prob_pred = calibration_curve((y_true==pos_id).astype(int), df_proba['p_hab'], n_bins=10)
