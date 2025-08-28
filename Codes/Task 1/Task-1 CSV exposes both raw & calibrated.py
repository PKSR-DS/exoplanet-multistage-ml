# unify_task1_preds.py
import pandas as pd

df = pd.read_csv(r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task1\multi_run_20250825_094156\V4_CBAM\calibration\task1_calibrated_preds.csv")  
# Standard column names for downstream steps
rename_map = {
    # use whatever your current headers are
    "y_proba_raw":"pred_prob_raw",
    "y_proba_cal":"pred_prob_cal",
    "y_true":"y_true",
}
df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
# Add best-F1 threshold predictions (0.825 from calibration run)
THR = 0.825
if "pred_prob_cal" in df.columns:
    df["y_pred_bestF1"] = (df["pred_prob_cal"] >= THR).astype(int)
elif "pred_prob_raw" in df.columns:
    df["y_pred_bestF1"] = (df["pred_prob_raw"] >= THR).astype(int)
df.to_csv(r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task1\multi_run_20250825_094156\V4_CBAM\calibration\task1_transit_predictions.csv", index=False)

