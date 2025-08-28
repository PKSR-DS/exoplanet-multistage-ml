# task1_calibrate_threshold.py
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve, auc

IN = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task1\multi_run_20250825_094156\V4_CBAM\predictions.csv"
OUT_DIR = os.path.join(os.path.dirname(IN), "calibration")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(IN)
# Try common column names
y_col = next(c for c in df.columns if c.lower() in ["y_true","label","target"])
p_col = next(c for c in df.columns if c.lower() in ["y_prob","y_pred_proba","probability","pred_prob","cnn_score"])

y = df[y_col].astype(int).values
p = df[p_col].astype(float).values

# -------- threshold tuning (max F1) --------
best_t, best_f1 = 0.5, -1
ts = np.linspace(0,1,201)
for t in ts:
    f1 = f1_score(y, (p>=t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

# -------- isotonic calibration on probs --------
# (maps raw prob -> calibrated prob)
iso = IsotonicRegression(out_of_bounds='clip')
p_cal = iso.fit_transform(p, y)

# Curves (raw vs calibrated)
pr_raw = precision_recall_curve(y, p)
pr_cal = precision_recall_curve(y, p_cal)
roc_raw = roc_curve(y, p)
roc_cal = roc_curve(y, p_cal)

aupr_raw = auc(pr_raw[1], pr_raw[0]); aupr_cal = auc(pr_cal[1], pr_cal[0])
auc_raw = auc(roc_raw[0], roc_raw[1]); auc_cal = auc(roc_cal[0], roc_cal[1])

# Save calibrated preds + best threshold
out_csv = os.path.join(OUT_DIR, "task1_calibrated_preds.csv")
pd.DataFrame({"y_true":y, "p_raw":p, "p_cal":p_cal}).to_csv(out_csv, index=False)

with open(os.path.join(OUT_DIR, "task1_threshold_and_metrics.json"), "w") as f:
    json.dump({
        "best_threshold_maxF1": best_t,
        "best_F1": best_f1,
        "PR_AUPR_raw": aupr_raw,
        "PR_AUPR_cal": aupr_cal,
        "ROC_AUC_raw": auc_raw,
        "ROC_AUC_cal": auc_cal
    }, f, indent=2)

# Plot PR (raw vs cal)
plt.figure()
plt.plot(pr_raw[1], pr_raw[0], label=f"raw (AUPR={aupr_raw:.3f})")
plt.plot(pr_cal[1], pr_cal[0], label=f"calibrated (AUPR={aupr_cal:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Task1 — PR (raw vs calibrated)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "task1_pr_raw_vs_cal.png")); plt.close()

# Plot ROC (raw vs cal)
plt.figure()
plt.plot(roc_raw[0], roc_raw[1], label=f"raw (AUC={auc_raw:.3f})")
plt.plot(roc_cal[0], roc_cal[1], label=f"calibrated (AUC={auc_cal:.3f})")
plt.plot([0,1],[0,1],'--',alpha=0.5)
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("Task1 — ROC (raw vs calibrated)")
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "task1_roc_raw_vs_cal.png")); plt.close()

print("DONE:", out_csv)
