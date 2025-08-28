# task4_threshold.py
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

CSV = r"D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task4\run_20250826_213438\task4_radio_silence_scores.csv"
df = pd.read_csv(CSV)
y = df['true_label'].values if 'true_label' in df.columns else (df['Radio_Silent'].values if 'Radio_Silent' in df.columns else None)
p = df['risk'].values if 'risk' in df.columns else df.filter(like='risk').iloc[:,0].values

best_t, best_f1 = 0.5, -1
for t in np.linspace(0,1,201):
    f1 = f1_score(y, (p>=t).astype(int)) if y is not None else 0
    if f1 > best_f1:
        best_f1, best_t = f1, t
print("Chosen threshold:", best_t, " (max F1 on available labels)")

