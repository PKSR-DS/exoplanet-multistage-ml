# Task 1 Multi‑Model Runner — Checklist + Science
- Data: `D:\UH One drive\Kepler_Preprocessed`
- KOI: `D:\UH One drive\cumulative_2025.07.16_10.28.33.csv`
- Models run: ['v1', 'v2', 'v3', 'v4']
- Output: `D:\UH One drive\OneDrive - University of Hertfordshire\Output\Task1\multi_run_20250825_094156`

## Metrics (comparison)
| model       |   threshold_primary |   threshold_precision_first |   threshold_recall_first |   chosen_threshold |   train_seconds |   accuracy |   precision |   recall |       f1 |   roc_auc |   pr_auc |     brier |   science_confirmed_in_test |   science_recovered |   science_missed |   science_recovery_rate |
|:------------|--------------------:|----------------------------:|-------------------------:|-------------------:|----------------:|-----------:|------------:|---------:|---------:|----------:|---------:|----------:|----------------------------:|--------------------:|-----------------:|------------------------:|
| V1_Plain    |                0.85 |                        0.9  |                     0.05 |               0.85 |         1436.94 |   0.965429 |    0.93151  | 0.850746 | 0.889298 |  0.982314 | 0.950138 | 0.0524052 |                         208 |                 190 |               18 |                0.913462 |
| V2_Residual |                0.8  |                        0.85 |                     0.05 |               0.8  |         1559.01 |   0.952686 |    0.928727 | 0.769144 | 0.841436 |  0.97738  | 0.930495 | 0.047461  |                         208 |                 188 |               20 |                0.903846 |
| V3_SE       |                0.8  |                        0.85 |                     0.05 |               0.8  |         1441.21 |   0.970008 |    0.941501 | 0.870322 | 0.904513 |  0.986683 | 0.962399 | 0.039229  |                         208 |                 196 |               12 |                0.942308 |
| V4_CBAM     |                0.8  |                        0.85 |                     0.05 |               0.8  |         1208.67 |   0.96598  |    0.934822 | 0.850896 | 0.890887 |  0.985787 | 0.95678  | 0.0458055 |                         208 |                 192 |               16 |                0.923077 |

**Best model (by F1):** V3_SE
