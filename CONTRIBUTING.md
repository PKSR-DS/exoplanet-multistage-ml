# Contributing / How to run

## Environment
- Python 3.12.3. Install with: `pip install -r requirements.txt`.

## Reproducibility
- Set random seeds (42) where applicable.
- Save metrics to `Output/Task*/metrics.json`; do not overwrite past runs (use timestamped folders).
- Keep raw data outside git; use `configs/paths.json` to point to local paths.

## Style
- One script per task is runnable from CLI (no notebook-only steps).
- Figures saved to `Output/Task*/.
