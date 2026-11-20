# Phase 6 — paired forecast diagnostics

No retraining is needed. Run the existing Phase 5 training first to generate
`lightgbm_predictions_2026.csv` and `lightgbm_2026.txt`.

```powershell
.\.venv\Scripts\python.exe -m gridwise.diagnostics
```

This recomputes **all three baselines on the same timestamps** as the saved
LightGBM predictions, checks that the targets and the validation-selected
baseline match the original project files, and writes:

- `reports/diagnostics_2026.json` — overall, monthly, Finnish local daytime/nighttime metrics
- `data/model_outputs/feature_importance_2026.png` — LightGBM gain importance
- `data/model_outputs/monthly_mae_2026.png` — monthly paired evaluation

Important safeguards:

- The original baseline is selected only from validation results, not test.
- "Best on test" is descriptive, **not** an invitation to reselect the model.
- Feature importance is fitted-model split gain, **not** causal significance.
- Analyzed timestamps are the common, non-null subset for every method.
- No demand values are imputed, models refitted, or raw data changed.

Do not tune hyperparameters against the held-out July–October 2026 period.
Future enhancements should use rolling validation windows and a new final
untouched test period.
