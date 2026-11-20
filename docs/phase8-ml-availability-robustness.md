# Phase 8 — Historical demand availability robustness

Run `python -m gridwise.robust_ml` after Phases 4–7. The output compares **retrained** 24-hour-ahead LightGBM forecasts with validation-selected historical baselines under assumed demand-publication buffers of 0, 15, 30 and 60 minutes. Use `--buffers 0 15 30 60 120` to include 120 minutes.

Every demand-derived input is shifted by the stated buffer; this includes point lags **and** trailing demand means. Target-time calendar features and ERA5 weather features lagged by two weeks are unchanged. Both model candidates and the three baselines are evaluated using the same target timestamps across buffers. Candidate selection and baseline selection use validation MAE, not test MAE.

`reports/robust_ml_2026.json` stores exact numbers; `data/model_outputs/robust_ml_sensitivity_2026.png` shows trends, and `data/model_outputs/robust_ml_predictions_2026.csv` keeps paired predictions.

**Limitations:** Buffers are assumptions, not measured Fingrid publication times. The source values may have been revised since they were published. The July–October 2026 test period was examined in earlier phases, so these are retrospective diagnostics, **not a new independent test**. The weather fields are old reanalysis measurements and are not point-in-time archived forecasts. Before making deployment claims, collect historical issuance-time availability records, use archived forecast weather, and evaluate on genuinely future data.
