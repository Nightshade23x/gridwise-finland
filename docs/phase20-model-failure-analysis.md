# Phase 20 — Forward-forecast failure diagnostics

This stage investigates why independently issued GridWise forecasts can have very different errors. It **does not train, recalibrate, issue, or replace** any models or captured predictions.

## Run (PowerShell)

From `D:\Projects\gridwise-finland-starter\gridwise-finland`:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.failure_analysis
.\.venv\Scripts\python.exe -m pytest -q
```

Existing archives must be present under `data/live/forecast_archive/` and/or `data/live/weather_model_archive/` (the latter uses the path defined by `gridwise.weather_forward.ARCHIVE`). The analyser reads archived snapshots and `data/live/consumption_recent.csv`, not the public Render demo or a revised historical backtest.

Outputs (ignored by Git):

- `data/live/failure_analysis.json`: one result per issued forecast, including measured coverage, error metrics, six-hour horizon breakdown, UTC-hour diagnostics, within-window demand-level split, and the largest individual errors.
- `data/model_outputs/failure_analysis_targets.csv`: observed prediction errors for each matched timestamp, identified by issuance. This is a debugging export, **not** a pooled performance score.
- `GET /api/failure-analysis`: read-only summary served by the local dashboard. The public historical Render demo does not expose these local forward-forecast results.

After a local analysis run, the Docker dashboard reads the saved JSON immediately. If this is a new code patch, rebuild Docker once with `docker compose up --build -d`, then open `http://127.0.0.1:8000`. Press **Reload failure analysis** to reread the saved report; it does not recalculate it.

## Interpretation

- **Signed bias** = predicted demand minus actual demand, so negative means underprediction. MAE is the average *absolute* error and cannot cancel positive/negative errors.
- A target is eligible only once its 15-minute interval ends, **plus** a configurable observation-publication safety buffer (60 minutes by default). Missing readings are not imputed.
- Every issuance is scored separately on precisely its own eligible, measured target timestamps. Weather, demand-only, and baseline candidates within the **same weather issuance** are paired for comparison.
- Two close-in-time snapshots may predict identical or overlapping targets. Identical curves are flagged. Neither duplicated curves nor overlapping issuances are treated as independent replications or pooled into a new headline MAE.
- The UTC-hour error ranking, observed-demand median split, and largest-error list are **descriptive diagnostics** for that issuance only. They are not causal explanations, general seasonal evidence, or legitimate future-observable input features.
- Partially observed forecasts remain marked `partial`; those MAEs must not be compared to 96/96 windows without accounting for different coverage.
- Local forecast issuance timestamps and archive hashes are locally stored evidence, **not** independently certified time-of-issuance proofs.

## Next modelling decisions (not included in Phase 20)

Investigate systematic level shifts and lag availability with controlled chronological experiments. Use multiple **new** issued forecast windows to evaluate weather features and compare with Fingrid; don't tune repeatedly against these already-inspected windows as if they were untouched holdouts.
