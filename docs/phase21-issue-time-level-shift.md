# Phase 21 — Issue-time level-shift diagnostic

## Goal

Investigate whether a 48-hour/7-day historical demand reference reacts too slowly when the national demand level changes. This is a **research diagnostic**, not a replacement for the existing LightGBM, weather-aware forecasts, or archived official benchmark.

## Run

From the project root (Windows PowerShell):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m gridwise.level_shift
```

The command reads `data/processed/consumption_2024_2026.csv` and saves:

- `data/live/level_shift_backtest.json` — chronological selection and later-period metrics.
- `data/model_outputs/level_shift_backtest.png` — concise MAE comparison.

Start/rebuild the local Docker dashboard to view **Phase 21 — Can recent demand correct an outdated baseline?**. The REST endpoint is `/api/level-shift` and is read-only.

An **optional** one-off candidate for the next 24 hours can be generated after an incremental refresh:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.level_shift --preview-now
```

This additional command writes `data/live/level_shift_preview.json`. It **does not** modify `forward_forecast_24h.json`, weather forecasts, model artifacts, or historical forecast verification. The preview requires a fresh measurement (within 6h) and sufficiently complete lag history. It is not connected to the official benchmark or public demo.

## What the diagnostic tests

At **12:00 UTC each day**, simulate a new 24-hour forecast with 96 nonoverlapping 15-minute target slots. The reference at each target is the demand 48 hours ago, 7 days ago, or a half-and-half blend. All target lag values are at least 24 hours older than the simulated issuance.

To estimate a current level shift, compare the reference and actual demand for the preceding 6 hours, **ending 60 minutes before the issuance**. Take the median signed error, cap it at ±1,500 MW, and add a fraction of it to future reference values with exponential decay:

`candidate(target) = max(0, reference(target) + alpha * capped_anchor_error * exp(-lead_hours / decay_hours))`

The correction strength `alpha` is selected from `{0, 0.25, 0.5, 0.75, 1}`; decay is either 6h or 24h. The no-correction candidate is always eligible. Ties favour no correction.

- **January–March 2026:** Select baseline/alpha/decay using MAE on shared observed targets.
- **April–June 2026:** Check the selected rule on subsequent daily origins without selecting a different rule based on the result.
- Skip days with fewer than 16 usable anchor observations or 80 usable future target observations.
- Exclude missing target readings from *every* method being compared. The data sources remain unchanged.
- Report MAE, RMSE, bias, measured-pair counts, exclusions and monthly diagnostics.

## Limitations

This is a **simulation of forecast issuance**, not a record of an actual live issuance. Historical availability is represented by a conservative one-hour cutoff; exact Fingrid publication latency is not independently established. The April–June period has already been viewed in earlier GridWise development, so it is not a new, untouched test. Daily 24-hour windows have non-overlapping target timestamps, but neighbouring days and errors remain temporally correlated.

A lower MAE on April–June would justify additional forward trials; it **would not** establish live performance, and it would not make the candidate an operational prediction. A higher MAE is also a useful result: it means this particular correction didn't generalize.

The diagnostic is independent of weather, official Fingrid forecast snapshots, and the 310 MW fixed-24h-lead backtest (different prediction task). The local dashboard displays its results; the public Render demo remains archived and unchanged.
