# Phase 22 — Level-shift shadow forecasting

This phase evaluates Phase 21's capped demand-level correction on **newly issued 24-hour forecasts** without changing GridWise's operational predictions or the public historical demo.

## Workflow (local PowerShell)

Run in the repository root with the virtual environment installed:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.level_shift_shadow issue
.\.venv\Scripts\python.exe -m gridwise.level_shift_shadow verify
```

The `issue` command reads `data/live/level_shift_backtest.json` from Phase 21, **freezes its preselected parameters**, and produces two series for the next 96 quarter-hour targets: uncorrected historical baseline and the capped, decaying level-shift correction. It refuses stale or insufficient measurements. The correction uses demand that is at least 60 minutes old and reference inputs at least 48 hours before each target. It **does not** retrain LightGBM or overwrite any previous forward forecast, official Fingrid capture, or weather candidate.

Each issuance is stored as a separate, content-addressed JSON snapshot in `data/live/level_shift_shadow_archive/`. The archive and report are ignored by Git. Snapshots are local research evidence, **not externally certified timestamps**. Do not retroactively issue forecasts to claim prospective accuracy.

`verify` never calls Fingrid and never creates a new forecast. It reads existing snapshots and `data/live/consumption_recent.csv` and saves `data/live/level_shift_shadow_verification.json`. A measured interval is eligible only after its 15-minute period **ends** plus an additional 60-minute buffer. Missing actual measurements remain missing; both strategies are scored on identical eligible timestamps. Results are reported per issuance and in 0–6, 6–12, 12–18, and 18–24 hour blocks, with MAE, RMSE and signed bias. Status is `pending`, `partial`, `complete`, or `incomplete_source_data`. A negative MAE reduction means the correction was worse.

To update the scores after new observations arrive:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.level_shift_shadow verify
```

Restart the local Docker dashboard **once** after installing Phase 22:

```powershell
docker compose down
docker compose up --build -d
```

Then open `http://127.0.0.1:8000` and inspect **Does the correction work on new forecasts?** The `Reload shadow verification` control merely rereads the report. API endpoint: `GET /api/level-shift-shadow` (read-only). It is disabled in the static public Render demo.

## Evaluation policy

The 25.71% MAE reduction from April–June 2026 was a previously studied development diagnostic, **not** a new forward-live result. Phase 22 is designed to collect prospective evidence without tuning on the resulting outcomes. Multiple issuances with overlapping targets are not independent samples. Keep the calibration configuration fixed while comparing future issued windows; collect several nonoverlapping days and evaluate the new records before considering promotion.

The `issue` and `verify` commands are manual; no scheduling was enabled.
