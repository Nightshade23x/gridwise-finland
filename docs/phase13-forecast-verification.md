# Phase 13 — Verifying forward forecasts

Phase 12 generates 96 **future** 15-minute demand predictions per issuance. Phase 13 archives each issuance and compares those *unchanged* predictions to the **later** observations collected by Phase 11.

## Run locally (Windows PowerShell)

From the repository root (with the existing virtual environment):

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m gridwise.verify
```

The first invocation imports the previous `data/live/forward_forecast_24h.json` into a content-addressed archive, if it exists. No forecasting retraining, network request, or changes to predicted values occur during verification. When the forecast's targets haven't been measured yet, the status is `pending` and MAE/RMSE/bias are **null**, not zero.

After some time has passed, update actual measurements and re-verify:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.verify
```

To issue another forecast, use `python -m gridwise.forward` after refreshing. The forward worker now **archives the previous snapshot first**, then archives the new predictions before updating the latest-forecast file. It does not overwrite archived issue snapshots.

The optional `--availability-buffer-minutes 60` defaults to 60 minutes; verification only scores measurements old enough to clear that availability buffer. This is a conservative assumption, **not** a measured Fingrid publication latency.

## Saved files

- `data/live/forecast_archive/<timestamp>-<hash>.json` — archived original issuance payloads; each filename includes a SHA-256 content digest. The verifier checks these hashes on every run. Local files are not externally timestamp-certified; manual deletion/replacement cannot be ruled out.
- `data/live/forecast_verification.json` — reproducible latest verification summary, rewritten atomically with each run.
- `data/live/consumption_recent.csv` — source-of-truth for actual measured demand; raw and historical training files are never changed by verification.

These are ignored by Git because they are generated or continuously updated.

## What counts as a match?

Timestamps are matched **exactly in UTC** at the original 15-minute resolution. Missing measurements stay missing. All model and reference-baseline errors are computed on the **same matching timestamps**, so a model doesn't get extra or fewer observations. Scores include MAE, RMSE, mean signed error (`prediction − actual`, positive means overprediction), and the 90th percentile absolute error. The result also includes 0–6, 6–12, 12–18, and 18–24-hour buckets measured from the first forecasted slot.

Each issuance is evaluated **separately**. Multiple forecast runs can predict overlapping times, so we do not pool all predictions into a misleading independent-sample count.

### Status meanings

- `pending`: no target is old enough to verify.
- `partial`: some targets are eligible but the forecast window is not fully elapsed; metrics describe only measured targets.
- `incomplete_source_data`: the entire window is eligible, but at least one source observation is still missing.
- `complete`: all 96 targets have matching eligible source measurements.

A partial MAE should **not** be compared directly to a complete 24-hour MAE. If the last forecast is still pending, the dashboard displays that fact rather than guessing its accuracy.

## Dashboard and API

Run or rebuild the existing web app:

```powershell
docker compose down
docker compose up --build -d
```

Open `http://127.0.0.1:8000` or `http://127.0.0.1:8000/api/forecast-verification`. The app is read-only and displays the stored verifier report. It never executes model training, uses the Fingrid API key, or makes outbound verification requests.

To see new verification scores, run `gridwise.refresh`, then `gridwise.verify`, and press **Reload verification** on the dashboard. Refresh scheduling is **not** enabled by this phase.

## Important limitations

- These are experimental forecasts; they are not operational grid advice.
- The first Phase 12 forecast can only be imported from the latest artifact if it still exists. Older predictions overwritten *before* Phase 13 cannot be recovered.
- The local archive is content-checkable, but it does not provide third-party proof of the original issuance time.
- The historical backtest MAE (~310 MW) belongs to a **different model**; do not present it as live forward-forecast accuracy.
- Fingrid data may contain gaps and possible source revisions. The Phase 11 store preserves existing measurements on conflict, and verification uses that stored snapshot.
