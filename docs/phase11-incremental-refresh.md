# Phase 11 — Incremental electricity demand refresh

## What this adds

A one-shot API refresh job, a separate `data/live/consumption_recent.csv` store, an
atomic status report and the **Recent measurements** dashboard view. It is **not** a
live forecast or an updated version of the original backtesting dataset.

Source: Fingrid, [Electricity consumption in Finland, dataset 124](https://data.fingrid.fi/en/datasets/124), CC BY 4.0. Observations are 15-minute mean electricity consumption (MWh/h = MW numerically). The latest published observation may lag the current time. The job uses a **60-minute buffer as an assumed safety margin**, not a verified publication SLA.

## First run (Windows PowerShell)

From the project directory:

```powershell
New-Item -ItemType Directory -Force data/live | Out-Null
.\.venv\Scripts\python.exe -m gridwise.refresh
```

Your existing `.env` must contain `FINGRID_API_KEY`. This command:

- Starts at **2026-11-01 00:00 UTC** on first run, then fetches only a two-day overlap around the most recent saved measurement through the latest completed, buffered quarter hour.
- Uses the existing 3-second API throttle/retry client and chunks requests into seven-day windows.
- Rejects off-grid/incomplete intervals, never guesses missing measurements, never modifies raw historic CSVs or model predictions, and keeps stored values at conflicting timestamps.
- Writes `data/live/consumption_recent.csv` and `data/live/refresh_status.json` atomically.
- Keeps the last good CSV when the API fails and records an error status.

Run it again to verify the second call doesn't duplicate records:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
```

Freshness, not actual publication delay, is displayed by the dashboard. A failure status does not automatically imply the measurements are wrong. If the process is terminated mid-download, the CSV stays intact. If a crash leaves `data/live/refresh.lock`, remove it **only after confirming no refresh is running**.

## Dashboard & API

The existing dashboard still runs with:

```powershell
.\.venv\Scripts\python.exe -m uvicorn gridwise.api:app --reload --host 127.0.0.1 --port 8000
```

Visit `/api/refresh-status`, or select **Recent measurements** in the dashboard.
For example:

`/api/series?kind=recent&start=2026-11-01&end=2026-11-22&interval=hour`

Dates are UTC, `end` is exclusive. No endpoint exposes the Fingrid API key, triggers a data refresh, or advertises a live forecast.

## Docker

Start the dashboard as before (`docker compose up --build -d`). The dashboard container reads `data/live` as a read-only bind mount. After setting `FINGRID_API_KEY` in `.env`, a refresh can also be run in a separate one-shot container:

```powershell
docker compose --profile refresh run --build --rm refresh
```

This worker alone has the key and read-write access to `data/live`. It does not restart the dashboard. Do **not** publish your `.env` or embed its key into a container image.

## Optional recurring schedule on Windows

For the laptop-local workflow, use **Windows Task Scheduler**, creating a task that runs every 3 hours with program `powershell.exe` and arguments:

`-NoProfile -ExecutionPolicy Bypass -File "D:\Projects\gridwise-finland-starter\gridwise-finland\scripts\refresh-gridwise.ps1"`

Choose **Run only when user is logged on** for your first local experiment. It runs when Windows is awake and the task is enabled; this is *not* a cloud-hosted always-on scheduler. The script uses your existing Python virtual environment and `.env`. View task history for failures. Do not run the Task Scheduler job and Docker refresh simultaneously.

## Known scope

- Only historical/current observations are refreshed; no new model inference is run.
- Existing 2024–2026 prepared history/backtest predictions are deliberately untouched.
- The source can revise previously published values; collisions are **counted**, not overwritten. A future reconciliation workflow can review revisions explicitly.
- The first run may include genuine missing intervals; check `last_7_days` in the status JSON.
- Future public deployments need an external scheduler and durable writable storage.
