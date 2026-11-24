# GridWise Finland

A data-driven platform for forecasting Finland's electricity demand and identifying smarter electricity-use windows. Built with publicly available data from Fingrid. Initially focused on a reproducible ingestion pipeline.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
```

Get a free Fingrid API key following https://data.fingrid.fi/en/instructions and place it in `.env` as `FINGRID_API_KEY=...`. Never commit the key.

## Download data

```powershell
.\.venv\Scripts\python.exe -m gridwise.ingest --start 2026-10-01 --end 2026-11-01
```

Results: `data/raw/fingrid_consumption_124.csv` (UTC timestamps, measured 15-minute mean power in MWh/h, numerically equivalent to average MW).

Change `--start` and `--end` to request other date ranges. The end is exclusive. Downloading the same range again safely overwrites its local output; it does **not** merge with an earlier date range. To combine periods, request one wider range.

## Inspect the data

```powershell
.\.venv\Scripts\python.exe -m gridwise.inspect
.\.venv\Scripts\python.exe -m pytest -q
```

The inspection command prints a summary and writes `data/consumption_daily.png`.

## Data attribution

Source: [Fingrid Open Data](https://data.fingrid.fi/en/datasets/124), dataset 124, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Fingrid data is provided by Fingrid Oyj; this project is independent of Fingrid.

## Roadmap

1. Ingest and validate demand observations.
2. Add weather data, create temporal features, and measure simple seasonal baselines.
3. Train and backtest forecasting models with chronological splits.
4. Add price/carbon signals, scheduling recommendations, and a deployed API/dashboard.

## Refresh newer measured consumption (Phase 11)

The refresh worker is separate from model training and does not modify any historical model inputs:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
```

With the dashboard in Docker, the one-shot equivalent is:

```powershell
docker compose --profile refresh run --build --rm refresh
```

Refresh status and recent observations are available through the web dashboard and `/api/refresh-status`. See [Phase 11 instructions](docs/phase11-incremental-refresh.md) for setup, limits, and optional Windows scheduling.

## Experimental next-24-hour forecasts (Phase 12)

After refreshing recent Fingrid observations, generate a *separate*, conservative forward-forecast model:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml,app]"
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.forward
```

The dashboard serves the resulting forecast through `/api/forward-forecast` and labels it separately from the earlier historical backtest. The new model uses only demand data from target time minus 48h or earlier, so all 96 predictions can be generated at one issuance. It does **not** inherit the previous model's 310 MW MAE. See [Phase 12 notes](docs/phase12-forward-forecast.md).

## Phase 15: forecast selection and weather snapshots

Forward forecasts now identify which candidate had lower pre-November validation MAE.
The saved `predicted_mw` values are **always LightGBM**, and `baseline_mw`
is always the historical reference. `preferred_mw` is the candidate favoured by
validation, never a claim of proven live superiority. Earlier archived forecasts
remain unchanged.

Capture Open-Meteo forecasts *at retrieval time* to create a point-in-time
archive for future weather-aware training; this does not alter demand forecasts:

```powershell
.\.venv\Scripts\python.exe -m gridwise.weather_forecast
```

See `docs/phase15-forward-safety-and-weather.md`.
