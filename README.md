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
