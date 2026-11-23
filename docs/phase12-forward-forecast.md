# Phase 12: Experimental next-24-hour demand forecasts

This milestone adds **forward-looking predictions** to the Phase 11 GridWise dashboard, without changing any previous model, reports, raw data, or retrospective metrics.

## Windows PowerShell commands

Use the existing virtual environment with the ML extras:

```powershell
cd D:\Projects\gridwise-finland-starter\gridwise-finland
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml,app]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.forward
```

Refresh should be run before forecasting. It needs `FINGRID_API_KEY` in the local `.env`. The forward worker itself makes **no external API calls** and requires no credentials.

Outputs are generated locally (and ignored by Git):

- `data/live/forward_forecast_24h.json` – dashboard/read-only API artifact
- `data/live/forward_forecast_24h.csv` – forecast with selected historical baseline
- `data/model_outputs/forward_24h_model.txt` – new trained LightGBM model

Restart Docker to serve the newly added API and dashboard files:

```powershell
docker compose down
docker compose up --build -d
docker compose ps
```

View `http://127.0.0.1:8000` and `http://127.0.0.1:8000/api/forward-forecast`.
The dashboard container only **reads** `data/live/`; it does not fetch Fingrid data, retrain models, or access the API key.

## Technical contract and limitations

- A single forecast issue produces 96 targets, one every 15 minutes, beginning at the next UTC quarter-hour and ending 23h45 later. This is a **next-24-hour forecast window**, **not a fixed 24-hour lead time for each target**.
- Retrained LightGBM features use demand from **target minus 48 hours or earlier**, which means that for the entire forward window, they predate the common forecast issue by at least 24 hours. A safety check enforces this. Target calendar features are known in advance.
- The newest Phase 11 demand measurement must be no older than six hours. A minimum 60-minute buffer excludes more recent raw readings from model fitting. If essential features are unavailable, the worker refuses to generate the forecast rather than silently imputing historical consumption.
- Weather is **not** used in this first forward model. The earlier weather-augmented fixed-lead-time LightGBM scored around 310 MW on a previously inspected Jul–Oct 2026 historical test, but that result **does not transfer** to this model.
- Training is fit using measured data available at issue. Candidate benchmarking uses 2024–2025 training and Jan–Jun 2026 validation. A separate model fitted before Nov 1 evaluates the newly available November rows (if at least 96 rows are eligible). The deployment model is then refitted on all available data; it cannot be evaluated using those same November rows without leakage.
- The forward baseline is selected by validation MAE among 48h persistence, one-week persistence, and their equal-weight blend. All predictions and their metadata are saved as auditable files.
- The forward forecast will become stale unless the standalone refresh + forecast workers are rerun. This milestone does **not** turn on Task Scheduler, automated jobs, or public hosting.
- This research prediction is not certified for electricity grid operations, balancing, trading, or household cost savings. Future milestones should integrate genuine point-in-time weather forecasts and assess predictions after the target times have passed.

## If the worker says data is stale

Run `python -m gridwise.refresh` first. If the API is delayed or unavailable, wait until new observations exist; do not bypass freshness safeguards. Inspect `data/live/refresh_status.json` locally without sharing your API key.

## Testing

Run `python -m pytest -q`. New Phase 12 tests verify prediction shape, stale-data refusal, forecast-input time constraints, missing-feature refusal, historical/recent merge behavior, and read-only API validation. Tests are offline and use synthetic data.
