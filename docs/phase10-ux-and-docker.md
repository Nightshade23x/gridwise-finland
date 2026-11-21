# Phase 10 — chart polish and local Docker packaging

## Dashboard improvements

- UTC x-axis automatically switches to hours for a single-day view and uses distinct source points for tick positions, fixing repeated date labels on short daily charts.
- Hover shows the precise UTC time for 15-minute/hourly data, or the UTC day for daily aggregates.
- **Export visible CSV** downloads the selected view and chart resolution, preserving missing values as blank cells. Period insights continue to use the underlying 15-minute data, not aggregated chart values.
- The dashboard remains a historical backtest, not a live forecast.

## Run locally (no Docker)

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn gridwise.api:app --reload --host 127.0.0.1 --port 8000
```

## Run in Docker Desktop

First generate `data/processed/consumption_2024_2026.csv` and `data/model_outputs/lightgbm_predictions_2026.csv` with the earlier GridWise commands. Docker Compose mounts the existing local `data/processed` and `data/model_outputs` folders **read-only**. They are not copied into the image.

```powershell
docker compose up --build -d
docker compose ps
docker compose logs --tail=30
docker compose down
```

Open http://127.0.0.1:8000 or http://127.0.0.1:8000/docs. The container uses a non-root user; it binds to `127.0.0.1` on the host, so it is not a public deployment. API keys and `.env` files are excluded from the build context. No model training runs inside the container.

A regular pip-installed package resolves its data root with `GRIDWISE_PROJECT_ROOT=/app` in Compose. The environment variable is optional for editable local development.

### Next step

Move to public hosting only after planning a data-refresh schedule, validating actual publication latency, and providing updated forward-looking forecasts. Never present retrospective predictions as a live service.
