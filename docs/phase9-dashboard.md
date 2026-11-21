# Phase 9 — Local dashboard and FastAPI

Install the optional web dependencies in the existing GridWise virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml,app]"
.\.venv\Scripts\python.exe -m pytest -q
```

Start the web application from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn gridwise.api:app --reload --host 127.0.0.1 --port 8000
```

- Dashboard: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/api/health
- Summary: http://127.0.0.1:8000/api/summary
- Examples:
  - `/api/series?kind=test&start=2026-07-01&end=2026-07-08&interval=hour`
  - `/api/series?kind=history&start=2024-01-01&end=2024-01-08&interval=day`

The API is **read-only** and reads the existing `data/processed/consumption_2024_2026.csv` and `data/model_outputs/lightgbm_predictions_2026.csv`. It optionally loads baseline predictions for comparisons. Generate these artifacts with the existing `gridwise.prepare`, `gridwise.baselines`, and `gridwise.forecast_ml` commands if necessary.

This dashboard shows historical demand and retrospective July–October 2026 model evaluations. **It is not real-time and does not issue new forecasts.** The evaluation set has already been inspected during development. Do not present these results as a new untouched holdout or live deployment performance. A production service will need reliable observation availability, point-in-time weather forecasts, monitoring, and a fresh evaluation period.

The date-range end is exclusive; long 15-minute series are limited to 3,000 points per request. The dashboard has no external JavaScript dependencies. The server is intended for local development; bind only to `127.0.0.1`, not a public interface, until deployment hardening is complete.
