# GridWise Finland

A Python-based electricity-demand forecasting and monitoring project using Finnish grid data. GridWise collects observations, validates gaps and unusual timestamps, benchmarks time-series forecasts, issues experimental next-day predictions, and evaluates saved predictions against later measurements. A FastAPI dashboard provides interactive charts and descriptive insights.

**Status:** Research prototype with a working local Docker dashboard. The optional public deployment is a **read-only historical snapshot**, not a live electricity forecast or operational grid advice.

## Highlights

- **Data engineering:** incremental ingestion from Fingrid, API rate-limit retries, missing-data audits, and reproducible 15-minute UTC preprocessing.
- **Forecasting:** seasonal baselines and LightGBM with calendar, lagged-demand and weather features; time-ordered validation.
- **Verification:** immutable forward-forecast archives, same-timestamp MAE/RMSE comparisons, forecast-horizon errors, and snapshots of official Fingrid demand predictions.
- **Application:** FastAPI JSON endpoints, interactive dark-themed dashboard with per-filter insights and CSV export, Docker packaging, and automated tests.
- **Safety:** the HTTP server is read-only; API secrets, model training, downloads, and refresh commands stay outside the web process.

## Historical results

| Experiment | Evaluation | MAE |
|---|---|---:|
| LightGBM with lagged demand and delayed weather | July–October 2026 historical test | **310.02 MW** |
| Yesterday persistence (validation-selected baseline) | Same 11,803 paired timestamps | 397.05 MW |
| 50/50 yesterday + last-week blend | Same paired timestamps | 340.56 MW |

This historical experiment improved MAE **21.92%** relative to the validation-selected yesterday baseline. The newer *operational* 24-hour model is **separate** and must not be described as achieving that score. Phase 16's weather-aware candidate improved historical validation MAE from **824.3 MW** to **574.4 MW**, but its forward performance must be independently verified.

Results and saved archives are research diagnostics, not validated future performance guarantees. The July–October historical test window has also been examined during development; it is no longer an untouched test set.

## Project layout

```text
src/gridwise/               Fingrid ingestion, preprocessing, models, API and dashboard
scripts/                    Optional local refresh script and public-demo exporter
tests/                      Unit and regression tests
Dockerfile, compose.yaml    Local read-only web server and opt-in refresh worker
Dockerfile.demo             Portable read-only static demo image
render.yaml                 Optional Render public demo blueprint
docs/                       Phase notes and methodology
```

Historical data, current measurements, and saved prediction archives live under `data/` and are excluded from normal source-code commits.

## Local setup (Windows PowerShell)

```powershell
cd D:\Projects\gridwise-finland-starter\gridwise-finland
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml,app]"
.\.venv\Scripts\python.exe -m pytest -q
```

For fetching data, obtain a personal Fingrid Open Data API key and create `.env` from `.env.example`. Keep the key private; **never commit `.env`**.

The main local workflow uses existing data and modules:

```powershell
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.forward
.\.venv\Scripts\python.exe -m gridwise.weather_verify
```

These steps rely on historical preprocessing and trained-model inputs from the earlier project phases. Consult `docs/` for the complete reproducible sequence and the official-comparison commands. No scheduled jobs are enabled by default.

Run the local dashboard:

```powershell
docker compose up --build -d
docker compose ps
```

Open <http://127.0.0.1:8000> or the interactive API specification at <http://127.0.0.1:8000/docs>. Docker Compose mounts the existing processed measurements and model outputs read-only. Stop it with `docker compose down`.

## Optional public portfolio demo (Phase 18)

The normal Docker image needs local CSV files that Git intentionally ignores. To deploy a **static historical demo**, generate a small allow-listed bundle from your *real* local results:

```powershell
.\.venv\Scripts\python.exe scripts/export_public_demo.py
```

Review `demo_bundle/` before committing it. Only selected historical source measurements and retrospective predictions are exported; **no API key, raw downloads, forward issuance archive or live refresh data** is copied. The complete historical July–October test set is included to avoid cherry-picking favourable weeks. The hosted interface prominently identifies the data as archived and hides unavailable live/verification panels.

Test the deployment image locally:

```powershell
docker build -f Dockerfile.demo -t gridwise-public-demo .
docker run --rm -p 8001:10000 gridwise-public-demo
```

Open <http://127.0.0.1:8001> and <http://127.0.0.1:8001/api/health>. Once checked, commit the reviewed `demo_bundle/data/` and `demo_bundle/manifest.json` to the public repository. You can then deploy using `render.yaml` on Render. The free plan may sleep when idle and **does not persist runtime file changes**; it must be treated as a snapshot only. See [Phase 18 deployment instructions](docs/phase18-public-demo.md).

## Sources and attribution

- [Fingrid Open Data — consumption dataset 124](https://data.fingrid.fi/en/datasets/124), CC BY 4.0. Fingrid is the original measurement provider. GridWise is independent and not endorsed by Fingrid.
- [Fingrid Open Data](https://data.fingrid.fi/en/) official forecast datasets are captured as time-specific local research evidence, not reconstructed issuance records.
- [Open-Meteo](https://open-meteo.com/) historical reanalysis and previous model-run weather inputs, subject to their applicable terms.

Data-quality gaps are preserved and documented. All forecast timestamps are processed in UTC; local calendar features use Finnish time when appropriate.

**Live Demo:** https://gridwise-finland-demo.onrender.com/

The public demo uses archived historical data.
Live data ingestion and experimental forward forecasts
are maintained separately.