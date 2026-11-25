# Phase 18 — Warning cleanup, CI and public historical demo

## What changed

- Replaced `pandas.Timedelta(...)` construction for fixed time durations with standard-library `datetime.timedelta(...)` in the affected modules and test fixtures, avoiding the recurring NumPy generic-timedelta deprecation warnings. This does not change timestamp semantics or the model.
- GitHub Actions now installs the **full app and ML extras** (`.[dev,ml,app]`) and tests Python 3.11 and 3.12. Prior CI only installed `.[dev]` even though the tests import FastAPI and LightGBM.
- Added a **real-data-only historical demo exporter**, read-only Docker image and optional Render service blueprint.
- Added a demo-mode API and banner explaining that hosted data and scores are archived, not live; hides unavailable live forecast panels.

## 1. Test the existing application

```powershell
cd D:\Projects\gridwise-finland-starter\gridwise-finland
Expand-Archive "$env:USERPROFILE\Downloads\gridwise-phase18-public-demo.zip" -DestinationPath . -Force
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ml,app]"
.\.venv\Scripts\python.exe -m pytest -q

docker compose down
docker compose up --build -d
docker compose ps
```

Expected: local dashboard still displays historical and forward sections; no models are retrained by installing this patch. Changes may leave an external-library deprecation warning depending on your installed FastAPI/Starlette version; this is distinct from the fixed Pandas/NumPy warnings.

## 2. Generate the demo data (explicit action)

```powershell
.\.venv\Scripts\python.exe scripts/export_public_demo.py
Get-ChildItem demo_bundle -Recurse
```

The exporter **fails** if the original CSVs are absent, timestamps are invalid or the historical test output is too small. It copies only:

- `data/processed/consumption_2024_2026.csv`: January 1–7 of 2024 and 2025 plus July–October 2026.
- `data/model_outputs/lightgbm_predictions_2026.csv`: the complete July–October 2026 historical test window.
- `data/model_outputs/baseline_predictions_2026.csv`: corresponding historical baselines.
- `manifest.json`: source attribution and the actual export time.

It deliberately does **not** copy `.env`, `.venv`, `data/live`, raw downloads, model binaries or forecast issuance archives. Review all generated CSVs because they will become public if you stage/commit them.

## 3. Test the public image locally

Run from the same repo directory, with Docker Desktop open:

```powershell
docker build -f Dockerfile.demo -t gridwise-public-demo .
docker run --rm -p 8001:10000 gridwise-public-demo
```

Open `http://127.0.0.1:8001` and `http://127.0.0.1:8001/api/health`. The dashboard should show an **Archived demo** banner and historical interactive charts only, not a live forecast. Exit the foreground container with Ctrl+C.

On an existing Docker Compose installation port 8000 remains unaffected; the demo runs on port 8001.

## 4. Commit after inspection

```powershell
git status --short
git add .
git status --short
git diff --cached --stat
```

Ensure `.env`, API keys and raw full datasets are absent, and that the reviewed `demo_bundle` CSVs are included. Commit and push only when you have verified the exported content and the Docker demo.

## 5. Optional Render public deployment

[Render's Blueprint format](https://render.com/docs/blueprint-spec) supports Docker builds, a free web plan and `/api/health` checks. From the Render dashboard create a Blueprint from the GitHub repository, selecting `render.yaml`. The `Dockerfile.demo` binds Uvicorn to Render's `$PORT` and includes only the small archived demo dataset. No API credentials are needed. **This step requires you to connect your GitHub account to Render and choose to deploy; it has not been done automatically.**

The free plan has an ephemeral filesystem and may spin down after inactivity. It cannot serve as durable storage for newly measured data or forecast archives. Public deployment of a genuinely updated forecast service would require persistent storage and a separately scheduled ingestion/verification worker. Keep public demo claims about model accuracy clearly labelled as retrospective.
