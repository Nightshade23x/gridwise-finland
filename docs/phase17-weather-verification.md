# Phase 17 — Verifying weather-aware forward forecasts

The Phase 16 `gridwise.weather_forward` command archives each newly issued 96-point candidate under `data/live/weather_model_comparison_archive/`. Phase 17 reads **those already-saved predictions** and matches them with **later Fingrid consumption measurements**. No archived predictions, historical backtests, models, or official Fingrid snapshots are changed.

## Run locally (Windows PowerShell)

From `D:\Projects\gridwise-finland-starter\gridwise-finland`:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.weather_verify
```

Phase 17 does **not** need you to issue another candidate first: the November 24 candidate, if saved under the Phase 16 archive, is used automatically. If no candidate exists yet, generate one using `python -m gridwise.weather_forward` after downloading its weather source.

When new measurements arrive, repeat only `gridwise.refresh` and `gridwise.weather_verify`. The verifier is safe to rerun: it updates `data/live/weather_forecast_verification.json` atomically and never overwrites a forecast archive. Scheduling remains disabled.

Rebuild the dashboard once after installing the code:

```powershell
docker compose down
docker compose up --build -d
docker compose ps
```

The new section **Did weather improve real forecasts?** reads `GET /api/weather-forecast-verification` (read-only). Click **Reload weather verification** only after running the verification command. The **Last recalculated** label indicates the time of the latest saved report; it is not a measurement-download time.

## What the report means

- Every candidate has 96 targets and an issuance time strictly before the first target.
- Weather-aware, demand-only, validation-preferred, and historical-baseline predictions are evaluated on **exactly the same** measured timestamps.
- A target is mature only after its 15-minute measurement interval ends **plus a 60-minute safety buffer** (configurable using `--availability-buffer-minutes`). This is an assumption, not a measured Fingrid publication latency.
- Missing or not-yet-available measurements are **not** interpolated or scored as zero error.
- `pending`: no target is old enough; `partial`: some targets are old enough but the horizon is unfinished; `complete`: all 96 targets have readings; `incomplete_source_data`: the horizon is mature but at least one reading remains missing.
- Metrics: MAE, RMSE, mean signed bias (`forecast - actual`), and 90th-percentile absolute error in MW. Four six-hour buckets show how performance changes with forecast horizon.
- Repeated issuances over similar target times are **not pooled**. Each is independently scored and shown in the archive history.
- The saved candidate contains its issued predictions; a SHA-256 identifier in the report allows distinguishing local snapshots. It is **not** a trustworthy external timestamp certificate.

This is operational *verification*, not another retrospective validation score. The weather candidate remains separate from the Phase 12 main forward forecast and Phase 14 official benchmark. It is too soon to generalize from a single forecast day; collect multiple forward issues before claiming weather improves live performance.

The dashboard does not contact Fingrid, Open-Meteo, or retrain any model; only the local CLI programs do so when explicitly run. Do not commit `data/live/`, API keys, or weather datasets.
