# Phase 15: Conservative forward forecast choice and weather snapshots

This patch follows the official Fingrid benchmark capture in Phase 14.

## What changed

- The forward worker uses a **validation MAE gate** to identify whether LightGBM or the best historical reference should be preferred. The 2026 January–June validation comparison is made before considering the newer November measurements. On a tie, the reference wins. This is a simple guard, not a proof that the chosen strategy will win on unseen dates.
- The two original forecast series are preserved exactly: `predicted_mw` = LightGBM, `baseline_mw` = reference. The new `preferred_mw` is a **separate** field, with `preferred_strategy` metadata. Historical verification and the Fingrid benchmark still compare LightGBM, the reference, and the official forecast on matched timestamps. Old archived forecasts remain valid.
- Forecast first target is strictly **after** the issuance, even when issuance falls exactly on a quarter-hour boundary.
- The dashboard states which method won validation; it does not disguise a reference forecast as ML.
- The new weather worker captures **actual future weather forecast API responses** for Helsinki, Turku, Tampere, Oulu, and Rovaniemi. It saves 24 future UTC hourly temperatures per location, their capture completion time, and source metadata, under `data/live/weather_forecast_archive/` and `data/live/weather_forecast_latest.json`. It never writes an incomplete five-city snapshot on API failure. The data stay out of Git.

## Commands (Windows PowerShell)

```powershell
cd D:\Projects\gridwise-finland-starter\gridwise-finland
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m gridwise.refresh
.\.venv\Scripts\python.exe -m gridwise.forward
.\.venv\Scripts\python.exe -m gridwise.weather_forecast
docker compose down
docker compose up --build -d
```

`gridwise.forward` trains a new forward LightGBM candidate and evaluates validation baselines, saves a comparison, archives the issuance, and reports the validation preference. The new weather forecasts are **not yet features in the LightGBM model**. Do not claim any weather-related improvement from this phase.

## Validity and limitations

- The retrospective historical 310 MW MAE came from a different model and forecast definition; it is **not** the expected live error.
- The Phase 15 preferred strategy is selected using old validation data. Its performance during a seasonally different November is not guaranteed. Continue to log actual subsequent errors, including both candidate curves, before changing how the choice is made.
- Point-in-time Open-Meteo responses are stored when downloaded. This is *not* an authenticated source-issued model run or a verified historical model issuance. Best-match model selection can change. For rigorous research, use fixed model-run archives and enforce model-run issue time <= demand forecast issue time.
- We deliberately do not use historical ERA5 **observed/reanalysis** values as future forecast features.
- No background scheduler is enabled.

Sources: [Open-Meteo Forecast API](https://open-meteo.com/en/docs), [Single Runs archives](https://open-meteo.com/en/docs/single-runs-api).
