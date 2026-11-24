# Phase 16 — Weather-aware 24-hour candidate (experimental)

## Purpose

Compare three **issuance-safe** forward approaches: an existing historical-demand baseline, demand-only LightGBM, and LightGBM with archived **48-hour-lead forecast temperatures** for Helsinki, Turku, Tampere, Oulu and Rovaniemi. All three are evaluated on the **same Jan–Jun 2026 validation timestamps**. The validation winner may be a simple baseline; no candidate is promoted automatically.

**Not a live-accuracy claim.** This phase does **not** modify `forward_forecast_24h.json`, any Phase 13/14 verification archive or Fingrid's official forecast snapshots. Model selection still needs independent prospective verification before production promotion.

## Weather source and available information

Open-Meteo **Previous Model Runs API**:

- Endpoint: `https://previous-runs-api.open-meteo.com/v1/forecast`
- Hourly variable: `temperature_2m_previous_day2`
- Time zone: **UTC**; temperature unit: **°C**
- Meaning: each temperature is a prediction with a *nominal* 48-hour lead relative to its **valid** time, rather than a later-observed or reanalysed temperature.
- For a forecast covering the 24 hours immediately after issuance, the latest weather forecast origin is at least ~24 hours before issuance. This offers a conservative allowance for weather-model publication delays. Open-Meteo does **not** certify exact historical publication times for individual values in this API.
- Historical forecasts are available from around 2024 for most models, but actual coverage may vary. Missing or malformed entries are never silently converted to observed weather.

References: https://open-meteo.com/en/docs/previous-runs-api and https://open-meteo.com/en/docs/single-runs-api

## Commands — Windows PowerShell

From the existing project root:

```powershell
cd D:\Projects\gridwise-finland-starter\gridwise-finland

# Run all tests after applying the Phase 16 ZIP
.\.venv\Scripts\python.exe -m pytest -q

# Download 48-hour-lead weather forecast history for training and next-day targets.
# On later dates, replace --end with a date at least two days ahead.
.\.venv\Scripts\python.exe -m gridwise.weather_previous --start 2024-01-01 --end 2026-11-26

# Refresh measured demand (already installed in Phase 11)
.\.venv\Scripts\python.exe -m gridwise.refresh

# Build and archive the separately-issued weather candidate
.\.venv\Scripts\python.exe -m gridwise.weather_forward

# Rebuild local dashboard and inspect the Phase 16 panel
docker compose up --build -d
```

The download makes sequential requests in chunks of at most 90 days per city, retries temporary errors, and validates schema/time alignment. If the source returns missing forecast values on necessary future hours, the experiment refuses to issue predictions: report the output rather than substituting ERA5 or using actual future weather.

The source data CSV is saved as `data/raw/open_meteo_forecast_48h.csv`. It is excluded from Git along with other downloaded datasets.

New output artifacts:

- `data/live/weather_model_comparison_24h.json` — latest separate candidate, for the dashboard
- `data/live/weather_model_comparison_archive/*.json` — preserved earlier candidates
- `data/live/weather_model_comparison_24h.csv` — all 96 predictions
- `data/model_outputs/weather_forward_candidate.txt` — saved weather candidate model
- `/api/weather-model-comparison` — read-only API endpoint

These generated files are excluded from Git. The API server does not contact the external weather API, train models, or access secrets.

## Model development rules

1. For every 15-minute target, demand inputs end at least 48h earlier; weather input is a forecast for the target hour created approximately 48h earlier.
2. No future targets enter model training. The separate evaluation uses training data before Jan 2026 and validation observations from Jan–Jun 2026.
3. All candidate validation errors are computed on **identical complete-case** observations, ensuring a fair comparison even when weather history has gaps.
4. The displayed experiment is **candidate only**. Weather may improve or worsen MAE. The validation winner is informational, not deployed.
5. Publication-time history is inferred from lead-time semantics rather than independently certified. The next phase should accumulate true issuance snapshots and score them against subsequently published Fingrid measurements.

## If something goes wrong

- **HTTP 429/5xx:** The weather downloader retries automatically; report the final output if it persists.
- **Insufficient historical forecast coverage:** Run the downloader first and review its per-city missing counts. If historical coverage is incomplete, we can narrow the training period for a carefully documented experiment.
- **Missing forecast for one of the future hours:** Run the downloader again with an end date encompassing all upcoming UTC target hours; never replace them with observed future temperatures.
- **Candidate dashboard empty:** Run `python -m gridwise.weather_forward` and check `/api/weather-model-comparison`. The dashboard reads the saved report; clicking Reload only refreshes that report.

Scheduled jobs remain disabled until explicitly enabled later.
