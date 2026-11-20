# Phase 5 — weather integration and first LightGBM benchmark

## Data inputs

- Fingrid dataset 124: electricity demand at 15-minute UTC intervals, prepared by `gridwise.prepare`.
- Open-Meteo Historical Weather API: fixed ERA5 hourly 2-m air temperature in °C at Helsinki, Turku, Tampere, Oulu and Rovaniemi. These are five location-specific proxies; they are **not a population-weighted national average**.

## Run

```powershell
python -m gridwise.weather --start 2024-01-01 --end 2026-10-31
python -m gridwise.forecast_ml
```

The weather collector downloads 2024, 2025, and January–October 2026 in year-sized requests for each city. No Open-Meteo key is required for noncommercial use subject to provider terms. Re-run only if the download failed or you intentionally need updated observations.

## Evaluation protocol

- Training: January 2024–December 2025.
- Validation: January–June 2026. Choose between calendar + historical-demand and calendar + historical-demand + weather using validation MAE only.
- Refit the selected model using all available complete observations before July 2026.
- Test: July–October 2026. Compare on the **same evaluated timestamps** against the fixed baseline winner previously selected on validation (normally `yesterday`).
- Task: rolling **24-hour-ahead** forecast of every 15-minute target. This is not a single forecast for the entire four-month test period; each timestamp uses demand measured at least 24 hours before that target.
- Demand targets with no Fingrid observation are never imputed.

## Leakage and deployment considerations

All demand features use values at t − 24 hours or earlier, including the historical rolling means. Future calendar values are known in advance. ERA5 is a **reanalysis** (a retrospectively reconstructed historical weather product), typically released days after observation; the initial model conservatively uses weather from **14 days before the target**. This experiment deliberately avoids the unknowable *realized temperature of the forecast target day*.

Even this historical benchmark is not a strict point-in-time operational simulation: the archived ERA5 weather record may have been updated after the original issue date, and it assumes instant availability of Fingrid demand data at t − 24 h. Before deployment, replace these assumptions with timestamped source releases and forecasts issued by the original forecast time, e.g. the Open-Meteo Historical Forecast/Single Runs products.

The original benchmark's test results have already been inspected in a prior phase. Continue to select all candidate features and hyperparameters **only on the validation set**, not from the test metrics.

## Outputs

- `data/raw/open_meteo_temperature_2024_2026.csv` (ignored by Git)
- `reports/lightgbm_2026.json` (committed, including metrics and assumptions)
- `data/model_outputs/lightgbm_predictions_2026.csv` (ignored)
- `data/model_outputs/lightgbm_first_week.png` (ignored)
- `data/model_outputs/lightgbm_2026.txt` (ignored)

## Sources

- https://data.fingrid.fi/en/datasets/124
- https://open-meteo.com/en/docs/historical-weather-api
- https://open-meteo.com/en/docs/historical-forecast-api
