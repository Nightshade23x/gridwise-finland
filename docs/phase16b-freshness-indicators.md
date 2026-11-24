# Phase 16b: source-aware update timestamps

The dashboard now shows UTC timestamps and relative ages for its saved artifacts:

- Historical backtest metrics: the local CSV file modification time, **not** the end of the evaluation period.
- Selected historical or recent chart: selected CSV file modification time, **not** the Fingrid publication timestamp.
- Recent demand: the last successful data refresh or the last failed attempt.
- Main forward prediction and weather-candidate prediction: the recorded forecast issuance time.
- Forecast verification and official comparison: the saved report's `verified_at_utc` (the calculation time).

Relative ages recalculate locally every minute without network calls. Browser `Reload` controls only read saved reports; they do **not** invoke the refresh, verify, forward, official or weather workers. Recalculation requires the explicit CLI commands shown in each indicator. The API never reads credentials or updates files, and this change does not enable scheduling.

`/api/summary` exposes `saved_artifact_times_utc` for the historical and test CSV files; `/api/series` exposes `source_file_modified_utc`. These are *local filesystem timestamps*, not independently certified source publication or issuance evidence.
