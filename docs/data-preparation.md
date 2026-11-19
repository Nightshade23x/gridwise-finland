# Data preparation — GridWise

Run after collecting the original Fingrid CSVs. Raw observations are never overwritten.

```powershell
.\.venv\Scripts\python.exe -m gridwise.prepare --inputs data/raw/fingrid_consumption_2024.csv data/raw/fingrid_consumption_2025.csv --start 2024-01-01 --end 2026-01-01 --output data/processed/consumption_2024_2025.csv --report data/quality/prepared_2024_2025_report.json
```

The processed CSV contains a **continuous UTC 15-minute index**, observed demand in MW, flags showing observed and corrected timestamps, and Finnish local-calendar fields. Missing demand targets remain empty. A five-second timestamp correction is only applied if there is no conflicting exact-grid observation and the original interval is 15 minutes long. All repairs and exclusions are recorded in the JSON audit report.

For forecasting, drop rows with missing target values when fitting the target; generate lag features causally without using future observations. Reserve later dates for test data and do not use random splitting. Neither interpolated nor time-corrected values should silently be labelled as original measurements.
