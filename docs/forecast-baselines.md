# GridWise 24-hour-ahead forecasting baselines

This is the first *measured* forecasting benchmark, not a trained ML model.

- **Target:** Fingrid national demand (MW) at a 15-minute UTC timestamp `t`, predicted exactly 24 hours in advance.
- **Yesterday:** observed demand at `t - 24 hours`.
- **Last week:** observed demand at `t - 7 days`.
- **Blend:** the average of these two measurements.
- **Validation:** January–June 2026, used to choose the lowest-MAE baseline.
- **Test:** July–October 2026, held out from selection.
- **Metrics:** MAE and RMSE in MW, on the *same* valid timestamps for all three methods.
- **Missing targets or lagged observations:** excluded from comparison; not filled artificially.
- **No future leakage:** all inputs are observations at least 24 hours old at the target timestamp. Weather forecasts and other predictive inputs will be added in later phases.

**Important:** Current forecasts are *exactly 24 hours ahead* per target timestamp, not rolling within-day or week-ahead predictions. The naive `t - 24h` baseline uses the corresponding observation from the previous day; it is not available for horizons greater than 24 hours. UTC lags do not always correspond to the same Finnish clock time around daylight-saving changes.

Run `python -m gridwise.baselines` after preparing the full 2024–October 2026 input dataset. The produced JSON is small and may be committed to the GitHub repo; the per-timestamp predictions CSV and image are ignored.
