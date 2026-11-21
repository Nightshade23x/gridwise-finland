# Phase 9b — Selected-period insights

The dashboard includes **What this period shows**, which updates when a user changes the view, date range, or chart resolution and clicks Apply.

- **Model comparison:** MAE for LightGBM and the validation-selected baseline on identical 15-minute timestamps; percent difference; mean signed error; maximum absolute model error, with date/time. This is a historical backtest, not a live forecast or causal attribution.
- **Historical demand:** average/minimum/maximum observed demand; timestamps for the extrema; proportion of valid measurements. Missing values are never imputed.
- **Empty selections:** display an informative message instead of invented findings.
- **Resolution:** insight metrics always use raw 15-minute observations, even if the chart shows hourly/daily means. This avoids artificially reducing MAE by averaging before computing errors.

New endpoint: `/api/insights?kind=test&start=2026-07-01&end=2026-07-08` (dates in UTC, `end` exclusive). Use `kind=history` for descriptive demand insights.

No extra packages, external AI service, or API key are required. Run the existing app and check `python -m pytest -q` after installing the patch. The insights are descriptive summaries of the current selection, not explanations of the underlying causes or independent forward-looking performance evidence.
