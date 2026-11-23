# Phase 14 — Fingrid forecast snapshot benchmarking

Phase 14 compares the **newly issued demand-only GridWise forecast** against Fingrid's **dataset 166**, using source measurements (dataset 124). It does **not** use the historical 310 MW LightGBM backtest model.

## Why capture rather than backfill official forecasts?

Fingrid 166 publishes rolling 72-hour demand predictions updated every 15 minutes. Its standard observation API returns forecast values for target periods but does not provide a verifiable historical **issue time / version identifier** for each returned value. Downloading forecast data for a past target date **now** is not proof of what Fingrid predicted **then**.

`gridwise.official capture` saves a locally timestamped API response immediately after a new GridWise forecast. It retains only official targets starting **after the API capture completes**. This is a prospective comparison with a documented capture-time difference, not independent proof of Fingrid's internal issuance time. Locally saved data can be edited; the archive is not tamper-proof.

## Workflow

1. Activate the GridWise Python environment, ensure `FINGRID_API_KEY` is in `.env`, and install `.[dev,ml,app]`.
2. Fetch new measurements and issue a **fresh** GridWise forecast:

   ```powershell
   .\.venv\Scripts\python.exe -m gridwise.refresh
   .\.venv\Scripts\python.exe -m gridwise.forward
   ```

3. **Immediately** capture Fingrid's currently returned predictions:

   ```powershell
   .\.venv\Scripts\python.exe -m gridwise.official capture
   ```

   Captures are refused when the current GridWise issue is older than 30 minutes. The API request needs an active network connection. Running the same command twice will preserve the first immutable local capture without replacing it. If the first target has already started, it will be excluded. No attempt is made to reconstruct snapshots from before Phase 14 was installed.

4. After the target observations become available, run:

   ```powershell
   .\.venv\Scripts\python.exe -m gridwise.refresh
   .\.venv\Scripts\python.exe -m gridwise.verify
   .\.venv\Scripts\python.exe -m gridwise.official verify
   ```

   The latest official comparison is saved in `data/live/official_benchmark.json` and presented on the dashboard under **GridWise versus Fingrid**. The read-only API endpoint is `/api/official-benchmark`.

## How scoring works

- Actual demand is from locally stored Fingrid dataset 124 measurements.
- GridWise predictions, historical reference predictions, official captured predictions, and actual demand are joined on exact UTC quarter-hour **target timestamps**.
- Only official predictions captured **before their target interval starts** and actual observations old enough for the configurable 60-minute availability buffer are scored.
- Missing values are excluded rather than filled in.
- MAE, RMSE and signed bias are computed on **identical matched points** for all three methods.
- Partial results and counts must not be represented as whole-horizon performance; timestamp coverage is shown explicitly.
- Captures at different clock times can have an informational advantage, so results should not be presented as strictly simultaneous issue-time comparisons.
- Predictions overlapping between separate issuances are evaluated separately; do not pool repeated target timestamps across issuances without a deliberate experimental design.

## Files

- `data/live/forecast_archive/` — archived GridWise issuances (Phase 13)
- `data/live/official_snapshots/` — archived dataset 166 snapshots (one per corresponding issue)
- `data/live/official_benchmark.json` — saved scoring report

These files remain under `data/live/`, which is excluded from Git. No API keys are exposed to the web server. Scheduling stays **disabled** unless explicitly configured.

## Source

- Fingrid dataset 166: https://data.fingrid.fi/en/datasets/166
- Fingrid consumption dataset 124: https://data.fingrid.fi/en/datasets/124
