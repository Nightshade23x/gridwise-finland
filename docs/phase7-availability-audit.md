# Phase 7 — Demand-data availability sensitivity

The historical demand model used a measurement at exactly `t - 24h` as its strongest feature. That is only valid for an operational 24-hour-ahead forecast if the measurement is already published at issue time (and available with the correct timestamp semantics).

This audit tests **hypothetical delays** by moving the most recent demand observation farther into the past by 0, 15, 30, 60, or 120 minutes. It compares the resulting yesterday-persistence predictions on a common evaluation set. It does **not** measure Fingrid's actual publication delays or reconstruct which historical value versions were available at issuance time, and it does not revise the previously reported LightGBM score.

Run:

```powershell
python -m gridwise.availability_audit
```

Outputs:

- `reports/availability_sensitivity_2026.json`
- `data/model_outputs/availability_sensitivity_2026.png`

**Next:** Verify provider timestamp and publication semantics, then test availability-safe ML features and re-evaluate on a *new* future holdout or through preregistered rolling-origin backtests. The July–October 2026 test has already been inspected and should not be described as untouched for any subsequent model tuning.
