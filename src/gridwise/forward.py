"""Phase 12: issue an honest next-24h experimental forecast from a fixed origin.

The model is retrained for this task. Unlike the previous t-24h backtests,
all predictor observations come from t-48h or earlier. For a 24-hour window
starting at the issue time, these observations precede issue by >=24 hours.
No future actual consumption, future reanalysis, or API calls are used here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_ml import model, errors
from .refresh import ROOT, LIVE_CSV, read_measurements, atomic_json, atomic_csv

HISTORY = Path("data/processed/consumption_2024_2026.csv")
FORECAST_JSON = Path("data/live/forward_forecast_24h.json")
FORECAST_CSV = Path("data/live/forward_forecast_24h.csv")
MODEL_PATH = Path("data/model_outputs/forward_24h_model.txt")
VALIDATION_START = pd.Timestamp("2026-01-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2026-07-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-11-01", tz="UTC")
INTERVAL = pd.Timedelta(minutes=15)
HORIZON_STEPS = 96

FEATURES = [
    "lag_48h", "lag_72h", "lag_7d", "prior_24h_avg", "prior_7d_avg",
    "local_hour_sin", "local_hour_cos", "weekday_sin", "weekday_cos",
    "year_day_sin", "year_day_cos", "is_weekend", "utc_offset_hours",
]
BASELINES = ("lag_48h", "lag_7d", "blend_50_50")


def select_forecast_by_validation(scores: dict, selected_baseline: str) -> str:
    """Select a production-facing candidate before evaluating future outcomes.

    Returns 'lightgbm' or the validation-selected reference name. A validation
    gate is not a guarantee of generalization and must never look at test data.
    """
    if selected_baseline not in BASELINES:
        raise ValueError("Unknown reference baseline")
    model_mae = float(scores["lightgbm"]["mae_mw"])
    baseline_mae = float(scores[selected_baseline]["mae_mw"])
    if not (np.isfinite(model_mae) and np.isfinite(baseline_mae) and
            model_mae >= 0 and baseline_mae >= 0):
        raise ValueError("Validation MAEs must be finite and nonnegative")
    # On ties, prefer the more transparent reference baseline.
    return "lightgbm" if model_mae < baseline_mae else selected_baseline


def utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Issue time must include a timezone")
    return t.tz_convert("UTC")


def combined_demand(root: Path) -> pd.DataFrame:
    """Merge prepared history with measured recent data; preserve raw sources."""
    old = root / HISTORY
    if not old.is_file():
        raise FileNotFoundError(f"Missing historical prepared dataset: {old}")
    if not (root / LIVE_CSV).is_file():
        raise FileNotFoundError("Missing Phase 11 data; run python -m gridwise.refresh first")
    history = pd.read_csv(old, usecols=["timestamp_utc", "demand_mw"])
    history["timestamp_utc"] = pd.to_datetime(history["timestamp_utc"], utc=True, errors="raise")
    recent = read_measurements(root / LIVE_CSV)
    if recent.empty:
        raise ValueError("Recent demand store is empty")
    if history["timestamp_utc"].isna().any() or history["timestamp_utc"].duplicated().any():
        raise ValueError("Invalid historical timestamps")
    history = history.set_index("timestamp_utc")["demand_mw"].astype(float)
    recent = recent.set_index("timestamp_utc")["demand_mw"].astype(float)
    overlap = recent.index.intersection(history.index)
    mismatched = overlap[~np.isclose(history.loc[overlap], recent.loc[overlap], rtol=0, atol=1e-6, equal_nan=True)]
    if len(mismatched):
        raise ValueError("Conflicting historical and recent observations; reconcile sources before forecasting")
    new = recent.loc[~recent.index.isin(history.index)]
    series = pd.concat([history, new]).sort_index()
    if not series.index.is_monotonic_increasing or series.index.has_duplicates:
        raise ValueError("Combined demand timestamps must be unique")
    full = pd.date_range(series.index.min(), series.index.max(), freq=INTERVAL, tz="UTC")
    if (series.index.minute % 15 != 0).any() or (series.index.second != 0).any():
        raise ValueError("Demand timestamps must align to 15-minute UTC grid")
    return series.reindex(full).rename("demand_mw").rename_axis("timestamp_utc").reset_index()


def feature_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create conservative target-time features; every input ends at t-48h."""
    if not {"timestamp_utc", "demand_mw"}.issubset(data.columns):
        raise ValueError("Expected timestamp_utc and demand_mw")
    d = data[["timestamp_utc", "demand_mw"]].copy().reset_index(drop=True)
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True, errors="raise")
    t = pd.DatetimeIndex(d["timestamp_utc"])
    if t.has_duplicates or len(t) < 2017 or not (t[1:] - t[:-1] == INTERVAL).all():
        raise ValueError("Expected unique continuous 15-minute timeline with at least 21 days")
    y = pd.to_numeric(d["demand_mw"], errors="raise")
    d["lag_48h"] = y.shift(192)
    d["lag_72h"] = y.shift(288)
    d["lag_7d"] = y.shift(672)
    d["prior_24h_avg"] = y.shift(192).rolling(96, min_periods=90).mean()
    d["prior_7d_avg"] = y.shift(192).rolling(672, min_periods=630).mean()
    local = t.tz_convert("Europe/Helsinki")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    weekday = local.dayofweek.to_numpy()
    day = local.dayofyear.to_numpy()
    d["local_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["local_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    d["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    d["year_day_sin"] = np.sin(2 * np.pi * day / 365.25)
    d["year_day_cos"] = np.cos(2 * np.pi * day / 365.25)
    d["is_weekend"] = (weekday >= 5).astype(int)
    d["utc_offset_hours"] = np.array([z.utcoffset().total_seconds()/3600 for z in local])
    d["blend_50_50"] = (d["lag_48h"] + d["lag_7d"]) / 2
    return d


def issue_forecast(
    combined: pd.DataFrame,
    issue_time: datetime | pd.Timestamp,
    *,
    estimators: int = 200,
    max_data_age_hours: float = 6.0,
    evaluate_november: bool = True,
) -> tuple[dict, pd.DataFrame, object]:
    """Train on available history, optionally evaluate November without fitting on it.

    The live artifact is a *new* conservative 24h-window model, not the
    retrospectively evaluated 24h-lead weather model.
    """
    issue = utc_timestamp(issue_time)
    if estimators < 1 or max_data_age_hours <= 0:
        raise ValueError("Invalid estimators or max_data_age_hours")
    first_target = issue.floor("15min") + INTERVAL  # Always strictly after issuance, even on grid
    known = combined.copy()
    known["timestamp_utc"] = pd.to_datetime(known["timestamp_utc"], utc=True)
    # Guard against inadvertent records dated in the future.
    known = known.loc[known["timestamp_utc"] <= issue - pd.Timedelta(minutes=60)]
    actual = known.loc[known["demand_mw"].notna(), "timestamp_utc"]
    if actual.empty:
        raise ValueError("No completed demand observations available")
    latest = actual.max()
    age_hrs = (issue - latest).total_seconds() / 3600
    if age_hrs > max_data_age_hours:
        raise ValueError(f"Recent demand is {age_hrs:.1f} hours old; run gridwise.refresh before issuing a forecast")
    if first_target <= latest:
        raise ValueError("Forecast target range must start after latest measurement")
    future = pd.DataFrame({
        "timestamp_utc": pd.date_range(first_target, periods=HORIZON_STEPS, freq=INTERVAL, tz="UTC"),
        "demand_mw": np.nan,
    })
    # The 60-minute observation buffer leaves a legitimate gap before the
    # first target; represent it explicitly rather than pretending it is observed.
    missing_grid = pd.date_range(
        known["timestamp_utc"].max() + INTERVAL, first_target - INTERVAL,
        freq=INTERVAL, tz="UTC",
    )
    unavailable = pd.DataFrame({"timestamp_utc": missing_grid, "demand_mw": np.nan})
    combined_forecast = pd.concat([known, unavailable, future], ignore_index=True)
    f = feature_table(combined_forecast)
    fit_rows = f.loc[(f["timestamp_utc"] < first_target) & f["demand_mw"].notna() & f[FEATURES].notna().all(axis=1)]
    train = fit_rows.loc[fit_rows["timestamp_utc"] < VALIDATION_START]
    validation = fit_rows.loc[(fit_rows["timestamp_utc"] >= VALIDATION_START) & (fit_rows["timestamp_utc"] < VALIDATION_END)]
    if len(train) < 1000 or len(validation) < 1000:
        raise ValueError("Insufficient historical training and validation observations")
    # The model type and hyperparameters are fixed before inspecting November.
    val_model = model(estimators)
    val_model.fit(train[FEATURES], train["demand_mw"])
    validation_scores = {
        "lightgbm": errors(validation["demand_mw"], val_model.predict(validation[FEATURES])),
        **{name: errors(validation["demand_mw"], validation[name]) for name in BASELINES},
    }
    selected_baseline = min(BASELINES, key=lambda name: validation_scores[name]["mae_mw"])
    # A forward model can be worse than a simple reference. Never present it as
    # the preferred forecast unless its *validation* MAE beats that reference.
    preferred_strategy = select_forecast_by_validation(validation_scores, selected_baseline)

    november = None
    if evaluate_november:
        candidate = fit_rows.loc[fit_rows["timestamp_utc"] >= HOLDOUT_START]
        prior = fit_rows.loc[fit_rows["timestamp_utc"] < HOLDOUT_START]
        if len(candidate) >= 96 and len(prior) >= 1000:
            frozen = model(estimators)
            frozen.fit(prior[FEATURES], prior["demand_mw"])
            november = {
                "start_utc": HOLDOUT_START.isoformat(),
                "end_utc": candidate["timestamp_utc"].max().isoformat(),
                "paired_rows": int(len(candidate)),
                "lightgbm": errors(candidate["demand_mw"], frozen.predict(candidate[FEATURES])),
                "selected_baseline": selected_baseline,
                "baseline": errors(candidate["demand_mw"], candidate[selected_baseline]),
                "caution": "Preliminary November evaluation; model fitted on pre-November data. Review only once before further tuning.",
            }

    # Retrain on everything genuinely known at issue time, including November;
    # the November evaluation model above is separate from this deployed model.
    trained = model(estimators)
    trained.fit(fit_rows[FEATURES], fit_rows["demand_mw"])
    horizon = f.tail(HORIZON_STEPS).copy()
    if not horizon[FEATURES].notna().all(axis=1).all():
        raise ValueError("Future feature inputs contain missing historical observations; refresh/recover gaps first")
    # Every target's newest predictor is t-48h, >24h behind the issue.
    if not (horizon["timestamp_utc"].max() - pd.Timedelta(hours=48) <= issue - pd.Timedelta(hours=24)):
        raise AssertionError("Future predictor exceeds the permitted issue-time cutoff")
    predicted = trained.predict(horizon[FEATURES])
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Invalid forward model output")
    reference = horizon[selected_baseline].to_numpy(dtype=float)
    points = pd.DataFrame({
        "timestamp_utc": horizon["timestamp_utc"].to_numpy(),
        "predicted_mw": np.round(predicted, 3),
        "baseline_mw": np.round(reference, 3),
        # Do not change the meaning of predicted_mw, which is already used by
        # immutable Phase 13/14 forecast verification. This is separate advice.
        "preferred_mw": np.round(predicted if preferred_strategy == "lightgbm" else reference, 3),
    })
    payload = {
        "kind": "experimental_forward_forecast",
        "status": "ok",
        "issued_at_utc": issue.isoformat(),
        "first_target_utc": points["timestamp_utc"].iloc[0].isoformat(),
        "last_target_utc": points["timestamp_utc"].iloc[-1].isoformat(),
        "source_latest_measurement_utc": latest.isoformat(),
        "source_age_hours_at_issue": round(age_hrs, 2),
        "resolution_minutes": 15,
        "forecast_horizon_hours": 24,
        "model": "LightGBM demand-only, conservative 48h input lag",
        "selected_reference_baseline": selected_baseline,
        "preferred_strategy": preferred_strategy,
        "preferred_validation_mae_mw": validation_scores[preferred_strategy]["mae_mw"],
        "validation_scores": validation_scores,
        "preliminary_november_evaluation": november,
        "training_observations": int(len(fit_rows)),
        "assumptions": [
            "Each of the 96 target slots uses demand measurements from at least 24 hours before the common issue time.",
            "Weather forecasts are not included in this version.",
            "This newly trained forward model is distinct from the earlier 310 MW historical backtest model.",
            "Outputs are research predictions, not operational grid dispatch advice.",
        ],
        "points": [
            {"timestamp_utc": r.timestamp_utc.isoformat(), "predicted_mw": float(r.predicted_mw),
             "baseline_mw": float(r.baseline_mw), "preferred_mw": float(r.preferred_mw)}
            for r in points.itertuples(index=False)
        ],
    }
    return payload, points, trained


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a next-24-hour experimental forecast, no future observations")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--issue-at", type=str, help="Explicit timezone-aware ISO time, for reproducible tests")
    parser.add_argument("--estimators", type=int, default=200)
    parser.add_argument("--max-data-age-hours", type=float, default=6.0)
    args = parser.parse_args()
    root = args.root.resolve()
    issue = utc_timestamp(args.issue_at) if args.issue_at else pd.Timestamp.now(tz="UTC")
    data = combined_demand(root)
    report, points, estimator = issue_forecast(data, issue, estimators=args.estimators,
                                              max_data_age_hours=args.max_data_age_hours)
    dest = root / FORECAST_JSON
    from .verify import archive_forecast
    # Preserve the previous Phase 12 latest snapshot before replacing it.
    if dest.is_file():
        archive_forecast(root, json.loads(dest.read_text(encoding="utf-8")))
    # Save the model separately from the read-only web container.
    model_path = root / MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    estimator.booster_.save_model(str(model_path))
    archive_forecast(root, report)
    atomic_csv(root / FORECAST_CSV, points)
    atomic_json(dest, report)
    print(f"Issued at: {report['issued_at_utc']}")
    print(f"Latest measured demand: {report['source_latest_measurement_utc']}")
    print(f"Forward targets: {len(points)} quarter-hour intervals")
    print(f"Baseline: {report['selected_reference_baseline']}")
    print(f"Validation-preferred forecast: {report['preferred_strategy']} "
          "(separate from saved LightGBM prediction for verification)")
    print(f"Validation MAE: model={report['validation_scores']['lightgbm']['mae_mw']:.1f} MW; "
          f"reference={report['validation_scores'][report['selected_reference_baseline']]['mae_mw']:.1f} MW")
    if report["preliminary_november_evaluation"]:
        test = report["preliminary_november_evaluation"]
        print(f"November (preliminary): model={test['lightgbm']['mae_mw']:.1f} MW; "
              f"baseline={test['baseline']['mae_mw']:.1f} MW ({test['paired_rows']} rows)")
    print(f"Saved {dest} and {root / FORECAST_CSV}")
    print("Note: this model predicts the next 24h window; it is distinct from the previous fixed-24h-lead backtest.")


if __name__ == "__main__":
    main()
