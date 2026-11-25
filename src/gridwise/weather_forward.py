"""Phase 16: side-by-side 24-hour forward experiment with 48h-lead weather.

This is intentionally separate from Phase 12-15 ``forward_forecast_24h.json``.
Archived official comparisons and verifiers keep their existing interpretation.
No actual future weather, ERA5 reanalysis, or unissued demand targets are inputs.
"""
from __future__ import annotations

from datetime import timedelta

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_ml import errors, model
from .forward import (
    BASELINES, FEATURES, HORIZON_STEPS, INTERVAL, VALIDATION_START,
    VALIDATION_END, combined_demand, feature_table, utc_timestamp,
)
from .refresh import ROOT, atomic_json, atomic_csv
from .weather_previous import COLUMNS, DEFAULT_OUTPUT

REPORT = Path("data/live/weather_model_comparison_24h.json")
ARCHIVE = Path("data/live/weather_model_comparison_archive")
CSV = Path("data/live/weather_model_comparison_24h.csv")
MODEL = Path("data/model_outputs/weather_forward_candidate.txt")
WEATHER_FEATURES = tuple(COLUMNS.values()) + ("forecast_temperature_avg_c", "forecast_heating_degree_c")


def _align_weather(targets: pd.DatetimeIndex, weather: pd.DataFrame) -> pd.DataFrame:
    """Join previous_day2 target-hour forecasts to each quarter-hour; no lookahead."""
    required = {"timestamp_utc", *COLUMNS.values()}
    if not required.issubset(weather.columns):
        raise ValueError(f"Missing 48h-lead forecast columns: {sorted(required - set(weather.columns))}")
    w = weather[["timestamp_utc", *COLUMNS.values()]].copy()
    w["timestamp_utc"] = pd.to_datetime(w["timestamp_utc"], utc=True, errors="raise")
    w = w.sort_values("timestamp_utc").set_index("timestamp_utc")
    ind = w.index
    if (ind.hasnans or ind.has_duplicates or (ind.minute != 0).any() or
        (ind.second != 0).any() or (ind.microsecond != 0).any()):
        raise ValueError("Invalid 48h-lead hourly forecast index")
    for col in COLUMNS.values():
        w[col] = pd.to_numeric(w[col], errors="raise")
        invalid = w[col].notna() & (~np.isfinite(w[col]) | (w[col] < -90) | (w[col] > 65))
        if invalid.any():
            raise ValueError("Invalid 48h-lead temperature values")
    # Reindex with 45-minute tolerance: 13:00 forecast valid for 13:00-13:45,
    # not 14:00. Never interpolate using a future weather reading.
    merged = w.reindex(targets, method="ffill", tolerance=timedelta(minutes=45))
    merged["forecast_temperature_avg_c"] = merged[list(COLUMNS.values())].mean(axis=1, skipna=False)
    merged["forecast_heating_degree_c"] = np.maximum(0, 18 - merged["forecast_temperature_avg_c"])
    return merged.reset_index(drop=True)


def experiment(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    issue_time,
    *,
    estimators: int = 200,
    max_data_age_hours: float = 6.0,
) -> tuple[dict, pd.DataFrame, object]:
    """Select three candidates using validation only, issue 96 independent targets."""
    issue = utc_timestamp(issue_time)
    if estimators <= 0 or max_data_age_hours <= 0:
        raise ValueError("Invalid estimator count or maximum data age")
    first = issue.floor("15min") + INTERVAL
    d = demand[["timestamp_utc", "demand_mw"]].copy()
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    if d["timestamp_utc"].isna().any() or d["timestamp_utc"].duplicated().any():
        raise ValueError("Invalid demand timeline")
    d = d.loc[d["timestamp_utc"] <= issue - timedelta(minutes=60)].copy()
    measured = d.loc[d.demand_mw.notna(), "timestamp_utc"]
    if measured.empty:
        raise ValueError("No available demand measurements")
    latest = measured.max()
    age = (issue - latest).total_seconds() / 3600
    if age > max_data_age_hours:
        raise ValueError(f"Demand is {age:.2f} hours old; refresh demand first")
    if d.empty or first <= d["timestamp_utc"].max():
        raise ValueError("Forward timeline must begin after available demand")
    targets = pd.date_range(first, periods=HORIZON_STEPS, freq=INTERVAL, tz="UTC")
    gap = pd.date_range(d["timestamp_utc"].max() + INTERVAL, first - INTERVAL, freq=INTERVAL, tz="UTC")
    fill = pd.DataFrame({"timestamp_utc":gap, "demand_mw": np.nan})
    future = pd.DataFrame({"timestamp_utc":targets, "demand_mw": np.nan})
    f = feature_table(pd.concat([d, fill, future], ignore_index=True))
    # Open-Meteo previous_day2 temperatures were predicted 48 hours before
    # each target; even the last target lies >24h after its weather origin.
    if targets[-1] - timedelta(hours=48) > issue - timedelta(hours=23):
        raise AssertionError("Weather forecast not sufficiently old at issuance")
    weather_cols = _align_weather(pd.DatetimeIndex(f["timestamp_utc"]), weather)
    f[list(WEATHER_FEATURES)] = weather_cols[list(WEATHER_FEATURES)]
    # Complete-case rows are identical for all evaluated candidates, so the
    # baseline cannot gain an unfair evaluation advantage from missing weather.
    required = ["demand_mw", *FEATURES, *WEATHER_FEATURES, *BASELINES]
    eligible = f[required].notna().all(axis=1)
    known = f.loc[(f["timestamp_utc"] < first) & eligible & f["demand_mw"].notna()]
    train = known.loc[known["timestamp_utc"] < VALIDATION_START]
    valid = known.loc[(known["timestamp_utc"] >= VALIDATION_START) &
                      (known["timestamp_utc"] < VALIDATION_END)]
    if len(train) < 1000 or len(valid) < 1000:
        raise ValueError("Insufficient matching historical demand and 48h forecast weather for training/validation")
    candidates = {"demand_only": FEATURES, "demand_weather": FEATURES + list(WEATHER_FEATURES)}
    scores = {}
    for name, features in candidates.items():
        regressor = model(estimators)
        regressor.fit(train[features], train["demand_mw"])
        scores[name] = errors(valid["demand_mw"], regressor.predict(valid[features]))
    for name in BASELINES:
        scores[name] = errors(valid["demand_mw"], valid[name])
    selected = min(scores, key=lambda key: scores[key]["mae_mw"])
    # Refit both candidates on all legitimately known measurements at issue time.
    fitted = {}
    for name, features in candidates.items():
        regressor = model(estimators)
        regressor.fit(known[features], known["demand_mw"])
        fitted[name] = regressor
    horizon = f.tail(HORIZON_STEPS).copy()
    if not horizon[list(FEATURES) + list(WEATHER_FEATURES) + list(BASELINES)].notna().all(axis=1).all():
        raise ValueError("48h-lead forecast weather or historical demand unavailable for some future targets")
    values = {name: fitted[name].predict(horizon[features]) for name, features in candidates.items()}
    for name in BASELINES:
        values[name] = horizon[name].to_numpy(dtype=float)
    for name, array in values.items():
        if not np.isfinite(array).all() or (array < 0).any():
            raise ValueError(f"Non-finite or negative {name} prediction")
    result = pd.DataFrame({"timestamp_utc":targets,
                           "demand_only_mw":np.round(values["demand_only"],3),
                           "demand_weather_mw":np.round(values["demand_weather"],3),
                           "baseline_mw":np.round(values[min(BASELINES, key=lambda k:scores[k]['mae_mw'])],3),
                           "preferred_mw":np.round(values[selected],3)})
    if len(result) != 96:
        raise AssertionError("Expected 96 forecast targets")
    report = {
        "kind": "weather_forward_candidate",
        "status": "ok",
        "schema_version": 1,
        "issued_at_utc":issue.isoformat(),
        "first_target_utc":targets[0].isoformat(),
        "last_target_utc":targets[-1].isoformat(),
        "source_latest_measurement_utc":latest.isoformat(),
        "weather_source":"Open-Meteo Previous Runs API: temperature_2m_previous_day2, nominal 48h weather forecast lead",
        "weather_source_vintage_certified":False,
        "weather_city_count":len(COLUMNS),
        "forecast_horizon_hours":24,
        "resolution_minutes":15,
        "validation_start_utc":VALIDATION_START.isoformat(),
        "validation_end_exclusive_utc":VALIDATION_END.isoformat(),
        "train_rows":int(len(train)),
        "validation_rows":int(len(valid)),
        "refit_rows":int(len(known)),
        "validation_mae_mw":{k:float(v["mae_mw"]) for k,v in scores.items()},
        "preferred_strategy":selected,
        "baseline_name":min(BASELINES, key=lambda k:scores[k]['mae_mw']),
        "notes":[
            "Candidate only. Existing Phase 12-15 issued forecast and Phase 13-14 verifiers are unchanged.",
            "All candidate scores use identical complete-case validation timestamps.",
            "48-hour-lead weather values are historical forecasts, not realized/ERA5 future weather.",
            "Previous Runs is not a certified record of individual model-release timestamps. Weather origins are nominal target minus 48 hours.",
            "Performance on new, later-issued forecasts must be verified independently before promotion.",
        ],
        "points":[{"timestamp_utc":r.timestamp_utc.isoformat(),
                   "demand_only_mw":float(r.demand_only_mw),
                   "demand_weather_mw":float(r.demand_weather_mw),
                   "baseline_mw":float(r.baseline_mw),
                   "preferred_mw":float(r.preferred_mw)}
                  for r in result.itertuples(index=False)]
    }
    return report, result, fitted["demand_weather"]


def save_report(root: Path, report: dict, result: pd.DataFrame, weather_model) -> Path:
    """Archive snapshots under issuance-specific paths before updating latest."""
    target = root / ARCHIVE / (pd.Timestamp(report["issued_at_utc"]).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    if target.exists():
        if json.loads(target.read_text(encoding="utf-8")) != report:
            raise ValueError("Conflicting weather-candidate snapshot at same issue time")
    else:
        atomic_json(target, report)
    dest = root / REPORT
    atomic_csv(root / CSV, result)
    atomic_json(dest, report)
    model_path = root / MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    weather_model.booster_.save_model(str(model_path))
    return dest


def main():
    parser = argparse.ArgumentParser(description="Weather-aware 24h experiment; does not modify official forward forecasts")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--weather", type=Path, help="48h-lead historical weather CSV")
    parser.add_argument("--issue-at", type=str, help="Explicit timezone-aware ISO time for reproducibility")
    parser.add_argument("--estimators", type=int, default=200)
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.weather if args.weather is not None else root / DEFAULT_OUTPUT
    if not path.is_file():
        raise FileNotFoundError(f"Missing 48h-lead weather forecasts: {path}. Run python -m gridwise.weather_previous first")
    weather = pd.read_csv(path)
    issue = utc_timestamp(args.issue_at) if args.issue_at else pd.Timestamp.now(tz="UTC")
    demand = combined_demand(root)
    report, frame, fitted = experiment(demand, weather, issue, estimators=args.estimators)
    dest = save_report(root, report, frame, fitted)
    print(f"Issued 96 candidate targets at {report['issued_at_utc']}")
    print(f"Matched training rows: {report['train_rows']:,}; validation rows: {report['validation_rows']:,}")
    for name, value in report["validation_mae_mw"].items():
        print(f"  {name:<15} validation MAE: {value:.2f} MW")
    print(f"Validation-preferred: {report['preferred_strategy']}")
    print(f"Saved candidate (not promoted): {dest}")

if __name__ == "__main__":
    main()
