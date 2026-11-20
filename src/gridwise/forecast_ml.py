"""First LightGBM 24-hour-ahead demand experiment with chronological evaluation.

The feature-generation function uses historical measurements only. ERA5 lagged
by 14 days is a conservative historical proxy, not a weather forecast for the
target day. This is a retrospective backtest, not yet an operational forecast.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .baselines import make_predictions, COLUMNS
from .ingest import ROOT

HOUR_STEPS = 4
DAY_STEPS = 96
WEEK_STEPS = 672
# Open-Meteo ERA5 is usually published after several days; choose a two-week
# lag rather than accidentally giving the model tomorrow's observed weather.
WEATHER_LAG_STEPS = 14 * DAY_STEPS
CALENDAR_FEATURES = [
    "local_hour_sin", "local_hour_cos", "weekday_sin", "weekday_cos",
    "year_day_sin", "year_day_cos", "is_weekend", "local_utc_offset_hours",
]
DEMAND_FEATURES = [
    "demand_lag_24h", "demand_lag_48h", "demand_lag_72h", "demand_lag_7d",
    "demand_24h_history_mean", "demand_7d_history_mean",
]
CORE_FEATURES = CALENDAR_FEATURES + DEMAND_FEATURES


def build_features(prepared: pd.DataFrame, hourly_weather: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Make target-time features whose demand and weather values predate t-24h."""
    required = {"timestamp_utc", "demand_mw"}
    if not required.issubset(prepared.columns):
        raise ValueError(f"Prepared input missing {sorted(required - set(prepared.columns))}")
    clean = prepared[["timestamp_utc", "demand_mw"]].copy()
    clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True)
    clean = clean.sort_values("timestamp_utc").reset_index(drop=True)
    idx = pd.DatetimeIndex(clean["timestamp_utc"])
    if idx.has_duplicates or len(idx) < 2017 or not (idx[1:] - idx[:-1] == pd.Timedelta(minutes=15)).all():
        raise ValueError("Prepared data must be a unique, continuous 15-minute UTC grid (at least 21 days)")
    clean["demand_mw"] = pd.to_numeric(clean["demand_mw"], errors="raise")
    demand = clean["demand_mw"]
    for name, lag in [("demand_lag_24h", 96), ("demand_lag_48h", 192),
                      ("demand_lag_72h", 288), ("demand_lag_7d", 672)]:
        clean[name] = demand.shift(lag)
    # All windows end with the most recent observation allowed at issue time t-24h.
    clean["demand_24h_history_mean"] = demand.shift(DAY_STEPS).rolling(96, min_periods=90).mean()
    clean["demand_7d_history_mean"] = demand.shift(DAY_STEPS).rolling(7 * DAY_STEPS, min_periods=630).mean()

    local = idx.tz_convert("Europe/Helsinki")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60.0
    dayofweek = local.dayofweek.to_numpy()
    year_day = local.dayofyear.to_numpy()
    clean["local_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    clean["local_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    clean["weekday_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    clean["weekday_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    clean["year_day_sin"] = np.sin(2 * np.pi * year_day / 365.25)
    clean["year_day_cos"] = np.cos(2 * np.pi * year_day / 365.25)
    clean["is_weekend"] = (dayofweek >= 5).astype(int)
    clean["local_utc_offset_hours"] = np.array([t.utcoffset().total_seconds() / 3600 for t in local])

    if "timestamp_utc" not in hourly_weather.columns:
        raise ValueError("Weather missing timestamp_utc")
    columns = [c for c in hourly_weather.columns if c.startswith("temp_") and c.endswith("_c")]
    if not columns:
        raise ValueError("Weather must have at least one temp_<city>_c column")
    weather = hourly_weather[["timestamp_utc", *columns]].copy()
    weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    weather = weather.sort_values("timestamp_utc").set_index("timestamp_utc")
    if weather.index.has_duplicates:
        raise ValueError("Duplicate weather timestamps")
    if not (weather.index.minute == 0).all():
        raise ValueError("Weather timestamps must be on UTC hour")
    for col in columns:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    # Only carry a single hourly value over the three following quarter-hours.
    quarter_hourly = weather.reindex(idx, method="ffill", tolerance=pd.Timedelta(minutes=45))
    weather_features = []
    for col in columns:
        feature = f"{col}_lag_14d"
        clean[feature] = quarter_hourly[col].to_numpy().astype(float)
        clean[feature] = clean[feature].shift(WEATHER_LAG_STEPS)
        weather_features.append(feature)
    return clean, weather_features


def errors(y: pd.Series, pred: np.ndarray | pd.Series) -> dict:
    truth = np.asarray(y, dtype=float)
    estimate = np.asarray(pred, dtype=float)
    return {
        "mae_mw": round(float(np.mean(np.abs(estimate - truth))), 3),
        "rmse_mw": round(float(np.sqrt(np.mean(np.square(estimate - truth)))), 3),
    }


def model(n_estimators: int = 350) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.045,
        num_leaves=31,
        min_child_samples=70,
        colsample_bytree=0.9,
        subsample=0.9,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )


def experiment(
    prepared: pd.DataFrame,
    weather: pd.DataFrame,
    baseline_report: dict,
    val_start: str = "2026-01-01",
    test_start: str = "2026-07-01",
    end: str = "2026-11-01",
    n_estimators: int = 350,
) -> tuple[dict, pd.DataFrame, LGBMRegressor]:
    """Select model features on validation only, refit on train+val, then test once."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    winner_baseline = baseline_report.get("selected_baseline")
    if winner_baseline not in COLUMNS:
        raise ValueError("Baseline report missing a valid validation-selected model")
    b = make_predictions(prepared)[["timestamp_utc", *COLUMNS.values()]]
    f, weather_features = build_features(prepared, weather)
    f = f.merge(b, on="timestamp_utc", validate="one_to_one")
    train_cut = pd.Timestamp(val_start, tz="UTC")
    val_cut = pd.Timestamp(test_start, tz="UTC")
    end_cut = pd.Timestamp(end, tz="UTC")
    if not (f["timestamp_utc"].min() < train_cut < val_cut < end_cut <= f["timestamp_utc"].max() + pd.Timedelta(minutes=15)):
        raise ValueError("Model splits must fall inside dataset in chronological order")
    if baseline_report.get("validation", {}).get("start_utc") != train_cut.isoformat() or baseline_report.get("test", {}).get("start_utc") != val_cut.isoformat():
        raise ValueError("Baseline report validation/test dates do not match model evaluation")
    features = {"calendar_and_demand": CORE_FEATURES, "calendar_demand_weather": CORE_FEATURES + weather_features}
    full_features = features["calendar_demand_weather"]
    baseline_col = COLUMNS[winner_baseline]
    # Evaluate both candidates and the previously selected baseline on precisely
    # the same rows. Neither validation nor test target appears as a feature.
    eligible = f[["demand_mw", baseline_col, *full_features]].notna().all(axis=1)
    trn = f[(f["timestamp_utc"] < train_cut) & eligible]
    valid = f[(f["timestamp_utc"] >= train_cut) & (f["timestamp_utc"] < val_cut) & eligible]
    held = f[(f["timestamp_utc"] >= val_cut) & (f["timestamp_utc"] < end_cut) & eligible]
    if min(map(len, (trn, valid, held))) < 50:
        raise ValueError("Insufficient complete training, validation or test observations")

    trained = {}
    validation_scores = {}
    for name, columns in features.items():
        regressor = model(n_estimators)
        regressor.fit(trn[columns], trn["demand_mw"])
        validation_scores[name] = errors(valid["demand_mw"], regressor.predict(valid[columns]))
        trained[name] = regressor
    selected = min(features, key=lambda key: validation_scores[key]["mae_mw"])

    # Validation model is selected without accessing the test targets.
    # Refit with data available before July 1, 2026 for the held-out test.
    pre_test = f[(f["timestamp_utc"] < val_cut) & eligible]
    final_model = model(n_estimators)
    final_model.fit(pre_test[features[selected]], pre_test["demand_mw"])
    preds = final_model.predict(held[features[selected]])
    baseline_preds = held[baseline_col].to_numpy()
    report = {
        "task": "Rolling 24-hour-ahead 15-minute Finnish demand prediction",
        "data_availability_assumptions": [
            "Historical demand is assumed available immediately at timestamp t - 24h; actual Fingrid release latency must be checked before deployment.",
            "Weather features use ERA5 observations from at least 14 days before each target; ERA5 is a reanalysis product and not an archived weather forecast issued at prediction time.",
            "Historical reanalysis vintages are not reconstructed; operational forecasts need a point-in-time weather provider.",
        ],
        "train_end_exclusive_utc": train_cut.isoformat(),
        "validation_start_utc": train_cut.isoformat(),
        "test_start_utc": val_cut.isoformat(),
        "test_end_exclusive_utc": end_cut.isoformat(),
        "selection_rule": "Choose smallest validation MAE; test unused until selection and refitting",
        "feature_sets": {key: list(cols) for key, cols in features.items()},
        "training_rows": len(trn),
        "validation_rows": len(valid),
        "validation_scores": validation_scores,
        "selected_candidate": selected,
        "refit_pretest_rows": len(pre_test),
        "test_rows": len(held),
        "test_model": errors(held["demand_mw"], preds),
        "reference_baseline_name": winner_baseline,
        "test_selected_baseline_same_rows": errors(held["demand_mw"], baseline_preds),
    }
    result = pd.DataFrame({
        "timestamp_utc": held["timestamp_utc"].to_numpy(),
        "actual_mw": held["demand_mw"].to_numpy(),
        "model_mw": preds,
        "selected_baseline_mw": baseline_preds,
    })
    return report, result, final_model


def save_plot(predictions: pd.DataFrame, output: Path) -> None:
    from datetime import timedelta

    start = predictions["timestamp_utc"].min()
    view = predictions.loc[predictions["timestamp_utc"] < start + timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(view["timestamp_utc"], view["actual_mw"], label="Actual", lw=1.5)
    ax.plot(view["timestamp_utc"], view["model_mw"], label="LightGBM", lw=1.25)
    ax.plot(view["timestamp_utc"], view["selected_baseline_mw"], label="Selected baseline", lw=1.1, alpha=0.75)
    ax.set(title="GridWise: 24-hour-ahead demand forecast — first test week", ylabel="Demand (MW)", xlabel="Time (UTC)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate first 24-hour LightGBM forecast against fixed baseline")
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/consumption_2024_2026.csv")
    parser.add_argument("--weather", type=Path, default=ROOT / "data/raw/open_meteo_temperature_2024_2026.csv")
    parser.add_argument("--baseline-report", type=Path, default=ROOT / "reports/baselines_2026.json")
    parser.add_argument("--validation-start", default="2026-01-01")
    parser.add_argument("--test-start", default="2026-07-01")
    parser.add_argument("--end", default="2026-11-01")
    parser.add_argument("--estimators", type=int, default=350)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/lightgbm_2026.json")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/model_outputs/lightgbm_predictions_2026.csv")
    parser.add_argument("--plot", type=Path, default=ROOT / "data/model_outputs/lightgbm_first_week.png")
    parser.add_argument("--model", type=Path, default=ROOT / "data/model_outputs/lightgbm_2026.txt")
    args = parser.parse_args()

    prepared = pd.read_csv(args.input)
    weather = pd.read_csv(args.weather)
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    report, preds, estimator = experiment(
        prepared, weather, baseline_report, args.validation_start, args.test_start, args.end, args.estimators
    )
    for path in [args.report, args.predictions, args.plot, args.model]:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    preds.to_csv(args.predictions, index=False)
    estimator.booster_.save_model(str(args.model))
    save_plot(preds, args.plot)

    print(f"Training rows: {report['training_rows']:,}; validation rows: {report['validation_rows']:,}")
    for name, metric in report["validation_scores"].items():
        print(f"Validation {name}: MAE {metric['mae_mw']:.3f} MW | RMSE {metric['rmse_mw']:.3f} MW")
    print(f"Selected candidate: {report['selected_candidate']} (validation only)")
    print(f"Test paired rows: {report['test_rows']:,}")
    print(f"Test LightGBM: MAE {report['test_model']['mae_mw']:.3f} MW | RMSE {report['test_model']['rmse_mw']:.3f} MW")
    print(f"Test baseline ({report['reference_baseline_name']} on same rows): "
          f"MAE {report['test_selected_baseline_same_rows']['mae_mw']:.3f} MW | "
          f"RMSE {report['test_selected_baseline_same_rows']['rmse_mw']:.3f} MW")
    print(f"Saved report: {args.report}")
    print(f"Saved predictions: {args.predictions}")
    print(f"Saved plot: {args.plot}")
    print(f"Saved model: {args.model}")


if __name__ == "__main__":
    main()
