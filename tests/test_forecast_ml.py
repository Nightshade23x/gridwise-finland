import numpy as np
import pandas as pd

from gridwise.baselines import benchmark
from gridwise.forecast_ml import WEATHER_LAG_STEPS, build_features, experiment


def inputs(days=110):
    idx = pd.date_range("2025-12-01", periods=days * 96, freq="15min", tz="UTC")
    i = np.arange(len(idx))
    weekday = idx.tz_convert("Europe/Helsinki").dayofweek.to_numpy()
    demand = 9000 + 600 * np.sin(i * 2 * np.pi / 96) - 250 * (weekday >= 5) + 0.05 * i
    demand = demand + np.random.default_rng(9).normal(0, 10, len(idx))
    prepared = pd.DataFrame({"timestamp_utc": idx, "demand_mw": demand})
    hourly = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    weather = pd.DataFrame({"timestamp_utc": hourly,
                            "temp_helsinki_c": np.sin(np.arange(len(hourly)) / 100) * 10})
    return prepared, weather


def test_features_use_only_old_demand_and_two_week_old_weather():
    prepared, weather = inputs()
    features, cols = build_features(prepared, weather)
    row = 3000
    assert features.loc[row, "demand_lag_24h"] == prepared.loc[row - 96, "demand_mw"]
    assert features.loc[row, "demand_lag_48h"] == prepared.loc[row - 192, "demand_mw"]
    assert "temp_helsinki_c_lag_14d" in cols
    assert features.loc[row, "temp_helsinki_c_lag_14d"] == weather.loc[(row-WEATHER_LAG_STEPS)//4, "temp_helsinki_c"]
    assert np.isnan(features.loc[96, "temp_helsinki_c_lag_14d"])

    changed = prepared.copy()
    changed.loc[row, "demand_mw"] = 1e8
    unchanged, _ = build_features(changed, weather)
    demand_cols = [c for c in features.columns if c.startswith("demand_") and c != "demand_mw"]
    assert features.loc[row, demand_cols].equals(unchanged.loc[row, demand_cols])

    changed_weather = weather.copy()
    changed_weather.loc[row // 4, "temp_helsinki_c"] = 9000.0
    new_features, _ = build_features(prepared, changed_weather)
    assert features.loc[row, cols].equals(new_features.loc[row, cols])


def test_chronological_experiment_and_paired_evaluation():
    prepared, weather = inputs()
    val_start, test_start, end = "2026-01-15", "2026-02-01", "2026-03-10"
    baseline, _ = benchmark(prepared, val_start, test_start, end)
    report, preds, fitted = experiment(
        prepared, weather, baseline, val_start, test_start, end, n_estimators=15
    )
    assert report["selected_candidate"] in {"calendar_and_demand", "calendar_demand_weather"}
    assert report["test_rows"] == len(preds)
    assert len(preds) > 100
    assert report["test_start_utc"] == "2026-02-01T00:00:00+00:00"
    assert pd.to_datetime(preds["timestamp_utc"], utc=True).min() >= pd.Timestamp(test_start, tz="UTC")
    assert "test_model" in report and "test_selected_baseline_same_rows" in report
    assert report["test_model"]["mae_mw"] >= 0
    assert fitted.n_features_in_ == len(report["feature_sets"][report["selected_candidate"]])
