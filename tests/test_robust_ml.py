import numpy as np
import pandas as pd
import pytest

from gridwise.robust_ml import buffered_features, run_stress_test


def fixture_data(days=90):
    idx = pd.date_range("2025-12-01", periods=96*days, freq="15min", tz="UTC")
    i = np.arange(len(idx))
    demand = 9000 + 550*np.sin(i*2*np.pi/96) + 0.1*i
    prep = pd.DataFrame({"timestamp_utc": idx, "demand_mw": demand})
    hours = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    weather = pd.DataFrame({"timestamp_utc": hours, "temp_helsinki_c": 2 + 8*np.sin(np.arange(len(hours))/200)})
    return prep, weather


def test_buffer_moves_all_demand_features_and_both_baselines():
    prep, weather = fixture_data()
    f0, _ = buffered_features(prep, weather, 0)
    f60, _ = buffered_features(prep, weather, 60)
    row = 4000
    for col in ["demand_lag_24h", "demand_lag_48h", "demand_lag_7d", "demand_24h_history_mean", "demand_7d_history_mean"]:
        assert f60.loc[row, col] == pytest.approx(f0.loc[row-4, col])
    assert f60.loc[row, "baseline_yesterday_mw"] == prep.loc[row-100, "demand_mw"]
    assert f60.loc[row, "baseline_last_week_mw"] == prep.loc[row-676, "demand_mw"]
    assert f0.loc[row, "local_hour_sin"] == f60.loc[row, "local_hour_sin"]
    assert f0.loc[row, "temp_helsinki_c_lag_14d"] == f60.loc[row, "temp_helsinki_c_lag_14d"]


def test_future_demand_cannot_change_older_buffered_input():
    prep, weather = fixture_data()
    before, _ = buffered_features(prep, weather, 60)
    changed = prep.copy()
    row = 4000
    changed.loc[row-96, "demand_mw"] += 100000
    after, _ = buffered_features(changed, weather, 60)
    assert before.loc[row, "demand_lag_24h"] == after.loc[row, "demand_lag_24h"]
    assert before.loc[row, "demand_24h_history_mean"] == after.loc[row, "demand_24h_history_mean"]


def test_bad_buffers_rejected():
    p, w = fixture_data()
    for b in (-15, 7, 2.5):
        with pytest.raises(ValueError):
            buffered_features(p, w, b)
    with pytest.raises(ValueError):
        run_stress_test(p, w, (0, 0), estimators=5)


def test_comparison_uses_same_test_timestamps_and_validation_selection():
    p, w = fixture_data()
    report, preds = run_stress_test(
        p, w, buffers=(0, 30), validation_start="2026-01-15",
        test_start="2026-02-01", end="2026-02-25", estimators=8,
    )
    assert report["common_test_rows"] == len(preds)
    assert len(preds) > 100
    assert preds[["actual_mw", "lightgbm_buffer_0m_mw", "lightgbm_buffer_30m_mw"]].notna().all().all()
    for b in (0, 30):
        r = report["buffer_results"][str(b)]
        winner = min(r["validation_models"], key=lambda k: r["validation_models"][k]["mae_mw"])
        assert r["selected_model"] == winner
        assert r["test_lightgbm"]["mae_mw"] >= 0
