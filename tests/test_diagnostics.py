import numpy as np
import pandas as pd
import pytest
from gridwise.baselines import make_predictions
from gridwise.diagnostics import diagnose, metrics


def sample():
    ts = pd.date_range("2026-07-01", periods=96 * 12, freq="15min", tz="UTC")
    v = 8000 + 500 * np.sin(np.arange(len(ts)) * 2 * np.pi / 96) + np.arange(len(ts)) / 10
    data = pd.DataFrame({"timestamp_utc": ts, "demand_mw": v})
    baseline = make_predictions(data)
    pred = baseline.iloc[672:].copy()
    saved = pd.DataFrame({
        "timestamp_utc": pred["timestamp_utc"],
        "actual_mw": pred["demand_mw"],
        "model_mw": pred["demand_mw"] + 12,
        "selected_baseline_mw": pred["pred_yesterday_mw"],
    })
    return data, saved, {"selected_baseline": "yesterday"}


def test_paired_comparison_includes_all_baselines_and_month():
    prepared, saved, baseline = sample()
    report, paired = diagnose(prepared, saved, baseline)
    assert len(paired) == len(saved)
    assert report["overall"]["lightgbm"]["mae_mw"] == 12.0
    assert set(report["overall"]) == {"lightgbm", "yesterday", "last_week", "blend_50_50"}
    assert report["selected_baseline_from_validation"] == "yesterday"
    assert "2026-07" in report["monthly"]


def test_rejects_mismatched_ground_truth():
    prepared, saved, baseline = sample()
    saved.loc[saved.index[0], "actual_mw"] += 25
    with pytest.raises(ValueError, match="Saved actuals disagree"):
        diagnose(prepared, saved, baseline)


def test_rejects_inconsistent_selected_baseline():
    prepared, saved, baseline = sample()
    saved.loc[saved.index[0], "selected_baseline_mw"] += 25
    with pytest.raises(ValueError, match="Saved selected baseline disagrees"):
        diagnose(prepared, saved, baseline)


def test_missing_prediction_rows_excluded_in_shared_comparison():
    prepared, saved, baseline = sample()
    saved.loc[saved.index[0], "model_mw"] = np.nan
    report, paired = diagnose(prepared, saved, baseline)
    assert report["excluded_unpaired_rows"] == 1
    assert len(paired) == len(saved) - 1


def test_metrics_signs_and_p90():
    m = metrics(pd.Series([10., 20.]), pd.Series([12., 18.]))
    assert m["mae_mw"] == 2
    assert m["rmse_mw"] == 2
    assert m["bias_mw"] == 0
    assert m["p90_absolute_error_mw"] == 2
