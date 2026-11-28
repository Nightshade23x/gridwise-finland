"""Phase 21: issue-time-only capped level-shift research candidate."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.forward import feature_table
from gridwise.level_shift import (
    _origin_window, backtest_windows, evaluate, live_preview, metrics, prediction,
    CAL_START, CAL_END, CHECK_START, CHECK_END, OFFSET_CAP_MW, REPORT,
)


def source(start="2025-12-01", end="2026-07-03", slope=0.0):
    dates = pd.date_range(start, end, freq="15min", inclusive="left", tz="UTC")
    i = np.arange(len(dates))
    # Deterministic nonconstant synthetic consumption with a weekly structure.
    hourly = np.sin(i * 2 * np.pi / 96)
    weekly = 180 * np.cos(i * 2 * np.pi / 672)
    values = 10400 + 1600 * hourly + weekly + (i/96)*slope
    return pd.DataFrame({"timestamp_utc": dates, "demand_mw": values})


def test_metrics_have_signed_bias_and_absolute_error():
    assert metrics([100, 100], [90, 120]) == {"mae_mw": 15, "rmse_mw": 15.811,
                                                "bias_mw": 5, "paired_targets": 2}
    with pytest.raises(ValueError):
        metrics([100], [float("nan")])


def test_level_shift_clipped_nonnegative_and_monotonically_decays():
    rows = pd.DataFrame({"lag_48h": [100, 100], "offset_lag_48h_mw": [1000, 1000],
                         "lead_hours": [1, 24]})
    got = prediction(rows, "lag_48h", 1, 6)
    assert got[0] > got[1] >= 100
    assert np.allclose(prediction(rows, "lag_48h", 0, 6), [100, 100])
    with pytest.raises(ValueError):
        prediction(rows, "wrong", .5, 6)


def test_anchor_does_not_use_future_outcomes():
    data = source(end="2026-01-10")
    issue = pd.Timestamp("2026-01-05T12:00:00Z")
    a, reason = _origin_window(feature_table(data), issue)
    assert reason is None and a is not None and len(a) == 96
    altered = data.copy()
    altered.loc[(altered.timestamp_utc > issue), "demand_mw"] += 5000
    b, reason = _origin_window(feature_table(altered), issue)
    assert reason is None
    for baseline in ("lag_48h", "lag_7d", "blend_50_50"):
        assert np.allclose(a[f"offset_{baseline}_mw"], b[f"offset_{baseline}_mw"])
        assert np.allclose(a[baseline], b[baseline])
    assert not np.allclose(a.demand_mw, b.demand_mw)


def test_missing_anchor_does_not_silently_impute():
    data = source(end="2026-01-10")
    issue = pd.Timestamp("2026-01-05T12:00:00Z")
    data.loc[(data.timestamp_utc > issue-pd.Timedelta(hours=5)) &
             (data.timestamp_utc <= issue-pd.Timedelta(hours=1)), "demand_mw"] = np.nan
    rows, why = _origin_window(feature_table(data), issue)
    assert rows is None and why == "missing_anchor_observations"


def test_missing_actual_targets_do_not_count_as_zero_error():
    data = source(end="2026-01-10")
    issue = pd.Timestamp("2026-01-05T12:00:00Z")
    data.loc[data.timestamp_utc == issue+pd.Timedelta(hours=3), "demand_mw"] = np.nan
    window, reason = _origin_window(feature_table(data), issue)
    assert reason is None and len(window) == 95
    assert not window.demand_mw.isna().any()
    assert all(window[f"offset_{name}_mw"].abs().max() <= OFFSET_CAP_MW
               for name in ("lag_48h", "lag_7d", "blend_50_50"))


def test_one_issuance_per_day_has_non_overlapping_targets():
    windows, coverage = backtest_windows(source(end="2026-01-10"),
        pd.Timestamp("2026-01-03T00:00Z"), pd.Timestamp("2026-01-08T00:00Z"))
    assert coverage["valid_issuances"] == 5
    assert windows.timestamp_utc.nunique() == len(windows) == 5*96
    assert all(windows.groupby("issue_utc").size() == 96)
    assert windows["origin_anchor_count"].min() >= 16


def test_preview_ignores_future_actuals_and_enforces_freshness():
    issue = pd.Timestamp("2026-01-05T12:05:00Z")
    data = source(end="2026-01-10")
    params = {"baseline":"blend_50_50", "alpha":0.5, "decay_hours":6.0}
    a = live_preview(data, issue, params)
    assert len(a["points"]) == 96 and a["points"][0]["timestamp_utc"] == "2026-01-05T12:15:00+00:00"
    altered = data.copy()
    altered.loc[altered.timestamp_utc > issue-pd.Timedelta(minutes=60), "demand_mw"] = 999999
    b = live_preview(altered, issue, params)
    assert a == b
    stale = data.copy()
    stale.loc[stale.timestamp_utc > issue-pd.Timedelta(hours=9), "demand_mw"] = np.nan
    with pytest.raises(ValueError, match="too old"):
        live_preview(stale, issue, params)


def test_chronological_selection_and_later_check_excluded_period_overlap():
    data = source(start="2025-09-01", slope=9.0)
    report = evaluate(data)
    assert report["selection_period"]["coverage"]["valid_issuances"] > 80
    assert report["later_period"]["coverage"]["valid_issuances"] > 80
    assert report["later_period"]["coverage"]["paired_target_rows"] > 7000
    assert pd.Timestamp(report["selection_period"]["end_exclusive_utc"]) < pd.Timestamp(report["later_period"]["start_utc"])
    assert report["selected_parameters"]["alpha"] in (0,.25,.5,.75,1)
    assert report["later_period"]["candidate"]["paired_targets"] == report["later_period"]["uncorrected_reference"]["paired_targets"]
    assert set(report["later_period"]["other_uncorrected_references"]) == {"lag_48h", "lag_7d", "blend_50_50"}


def test_diagnostic_api_missing_saved_and_demo(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path))
    assert client.get('/api/level-shift').json()["status"] == "not_run"
    path = tmp_path / REPORT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(evaluate(source(start="2025-09-01"))), encoding="utf-8")
    r=client.get('/api/level-shift')
    assert r.status_code == 200 and r.json()["status"] == "evaluated"
    monkeypatch.setenv('GRIDWISE_DEMO_MODE','1')
    assert client.get('/api/level-shift').json()["status"] == "not_available_in_public_demo"


def test_invalid_api_report_fails_closed(tmp_path):
    path = tmp_path / REPORT
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"status":"evaluated","selected_parameters":{}}')
    assert TestClient(create_app(tmp_path)).get('/api/level-shift').status_code == 503
