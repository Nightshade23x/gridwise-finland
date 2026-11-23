"""Phase 12 forward-forecast tests: deterministic, offline, no API calls."""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.forward import (
    feature_table, issue_forecast, combined_demand,
    FORECAST_JSON, FEATURES,
)
from gridwise.api import create_app


def synthetic_data():
    idx = pd.date_range("2024-01-01", "2026-11-21 13:00", freq="15min", tz="UTC")
    v = 9000+450*np.sin(np.arange(len(idx))*2*np.pi/96)+250*np.sin(np.arange(len(idx))*2*np.pi/(96*7))
    return pd.DataFrame({"timestamp_utc": idx, "demand_mw": v})


def test_forward_features_never_use_recent_or_future_actual_values():
    data = synthetic_data()
    issue = pd.Timestamp("2026-11-21 14:15", tz="UTC")
    forecast_index = pd.date_range(issue, periods=96, freq="15min", tz="UTC")
    gap = pd.date_range(data["timestamp_utc"].iloc[-1] + pd.Timedelta(minutes=15),issue-pd.Timedelta(minutes=15),freq="15min",tz="UTC")
    frame = pd.concat([data, pd.DataFrame({"timestamp_utc":gap,"demand_mw":np.nan}),
                       pd.DataFrame({"timestamp_utc":forecast_index,"demand_mw":np.nan})],ignore_index=True)
    before=feature_table(frame).tail(96)[FEATURES]
    changed=frame.copy()
    changed.loc[changed["timestamp_utc"] >= issue-pd.Timedelta(hours=24),"demand_mw"] = 123456.0
    after=feature_table(changed).tail(96)[FEATURES]
    pd.testing.assert_frame_equal(before,after)


def test_issue_makes_96_future_quarters_with_safe_lags():
    data=synthetic_data()
    # Observed until 13:00, issuing 14:15: observation is 75 minutes old.
    report, points, regressor = issue_forecast(data, pd.Timestamp("2026-11-21 14:15",tz="UTC"), estimators=8)
    assert len(points) == 96
    assert report["source_age_hours_at_issue"] == 1.25
    assert points["timestamp_utc"].iloc[0] == pd.Timestamp("2026-11-21 14:15",tz="UTC")
    assert points["timestamp_utc"].iloc[-1] == pd.Timestamp("2026-11-22 14:00",tz="UTC")
    assert (points["predicted_mw"] > 0).all()
    assert report["training_observations"] > 50000
    assert report["preliminary_november_evaluation"]["paired_rows"] > 100
    assert regressor.booster_ is not None


def test_stale_recent_data_fails_closed():
    with pytest.raises(ValueError, match="hours old"):
        issue_forecast(synthetic_data(), pd.Timestamp("2026-11-23 18:00",tz="UTC"),estimators=8)


def test_missing_essential_future_lag_fails_closed():
    data=synthetic_data()
    # First target uses exactly 48h historical lag.
    data.loc[data["timestamp_utc"] == pd.Timestamp("2026-11-19 14:15",tz="UTC"),"demand_mw"] = np.nan
    with pytest.raises(ValueError,match="missing historical observations"):
        issue_forecast(data,pd.Timestamp("2026-11-21 14:15",tz="UTC"),estimators=8,evaluate_november=False)


def test_api_absent_forward_artifact_is_clear(tmp_path):
    r=TestClient(create_app(tmp_path)).get("/api/forward-forecast")
    assert r.status_code==200
    assert r.json()["status"]=="not_generated"
    assert r.json()["points"]==[]


def test_api_reads_saved_artifact(tmp_path):
    path=tmp_path/FORECAST_JSON
    path.parent.mkdir(parents=True)
    idx=pd.date_range("2026-11-22",periods=96,freq="15min",tz="UTC")
    obj={"kind":"experimental_forward_forecast","status":"ok","issued_at_utc":"2026-11-21T23:00:00+00:00",
         "resolution_minutes":15,"model":"LightGBM demand-only","points":
         [{"timestamp_utc":i.isoformat(),"predicted_mw":9000.0,"baseline_mw":9100.0} for i in idx]}
    path.write_text(json.dumps(obj),encoding="utf-8")
    response=TestClient(create_app(tmp_path)).get("/api/forward-forecast")
    assert response.status_code==200
    assert len(response.json()["points"])==96
    obj["points"][5]["predicted_mw"]=float("nan")
    path.write_text(json.dumps(obj),encoding="utf-8")
    assert TestClient(create_app(tmp_path)).get("/api/forward-forecast").status_code==503


def test_combine_preserves_history_and_rejects_conflicts(tmp_path):
    history=tmp_path/"data/processed/consumption_2024_2026.csv"
    history.parent.mkdir(parents=True)
    idx=pd.date_range("2026-10-31 23:30",periods=2,freq="15min",tz="UTC")
    pd.DataFrame({"timestamp_utc":idx,"demand_mw":[8000,8100]}).to_csv(history,index=False)
    recent=tmp_path/"data/live/consumption_recent.csv"
    recent.parent.mkdir(parents=True)
    idx2=pd.date_range("2026-11-01",periods=2,freq="15min",tz="UTC")
    pd.DataFrame({"timestamp_utc":idx2,"demand_mw":[8200,8300]}).to_csv(recent,index=False)
    assert combined_demand(tmp_path)["demand_mw"].tolist()==[8000,8100,8200,8300]
    pd.DataFrame({"timestamp_utc":idx2.insert(0,idx[-1]),"demand_mw":[9999,8200,8300]}).to_csv(recent,index=False)
    with pytest.raises(ValueError,match="Conflicting"):
        combined_demand(tmp_path)
