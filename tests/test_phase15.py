"""Phase 15 validation gate and captured weather forecast tests; offline."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.forward import select_forecast_by_validation, issue_forecast, FORECAST_JSON
from gridwise.weather_forecast import capture_weather, parse_hourly, ARCHIVE, LATEST


def test_validation_gate_does_not_use_november_outcomes():
    scores = {"lightgbm":{"mae_mw":776.2}, "blend_50_50":{"mae_mw":595.5}}
    assert select_forecast_by_validation(scores,"blend_50_50") == "blend_50_50"
    scores["lightgbm"]["mae_mw"] = 500
    assert select_forecast_by_validation(scores,"blend_50_50") == "lightgbm"
    scores["lightgbm"]["mae_mw"] = 595.5
    assert select_forecast_by_validation(scores,"blend_50_50") == "blend_50_50"
    scores["lightgbm"]["mae_mw"] = float("nan")
    with pytest.raises(ValueError,match="finite"):
        select_forecast_by_validation(scores,"blend_50_50")


def test_new_forward_issuance_preserves_model_for_existing_verifiers():
    from test_forward import synthetic_data
    data=synthetic_data()
    report, points, _ = issue_forecast(data, pd.Timestamp("2026-11-21 14:15Z"),estimators=8,
                                       evaluate_november=False)
    assert report["preferred_strategy"] in ("lightgbm",report["selected_reference_baseline"])
    assert len(points)==96
    prefer = "predicted_mw" if report["preferred_strategy"] == "lightgbm" else "baseline_mw"
    assert np.allclose(points["preferred_mw"],points[prefer],atol=0.002,rtol=0)
    assert report["preferred_validation_mae_mw"] == report["validation_scores"][report["preferred_strategy"]]["mae_mw"]
    from gridwise.verify import validate_forecast
    assert len(validate_forecast(report))==96


def test_api_exposes_gate_and_rejects_tampered_preference(tmp_path):
    ts=pd.date_range("2026-11-24 12:00Z",periods=96,freq="15min")
    p={"kind":"experimental_forward_forecast", "status":"ok", "resolution_minutes":15,
       "issued_at_utc":"2026-11-24T11:57:00Z", "preferred_strategy":"blend_50_50",
       "preferred_validation_mae_mw": 595.5, "selected_reference_baseline":"blend_50_50",
       "points":[{"timestamp_utc":t.isoformat(),"predicted_mw":10000.0,
                  "baseline_mw":11000.0,"preferred_mw":11000.0} for t in ts]}
    path=tmp_path/FORECAST_JSON
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(p))
    response=TestClient(create_app(tmp_path)).get("/api/forward-forecast")
    assert response.status_code==200
    assert response.json()["preferred_strategy"]=="blend_50_50"
    p["points"][3]["preferred_mw"]=10200.0
    path.write_text(json.dumps(p))
    assert TestClient(create_app(tmp_path)).get("/api/forward-forecast").status_code==503


def raw_weather():
    hours=pd.date_range("2026-11-24 12:00Z",periods=72,freq="h")
    return {"utc_offset_seconds":0,"hourly_units":{"temperature_2m":"°C"},
            "hourly":{"time":[x.strftime("%Y-%m-%dT%H:%M") for x in hours],
                      "temperature_2m":[5.0]*len(hours)}}


class Response:
    status_code=200
    def __init__(self, data): self.data=data
    def raise_for_status(self): pass
    def json(self): return self.data


class FakeSession:
    def __init__(self,payload=None):
        self.payload=raw_weather() if payload is None else payload
        self.calls=[]
    def get(self,url,params,timeout):
        self.calls.append((url,params,timeout))
        return Response(self.payload)


def test_weather_capture_archives_only_future_hourly_predictions(tmp_path):
    session=FakeSession()
    report=capture_weather(tmp_path,session=session,now=pd.Timestamp("2026-11-24 13:05Z"),pause=lambda _:None)
    assert len(report["cities"]) == 5
    assert len(session.calls)==5
    assert all("timezone" in params and params["timezone"]=="UTC" for _,params,_ in session.calls)
    assert all(len(item["hourly"])==24 for item in report["cities"])
    assert report["cities"][0]["hourly"][0]["timestamp_utc"] == "2026-11-24T14:00:00+00:00"
    assert not report["source_model_run_time_verified"]
    assert len(list((tmp_path/ARCHIVE).glob("*.json")))==1
    assert json.loads((tmp_path/LATEST).read_text()) == report


def test_weather_rejects_bad_data_before_writing_files(tmp_path):
    broken=raw_weather()
    broken["hourly"]["temperature_2m"][7]=None
    with pytest.raises(ValueError,match="Missing or implausible"):
        capture_weather(tmp_path,session=FakeSession(broken),now=pd.Timestamp("2026-11-24 13:05Z"),pause=lambda _:None)
    assert not (tmp_path/LATEST).exists()
    assert not (tmp_path/ARCHIVE).exists()


def test_weather_rejects_off_hour_duplicate_and_non_utc():
    raw=raw_weather()
    raw["hourly"]["time"][1]=raw["hourly"]["time"][0]
    with pytest.raises(ValueError,match="time grid"):
        parse_hourly(raw,"helsinki")
    raw=raw_weather();raw["utc_offset_seconds"]=7200
    with pytest.raises(ValueError,match="UTC offset"):
        parse_hourly(raw,"helsinki")


def test_weather_missing_future_hour_fails_closed(tmp_path):
    p=raw_weather()
    p["hourly"]["time"]=p["hourly"]["time"][:1]
    p["hourly"]["temperature_2m"]=p["hourly"]["temperature_2m"][:1]
    with pytest.raises(ValueError,match="Missing future"):
        capture_weather(tmp_path,session=FakeSession(p),now=pd.Timestamp("2026-11-24 13:05Z"),pause=lambda _:None)
    assert not (tmp_path/LATEST).exists()
