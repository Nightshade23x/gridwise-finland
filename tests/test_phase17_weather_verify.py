"""Phase 17: immutable local candidate snapshots and matched future measurements."""
from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.refresh import atomic_csv, LIVE_CSV
from gridwise.weather_forward import ARCHIVE
from gridwise.weather_verify import (
    OUTPUT, archived_candidates, run_verification, validate_snapshot,
    verify_candidate,
)


def candidate(*, issued="2026-11-24T13:30:30Z"):
    origin = pd.Timestamp(issued)
    first = origin.floor("15min") + timedelta(minutes=15)
    stamps = pd.date_range(first, periods=96, freq="15min", tz="UTC")
    return {"kind":"weather_forward_candidate", "status":"ok", "schema_version":1,
            "issued_at_utc":origin.isoformat(),
            "first_target_utc":stamps[0].isoformat(), "last_target_utc":stamps[-1].isoformat(),
            "resolution_minutes":15, "preferred_strategy":"demand_weather", "baseline_name":"blend_50_50",
            "points":[{"timestamp_utc":t.isoformat(),
                       "demand_only_mw":115.0, "demand_weather_mw":105.0,
                       "baseline_mw":130.0, "preferred_mw":105.0} for t in stamps]}


def measurements(doc, count=96, missing=None):
    missing = set(missing or [])
    targets=[p["timestamp_utc"] for p in doc["points"]]
    return pd.DataFrame({"timestamp_utc":[pd.Timestamp(t) for i,t in enumerate(targets[:count]) if i not in missing],
                         "demand_mw":[100.0 for i in range(count) if i not in missing]})


def save_candidate(root, doc):
    path=root/ARCHIVE/(pd.Timestamp(doc["issued_at_utc"]).strftime("%Y%m%dT%H%M%S%fZ")+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_validate_strict_weather_candidate_schema_and_future_stamps():
    doc=candidate()
    assert len(validate_snapshot(doc))==96
    corrupt=deepcopy(doc)
    corrupt["points"][0]["preferred_mw"]=200
    with pytest.raises(ValueError,match="Preferred"):
        validate_snapshot(corrupt)
    corrupt=deepcopy(doc)
    corrupt["points"][0]["timestamp_utc"]=doc["issued_at_utc"]
    with pytest.raises(ValueError,match="forecast timestamps"):
        validate_snapshot(corrupt)
    corrupt=deepcopy(doc)
    corrupt["points"][1]["demand_weather_mw"]=float("nan")
    with pytest.raises(ValueError,match="finite"):
        validate_snapshot(corrupt)
    corrupt=deepcopy(doc)
    corrupt["points"][10]["timestamp_utc"]=doc["points"][9]["timestamp_utc"]
    with pytest.raises(ValueError,match="forecast timestamps"):
        validate_snapshot(corrupt)


def test_pending_before_first_completed_interval():
    doc=candidate()
    first=pd.Timestamp(doc["points"][0]["timestamp_utc"])
    result=verify_candidate("f"*64, doc, measurements(doc), as_of=first+timedelta(minutes=59))
    assert result["status"]=="pending"
    assert (result["eligible_targets"],result["matched_targets"])==(0,0)
    for m in result["metrics"].values():
        assert m["paired_rows"]==0 and m["mae_mw"] is None


def test_partial_comparison_uses_identical_actual_targets_and_no_zero_fill():
    doc=candidate()
    first=pd.Timestamp(doc["points"][0]["timestamp_utc"])
    # First 30 target intervals are mature, but one true source reading is missing.
    as_of=first+timedelta(minutes=29*15+15+60)
    result=verify_candidate("a"*64,doc,measurements(doc,count=30,missing={12}),as_of=as_of)
    assert result["status"]=="partial"
    assert result["eligible_targets"]==30
    assert result["matched_targets"]==29
    assert result["missing_eligible_targets"]==1
    assert {m["paired_rows"] for m in result["metrics"].values()}=={29}
    assert result["metrics"]["demand_weather"]["mae_mw"]==5
    assert result["metrics"]["demand_only"]["mae_mw"]==15
    assert result["metrics"]["baseline"]["mae_mw"]==30
    assert result["metrics"]["demand_weather"]["bias_mw"]==5
    assert result["weather_mae_reduction_vs_demand_only_pct"]==66.67
    assert result["weather_mae_reduction_vs_baseline_pct"]==83.33
    assert result["by_horizon"][0]["matched_targets"]==23
    assert result["by_horizon"][1]["matched_targets"]==6
    assert result["by_horizon"][2]["matched_targets"]==0


def test_complete_incomplete_source_and_verified_forecast_unchanged(tmp_path):
    doc=candidate()
    original=save_candidate(tmp_path,doc)
    before=original.read_bytes()
    atomic_csv(tmp_path/LIVE_CSV,measurements(doc))
    end=pd.Timestamp(doc["points"][-1]["timestamp_utc"])+timedelta(hours=3)
    report=run_verification(tmp_path,as_of=end)
    assert report["forecast_count"]==1
    assert report["forecasts"][0]["status"]=="complete"
    assert report["forecasts"][0]["matched_targets"]==96
    assert original.read_bytes()==before
    assert (tmp_path/OUTPUT).is_file()
    again=run_verification(tmp_path,as_of=end)
    assert again==report
    atomic_csv(tmp_path/LIVE_CSV,measurements(doc,missing={95}))
    after=run_verification(tmp_path,as_of=end)
    assert after["forecasts"][0]["status"]=="incomplete_source_data"
    assert after["forecasts"][0]["missing_eligible_targets"]==1
    assert original.read_bytes()==before


def test_reject_tampered_archive_filename_and_bad_measured_values(tmp_path):
    doc=candidate()
    path=save_candidate(tmp_path,doc)
    assert len(archived_candidates(tmp_path))==1
    path.rename(path.with_name("incorrect.json"))
    with pytest.raises(ValueError,match="filename"):
        archived_candidates(tmp_path)
    invalid=measurements(doc)
    invalid.loc[0,"demand_mw"]=-10
    with pytest.raises(ValueError,match="Invalid actual"):
        verify_candidate("f"*64,doc,invalid,as_of=pd.Timestamp("2026-11-27T00:00Z"))


def test_no_candidates_report_is_explicit_and_api_safe(tmp_path):
    result=run_verification(tmp_path)
    assert result["status"]=="no_forecasts"
    assert result["forecast_count"]==0
    r=TestClient(create_app(tmp_path)).get("/api/weather-forecast-verification")
    assert r.status_code==200
    assert r.json()["status"]=="no_forecasts"


def test_api_returns_saved_report_not_side_effects(tmp_path):
    doc=candidate()
    path=save_candidate(tmp_path,doc)
    initial=path.read_bytes()
    atomic_csv(tmp_path/LIVE_CSV,measurements(doc))
    run_verification(tmp_path,as_of=pd.Timestamp("2026-11-27T00:00Z"))
    client=TestClient(create_app(tmp_path))
    response=client.get("/api/weather-forecast-verification")
    assert response.status_code==200
    data=response.json()
    assert data["forecast_count"]==1
    assert data["forecasts"][0]["metrics"]["demand_weather"]["mae_mw"]==5
    assert data["forecasts"][0]["preferred_strategy"]=="demand_weather"
    assert "snapshot_sha256" not in data["forecasts"][0]
    assert path.read_bytes()==initial
    r=client.get("/api/weather-forecast-verification?limit=101")
    assert r.status_code==422
    (tmp_path/OUTPUT).write_text('{"not":"a report"}',encoding="utf-8")
    assert client.get("/api/weather-forecast-verification").status_code==503


def test_multiple_weather_issuances_never_pooled(tmp_path):
    a=candidate()
    b=candidate(issued="2026-11-24T13:45:30Z")
    save_candidate(tmp_path,a)
    save_candidate(tmp_path,b)
    # Two overlapping forecasts should produce two separate records, not 192 pooled rows.
    target=pd.to_datetime(sorted({p["timestamp_utc"] for doc in (a,b) for p in doc["points"]}),utc=True)
    atomic_csv(tmp_path/LIVE_CSV,pd.DataFrame({"timestamp_utc":target,"demand_mw":100.0}))
    result=run_verification(tmp_path,as_of=pd.Timestamp("2026-11-27T00:00Z"))
    assert result["forecast_count"]==2
    assert [row["matched_targets"] for row in result["forecasts"]]==[96,96]
    assert result["forecasts"][0]["issued_at_utc"]>result["forecasts"][1]["issued_at_utc"]


def test_dashboard_contains_phase17_controls():
    from gridwise.api import create_app
    page=TestClient(create_app()).get("/").text
    assert 'id="weather-verification-updated"' in page
    assert 'id="weather-verification-summary"' in page
    js=TestClient(create_app()).get("/static/app.js").text
    assert 'updateWeatherVerification' in js
    assert 'python -m gridwise.weather_verify' in js


def test_command_line_reports_saved_path_without_name_error(tmp_path, monkeypatch, capsys):
    # Regression: the verifier used to finish writing, then crash in its last print.
    import sys
    from gridwise.weather_verify import main
    monkeypatch.setattr(sys, "argv", ["weather_verify", "--root", str(tmp_path)])
    main()
    assert (tmp_path / OUTPUT).is_file()
    assert f"Saved: {tmp_path.resolve() / OUTPUT}" in capsys.readouterr().out
