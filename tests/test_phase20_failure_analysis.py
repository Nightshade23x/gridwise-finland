"""Deterministic offline checks for Phase 20 failure diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.failure_analysis import OUTPUT, ROWS_CSV, diagnose_issuance, run_analysis
from gridwise.refresh import LIVE_CSV, atomic_csv
from gridwise.verify import archive_forecast, archived_forecasts
from gridwise.weather_forward import ARCHIVE


def forecast(issued="2026-11-23T13:11:00Z", value=90.0):
    stamp = pd.Timestamp(issued)
    first = stamp.floor("15min") + pd.Timedelta(minutes=15)
    target = pd.date_range(first, periods=96, freq="15min", tz="UTC")
    return {"kind": "experimental_forward_forecast", "status": "ok", "resolution_minutes": 15,
            "issued_at_utc": stamp.isoformat(), "first_target_utc": target[0].isoformat(),
            "last_target_utc": target[-1].isoformat(),
            "points": [{"timestamp_utc": t.isoformat(), "predicted_mw": value,
                        "baseline_mw": 105.0} for t in target]}


def weather(issued="2026-11-24T13:31:00Z"):
    stamp = pd.Timestamp(issued)
    target = pd.date_range(stamp.floor("15min") + pd.Timedelta(minutes=15),
                           periods=96, freq="15min", tz="UTC")
    return {"kind": "weather_forward_candidate", "status": "ok", "schema_version": 1,
            "issued_at_utc": stamp.isoformat(), "first_target_utc": target[0].isoformat(),
            "last_target_utc": target[-1].isoformat(), "resolution_minutes": 15,
            "preferred_strategy": "demand_weather", "baseline_name": "blend_50_50",
            "points": [{"timestamp_utc": t.isoformat(), "demand_weather_mw": 108.0,
                        "demand_only_mw": 102.0, "baseline_mw": 105.0,
                        "preferred_mw": 108.0} for t in target]}


def measurements(doc, *, count=96, omit=(), actual=100.0):
    stamps = [pd.Timestamp(p["timestamp_utc"]) for i, p in enumerate(doc["points"][:count]) if i not in omit]
    return pd.DataFrame({"timestamp_utc": stamps, "demand_mw": [actual]*len(stamps)})


def save_weather(root: Path, doc: dict):
    folder = root / ARCHIVE
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (pd.Timestamp(doc["issued_at_utc"]).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_signed_bias_and_absolute_errors_differ():
    doc = forecast()
    actual = measurements(doc)
    item, rows = diagnose_issuance("forward", "abc", doc, actual,
                                  as_of=pd.Timestamp("2026-11-26T00:00Z"))
    assert item["status"] == "complete"
    assert item["coverage"] == {"eligible": 96, "matched": 96, "missing_eligible": 0, "total": 96}
    assert item["methods"]["lightgbm"]["mae_mw"] == 10
    assert item["methods"]["lightgbm"]["bias_mw"] == -10
    assert item["methods"]["baseline"]["bias_mw"] == 5
    assert item["methods"]["lightgbm"]["overprediction_pct"] == 0
    assert sum(x["matched"] for x in item["by_horizon"]) == 96
    assert sum(x["matched"] for x in item["by_utc_hour"]) == 96
    assert len(item["by_actual_demand_level"]) == 2 or len(item["by_actual_demand_level"]) == 2
    assert all(x["signed_error_mw"] == -10 for x in item["largest_errors"])
    assert "underprediction" in item["flags"][0]
    assert len(rows) == 96


def test_maturity_waits_until_end_of_interval_then_buffer():
    doc = forecast()
    first = pd.Timestamp(doc["points"][0]["timestamp_utc"])
    before, _ = diagnose_issuance("forward", "id", doc, measurements(doc),
                                  as_of=first+pd.Timedelta(minutes=74))
    assert before["status"] == "pending"
    assert before["methods"]["lightgbm"]["mae_mw"] is None
    exact, _ = diagnose_issuance("forward", "id", doc, measurements(doc),
                                 as_of=first+pd.Timedelta(minutes=75))
    assert exact["status"] == "partial" and exact["coverage"]["matched"] == 1


def test_missing_measurements_are_not_zero_and_partial_status_is_explicit():
    doc = forecast()
    first = pd.Timestamp(doc["points"][0]["timestamp_utc"])
    report, rows = diagnose_issuance("forward", "id", doc,
                                    measurements(doc, count=26, omit={10}),
                                    as_of=first+pd.Timedelta(minutes=25*15+75))
    assert report["coverage"] == {"eligible": 26, "matched": 25, "missing_eligible": 1, "total": 96}
    assert report["status"] == "partial"
    assert report["methods"]["lightgbm"]["mae_mw"] == 10
    assert len(rows) == 25
    assert report["by_horizon"][0]["matched"] == 23


def test_weather_candidate_uses_all_methods_on_same_rows():
    doc = weather()
    item, rows = diagnose_issuance("weather", "w", doc, measurements(doc),
                                  as_of=pd.Timestamp("2026-11-27T00:00Z"))
    assert item["methods"]["demand_weather"]["mae_mw"] == 8
    assert item["methods"]["demand_only"]["mae_mw"] == 2
    assert item["methods"]["baseline"]["mae_mw"] == 5
    assert len({m["count"] for m in item["methods"].values()}) == 1
    assert any("Weather candidate underperformed" in flag for flag in item["flags"])
    assert len(rows) == 96


def test_no_forecasts_writes_explicit_empty_report(tmp_path):
    r = run_analysis(tmp_path, as_of=pd.Timestamp("2026-11-25T00:00Z"))
    assert r["status"] == "no_forecasts" and r["issuance_count"] == 0
    assert (tmp_path/OUTPUT).exists()
    assert (tmp_path/ROWS_CSV).exists()
    assert TestClient(create_app(tmp_path)).get('/api/failure-analysis').json()["status"] == "no_forecasts"


def test_output_keeps_snapshots_unchanged_and_flags_identical_curves(tmp_path):
    a = forecast(issued="2026-11-23T13:11:00Z")
    b = forecast(issued="2026-11-23T13:13:00Z")  # same first target and exact predictions
    path1 = archive_forecast(tmp_path, a)
    path2 = archive_forecast(tmp_path, b)
    before = (path1.read_bytes(), path2.read_bytes())
    atomic_csv(tmp_path/LIVE_CSV, measurements(a))
    r = run_analysis(tmp_path, as_of=pd.Timestamp("2026-11-26T00:00Z"))
    assert r["issuance_count"] == 2
    assert sum(v["identical_prediction_to"] is not None for v in r["issuances"]) == 1
    assert (path1.read_bytes(), path2.read_bytes()) == before
    assert len(archived_forecasts(tmp_path)) == 2
    csv = pd.read_csv(tmp_path/ROWS_CSV)
    assert len(csv) == 192   # two independent rows-of-predictions, no pooled aggregate MAE
    again = run_analysis(tmp_path, as_of=pd.Timestamp("2026-11-26T00:00Z"))
    assert again == r


def test_weather_and_forward_separate_and_api_paginates(tmp_path):
    f, w = forecast(), weather()
    archive_forecast(tmp_path, f)
    save_weather(tmp_path, w)
    both = pd.concat([measurements(f), measurements(w)]).drop_duplicates('timestamp_utc').sort_values('timestamp_utc')
    atomic_csv(tmp_path/LIVE_CSV, both)
    result = run_analysis(tmp_path, as_of=pd.Timestamp("2026-11-27T00:00Z"))
    assert set(item["kind"] for item in result["issuances"]) == {"forward", "weather"}
    client = TestClient(create_app(tmp_path))
    r = client.get('/api/failure-analysis?limit=1')
    assert r.status_code == 200
    assert r.json()["issuance_count"] == 2
    assert len(r.json()["issuances"]) == 1
    assert client.get('/api/failure-analysis?limit=0').status_code == 422
    assert 'curve_sha256' not in r.text


def test_bad_snapshots_fail_closed_without_overwriting(tmp_path):
    doc = forecast()
    doc['points'][0]['predicted_mw'] = float('nan')
    with pytest.raises(ValueError):
        diagnose_issuance('forward', 'id', doc, measurements(forecast()),
                         as_of=pd.Timestamp("2026-11-26T00:00Z"))
    with pytest.raises(ValueError):
        diagnose_issuance('forward', 'id', forecast(), measurements(forecast()),
                         as_of=pd.Timestamp("2026-11-26T00:00Z"), buffer_minutes=-1)


def test_dashboard_panel_and_public_demo_are_safe(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path))
    html=client.get('/').text
    assert 'Where did our forecasts go wrong?' in html
    assert 'failure-issuance' in html
    assert client.get('/api/failure-analysis').json()['status'] == 'not_analyzed'
    monkeypatch.setenv('GRIDWISE_DEMO_MODE','1')
    assert client.get('/api/failure-analysis').json()['status'] == 'not_available_in_public_demo'


def test_api_rejects_invalid_report(tmp_path):
    path = tmp_path/OUTPUT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "status": "ok", "issuance_count": 1,
                                "issuances": [{"kind": "forward", "status": "complete", "coverage": {"eligible": 1, "matched": 2}, "by_horizon": []}]}))
    assert TestClient(create_app(tmp_path)).get('/api/failure-analysis').status_code == 503


def test_finland_hour_groups_are_derived_from_actual_target_dates():
    doc = forecast(issued="2026-11-23T13:11:00Z")
    item, paired = diagnose_issuance(
        "forward", "timezone-test", doc, measurements(doc),
        as_of=pd.Timestamp("2026-11-26T00:00Z")
    )
    assert sum(row["matched"] for row in item["by_finland_hour"]) == len(paired) == 96
    assert item["by_finland_hour"]
    # EET is UTC+2 in November, so 14:00 UTC becomes 16:00 local.
    assert paired.loc[paired["utc_hour"] == 14, "finland_hour"].eq(16).all()
    assert "by_utc_hour" in item  # Retained for older API clients.
