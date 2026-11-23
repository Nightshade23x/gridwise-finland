"""Phase 13 forecast verification: deterministic, offline, reproducible."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.forward import FORECAST_JSON
from gridwise.verify import (archive_forecast, archived_forecasts, validate_forecast,
                             verify_one, run_verification, VERIFICATION_JSON)


ISSUE = pd.Timestamp("2026-11-23 13:11:00+00:00")
FIRST = pd.Timestamp("2026-11-23 13:15:00+00:00")


def issued(pred=9000.0, baseline=9200.0):
    ix = pd.date_range(FIRST, periods=96, freq="15min", tz="UTC")
    return {
        "kind": "experimental_forward_forecast", "status": "ok",
        "issued_at_utc": ISSUE.isoformat(),
        "first_target_utc": ix[0].isoformat(), "last_target_utc": ix[-1].isoformat(),
        "resolution_minutes": 15,
        "points": [{"timestamp_utc": stamp.isoformat(), "predicted_mw": pred,
                    "baseline_mw": baseline} for stamp in ix],
    }


def measurements(n=96, actual=9050.0):
    ix = pd.date_range(FIRST, periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"timestamp_utc": ix, "demand_mw": [actual] * n})


def test_archive_roundtrip_idempotent_and_distinct_issue(tmp_path):
    original = issued()
    first = archive_forecast(tmp_path, original)
    assert first == archive_forecast(tmp_path, original)
    assert len(archived_forecasts(tmp_path)) == 1
    newer = issued(pred=9020)
    second = archive_forecast(tmp_path, newer)
    assert second != first
    assert first.read_text() != second.read_text()
    assert len(archived_forecasts(tmp_path)) == 2


def test_archive_detects_modified_prediction(tmp_path):
    path = archive_forecast(tmp_path, issued())
    record = json.loads(path.read_text())
    record["points"][3]["predicted_mw"] = 8000
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="no longer matches"):
        archived_forecasts(tmp_path)


def test_rejects_malformed_and_backdated_forecasts():
    p = issued()
    p["points"][1]["timestamp_utc"] = p["points"][0]["timestamp_utc"]
    with pytest.raises(ValueError, match="quarter-hour"):
        validate_forecast(p)
    p = issued()
    p["issued_at_utc"] = "2026-11-24T00:00:00+00:00"
    with pytest.raises(ValueError, match="future"):
        validate_forecast(p)
    p = issued()
    p["points"][12]["baseline_mw"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_forecast(p)


def test_pending_future_targets_have_no_fake_zero_error():
    report = verify_one("id", issued(), measurements(), as_of=ISSUE)
    assert report["status"] == "pending"
    assert report["matched_targets"] == 0
    assert report["model"]["mae_mw"] is None
    assert report["baseline"]["mae_mw"] is None
    assert report["by_horizon"][0]["model_mae_mw"] is None


def test_partial_scores_only_available_targets_and_both_methods():
    # At 20:15 UTC, with 60m buffer, only 13:15...19:15 are eligible.
    report = verify_one("id", issued(), measurements(),
                        as_of=pd.Timestamp("2026-11-23 20:15:00+00:00"))
    assert report["status"] == "partial"
    assert report["eligible_targets"] == 25
    assert report["matched_targets"] == 25
    assert report["model"]["mae_mw"] == 50.0
    assert report["baseline"]["mae_mw"] == 150.0
    assert report["model"]["bias_mw"] == -50.0
    assert report["mae_reduction_percent"] == 66.67
    assert report["by_horizon"][0]["matched"] == 24
    assert report["by_horizon"][1]["matched"] == 1
    assert report["by_horizon"][2]["matched"] == 0


def test_complete_and_missing_measurement_status():
    after = pd.Timestamp("2026-11-25 00:00:00+00:00")
    full = verify_one("id", issued(), measurements(), as_of=after)
    assert full["status"] == "complete"
    assert full["matched_targets"] == 96
    assert full["model"]["rmse_mw"] == 50.0
    missing = measurements().drop(index=[2, 40, 93])
    incomplete = verify_one("id", issued(), missing, as_of=after)
    assert incomplete["status"] == "incomplete_source_data"
    assert incomplete["eligible_targets"] == 96
    assert incomplete["matched_targets"] == 93
    assert incomplete["missing_eligible_targets"] == 3
    assert incomplete["model"]["mae_mw"] == 50.0


def test_zero_baseline_error_is_not_division_by_zero():
    report = verify_one("id", issued(), measurements(actual=9200),
                        as_of=pd.Timestamp("2026-11-25", tz="UTC"))
    assert report["baseline"]["mae_mw"] == 0.0
    assert report["mae_reduction_percent"] is None


def test_verification_imports_existing_phase12_and_is_repeatable(tmp_path):
    path = tmp_path / FORECAST_JSON
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(issued()), encoding="utf-8")
    recent = tmp_path / "data/live/consumption_recent.csv"
    measurements(96).to_csv(recent, index=False)
    when = pd.Timestamp("2026-11-25", tz="UTC")
    first = run_verification(tmp_path, as_of=when)
    second = run_verification(tmp_path, as_of=when)
    assert first == second
    assert first["forecasts"][0]["status"] == "complete"
    assert first["forecast_count"] == 1
    assert (tmp_path / VERIFICATION_JSON).is_file()
    assert len(list((tmp_path / "data/live/forecast_archive").glob("*.json"))) == 1
    assert json.loads(path.read_text()) == issued()  # original untouched


def test_verification_without_forecasts(tmp_path):
    r = run_verification(tmp_path, as_of=ISSUE)
    assert r["status"] == "no_forecasts"
    assert r["forecasts"] == []


def test_api_missing_report_and_saved_verification(tmp_path):
    client = TestClient(create_app(tmp_path))
    empty = client.get("/api/forecast-verification")
    assert empty.status_code == 200
    assert empty.json()["status"] == "not_verified"
    archive_forecast(tmp_path, issued())
    path = tmp_path / "data/live/consumption_recent.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    measurements().to_csv(path, index=False)
    run_verification(tmp_path, as_of=pd.Timestamp("2026-11-25", tz="UTC"))
    response = client.get("/api/forecast-verification")
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_count"] == 1
    assert data["forecasts"][0]["model"]["mae_mw"] == 50.0
    assert "password" not in response.text


def test_api_rejects_corrupt_report(tmp_path):
    path = tmp_path / VERIFICATION_JSON
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "status": "ok", "forecasts": [{"status": "complete", "matched_targets": 99, "eligible_targets": 96}]}))
    assert TestClient(create_app(tmp_path)).get("/api/forecast-verification").status_code == 503
