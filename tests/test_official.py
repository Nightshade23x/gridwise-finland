import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.forward import FORECAST_JSON
from gridwise.official import (
    OFFICIAL_DATASET_ID, BENCHMARK_REPORT, capture_snapshot, validate_official_records,
    validate_snapshot, benchmark_one, run_benchmark,
)
from gridwise.refresh import LIVE_CSV


def _times():
    return pd.date_range("2026-11-24T12:00:00Z", periods=96, freq="15min")


def _forecast():
    ts = _times()
    return {"kind": "experimental_forward_forecast", "status": "ok", "resolution_minutes": 15,
            "issued_at_utc": "2026-11-24T11:57:00Z",
            "first_target_utc": ts[0].isoformat(), "last_target_utc": ts[-1].isoformat(),
            "points": [{"timestamp_utc": t.isoformat(), "predicted_mw": 10000., "baseline_mw": 10100.} for t in ts]}


def _records():
    return [{"datasetId":166,"startTime":t.isoformat(),
             "endTime":(t+timedelta(minutes=15)).isoformat(), "value":10050.} for t in _times()]


class FakeClient:
    def __init__(self, values=None):
        self.calls = 0
        self.values = _records() if values is None else values

    def fetch_window(self, dataset_id, start, end):
        self.calls += 1
        assert dataset_id == OFFICIAL_DATASET_ID
        assert start == _times()[0].to_pydatetime()
        assert end == (_times()[-1]+timedelta(minutes=15)).to_pydatetime()
        return self.values


def _save_forecast(root):
    dest = root / FORECAST_JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(_forecast()))


def _save_measurements(root, n=96):
    p = root / LIVE_CSV
    p.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"timestamp_utc":_times()[:n], "demand_mw": [10020.]*n}).to_csv(p,index=False)


def _capture(root, now="2026-11-24T11:58:00Z"):
    _save_forecast(root)
    return capture_snapshot(root, now=pd.Timestamp(now), client=FakeClient())


def test_capture_is_content_preserving_and_idempotent(tmp_path):
    _save_forecast(tmp_path)
    client = FakeClient()
    data = capture_snapshot(tmp_path, now=pd.Timestamp("2026-11-24T11:58:00Z"), client=client)
    assert client.calls == 1
    assert len(data["points"]) == 96
    assert data["source_issuance_time_verified"] is False
    newer = capture_snapshot(tmp_path, now=pd.Timestamp("2026-11-24T11:59:00Z"), client=client)
    assert newer == data
    assert client.calls == 1


def test_capture_excludes_targets_already_started(tmp_path):
    payload = _capture(tmp_path, "2026-11-24T12:10:00Z")
    assert len(payload["points"]) == 95
    assert payload["points"][0]["timestamp_utc"] == _times()[1].isoformat()


def test_capture_rejects_stale_issuance(tmp_path):
    _save_forecast(tmp_path)
    with pytest.raises(ValueError, match="too old"):
        capture_snapshot(tmp_path, now=pd.Timestamp("2026-11-24T13:00:00Z"), client=FakeClient())


def test_rejects_conflicting_revisions_and_bad_data():
    rows = _records()
    rows.append({**rows[0], "value": 11000.})
    with pytest.raises(ValueError, match="Conflicting"):
        validate_official_records(rows, _times(), pd.Timestamp("2026-11-24T11:58:00Z"))
    with pytest.raises(ValueError, match="Unexpected official dataset"):
        validate_official_records([{**rows[0], "datasetId": 124}], _times(), pd.Timestamp("2026-11-24T11:58:00Z"))


def test_snapshot_disallows_lookahead(tmp_path):
    data = _capture(tmp_path)
    altered = json.loads(json.dumps(data))
    altered["captured_at_utc"] = "2026-11-24T12:01:00Z"
    with pytest.raises(ValueError, match="Invalid official"):
        validate_snapshot(altered)


def test_scores_pending_partial_complete_on_common_measurements(tmp_path):
    snap = _capture(tmp_path)
    base = _forecast()
    measured = pd.DataFrame({"timestamp_utc":_times(), "demand_mw":10020.})
    pending = benchmark_one(snap,base,measured,now=pd.Timestamp("2026-11-24T12:30:00Z"),buffer_minutes=60)
    assert pending["status"] == "pending" and pending["model"]["mae_mw"] is None
    partial = benchmark_one(snap,base,measured,now=pd.Timestamp("2026-11-24T15:00:00Z"),buffer_minutes=60)
    assert partial["status"] == "partial" and partial["matched_targets"] == 9
    assert partial["model"]["mae_mw"] == 20
    assert partial["fingrid"]["mae_mw"] == 30
    assert partial["reference_baseline"]["mae_mw"] == 80
    complete = benchmark_one(snap,base,measured,now=pd.Timestamp("2026-11-25T14:00:00Z"))
    assert complete["status"] == "complete" and complete["matched_targets"] == 96
    assert complete["model_mae_improvement_vs_fingrid_percent"] == 33.33


def test_missing_measured_values_never_count_as_zero_error(tmp_path):
    snap = _capture(tmp_path)
    actual = pd.DataFrame({"timestamp_utc":_times()[:3], "demand_mw":[10020.]*3})
    report = benchmark_one(snap, _forecast(), actual, now=pd.Timestamp("2026-11-25T14:00:00Z"))
    assert report["status"] == "incomplete_source_data"
    assert report["matched_targets"] == 3 and report["eligible_targets"] == 96


def test_persisted_comparison_and_read_only_api(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/official-benchmark").json()["status"] == "no_snapshots"
    _capture(tmp_path)
    _save_measurements(tmp_path)
    report = run_benchmark(tmp_path,now=pd.Timestamp("2026-11-25T14:00:00Z"))
    assert report["snapshot_count"] == 1
    assert report["comparisons"][0]["status"] == "complete"
    api = client.get("/api/official-benchmark")
    assert api.status_code == 200
    assert api.json()["comparisons"][0]["matched_targets"] == 96
    assert api.json()["comparisons"][0]["fingrid"]["mae_mw"] == 30


def test_empty_snapshot_report(tmp_path):
    report = run_benchmark(tmp_path)
    assert report["status"] == "no_snapshots"
    assert (tmp_path / BENCHMARK_REPORT).exists()
