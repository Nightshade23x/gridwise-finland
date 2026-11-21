"""Offline incremental refresh regression tests: no API key and no network."""
from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gridwise.api import create_app
from gridwise.refresh import (
    LIVE_CSV, STATUS_JSON, fetch_cutoff, read_measurements, run_refresh, valid_new_rows,
)

UTC = timezone.utc
START = datetime(2026, 11, 1, tzinfo=UTC)
NOW = datetime(2026, 11, 2, 13, 10, tzinfo=UTC)


def record(t, v=9000.0, *, duration=15):
    return {
        "datasetId": 124,
        "startTime": t.isoformat(),
        "endTime": (t + timedelta(minutes=duration)).isoformat(),
        "value": v,
    }


class FakeClient:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def fetch_window(self, dataset_id, start, end):
        self.calls.append((dataset_id, start, end))
        rows = [row for t, row in self.mapping.items() if start <= t < end]
        return rows


def test_completed_buffer_excludes_last_hour():
    assert fetch_cutoff(NOW, 60) == datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        fetch_cutoff(NOW, -1)


def test_first_and_second_refresh_are_idempotent(tmp_path):
    source = {START + timedelta(minutes=i * 15):record(START + timedelta(minutes=i * 15), v=8100+i)
              for i in range(24 * 4)}
    fake = FakeClient(source)
    one = run_refresh(tmp_path, fake, NOW, initial_start=START)
    assert one["status"] == "ok"
    assert one["added_rows"] == 96
    assert one["latest_measurement_utc"].startswith("2026-11-01T23:45")
    two = run_refresh(tmp_path, fake, NOW, initial_start=START)
    assert two["added_rows"] == 0
    assert two["stored_rows"] == one["stored_rows"]
    assert len(read_measurements(tmp_path/LIVE_CSV)) == 96
    assert fake.calls[0][1] == START
    assert fake.calls[-1][2] == datetime(2026,11,2,12,tzinfo=UTC)


def test_backfills_missing_and_preserves_old_value(tmp_path):
    t0 = START
    t1 = START + timedelta(minutes=15)
    fake = FakeClient({t0:record(t0, 100.0)})
    run_refresh(tmp_path, fake, NOW)
    fake.mapping[t0] = record(t0, 999.0)  # source revised existing measurement
    fake.mapping[t1] = record(t1, 200.0)  # newly available gap
    result = run_refresh(tmp_path, fake, NOW)
    df = read_measurements(tmp_path/LIVE_CSV)
    assert result["added_rows"] == 1
    assert result["existing_values_preserved_on_conflict"] == 1
    assert df["demand_mw"].tolist() == [100.0, 200.0]


def test_offgrid_and_incomplete_period_are_rejected(tmp_path):
    base = START
    fake = FakeClient({
        base: record(base),
        base + timedelta(seconds=5): record(base + timedelta(seconds=5)),
        base + timedelta(minutes=15): record(base + timedelta(minutes=15), duration=30),
    })
    out=run_refresh(tmp_path, fake, NOW)
    assert out["added_rows"] == 1
    assert out["rejected_offgrid_or_invalid_rows"] == 2


def test_error_preserves_existing_csv_and_reports_failure(tmp_path):
    t=START
    run_refresh(tmp_path, FakeClient({t: record(t)}), NOW)
    data_before = (tmp_path/LIVE_CSV).read_bytes()
    class Broken:
        def fetch_window(self,*args):
            raise RuntimeError("Temporary Fingrid outage")
    with pytest.raises(RuntimeError, match="outage"):
        run_refresh(tmp_path, Broken(), NOW)
    assert (tmp_path/LIVE_CSV).read_bytes() == data_before
    status=json.loads((tmp_path/STATUS_JSON).read_text())
    assert status["status"] == "failed"
    assert status["last_success_utc"] is not None
    assert status["latest_measurement_utc"].startswith("2026-11-01")
    assert not (tmp_path/"data/live/refresh.lock").exists()


def test_api_recent_kind_and_status(tmp_path):
    run_refresh(tmp_path, FakeClient({START:record(START, 9300)}), NOW)
    client=TestClient(create_app(tmp_path))
    result=client.get("/api/refresh-status").json()
    assert result["status"] == "ok"
    assert result["stored_rows"] == 1
    rows=client.get("/api/series", params={"kind":"recent","start":"2026-11-01","end":"2026-11-02","interval":"15min"})
    assert rows.status_code == 200
    assert rows.json()["points"][0]["demand_mw"] == 9300
    insights=client.get("/api/insights", params={"kind":"recent","start":"2026-11-01","end":"2026-11-02"})
    assert insights.status_code == 200
    assert "Average measured demand" in insights.json()["headline"]
    assert "forecast" in insights.json()["implication"]


def test_api_handles_absent_status(tmp_path):
    client=TestClient(create_app(tmp_path))
    status=client.get("/api/refresh-status")
    assert status.status_code == 200
    assert status.json()["status"] == "not_configured"
    recent=client.get("/api/series",params={"kind":"recent","start":"2026-11-01","end":"2026-11-02"})
    assert recent.status_code == 503


def test_cannot_run_concurrently(tmp_path):
    lock=tmp_path / 'data/live/refresh.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text('existing')
    with pytest.raises(RuntimeError,match="Another refresh"):
        run_refresh(tmp_path, FakeClient({}), NOW)
    assert lock.exists()


def test_ui_includes_recent_view(tmp_path):
    from pathlib import Path
    dashboard=Path(__file__).resolve().parents[1] / "src/gridwise/dashboard"
    html=(dashboard/"index.html").read_text()
    js=(dashboard/"app.js").read_text()
    assert 'value="recent"' in html
    assert 'id="recent-latest"' in html
    assert 'updateRecentStatus()' in js
    assert 'recentDateRange()' in js
