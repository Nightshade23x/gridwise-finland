"""Freshness metadata is sourced from real local files, never response fetch time."""

from datetime import datetime, timezone
from fastapi.testclient import TestClient
from gridwise.api import create_app
from test_api import _client


def test_summary_exposes_saved_file_modification_times(tmp_path):
    client=_client(tmp_path)
    response=client.get("/api/summary")
    assert response.status_code==200
    times=response.json()["saved_artifact_times_utc"]
    assert set(times)=={"history_file_modified","backtest_file_modified"}
    assert all(datetime.fromisoformat(v).tzinfo is not None for v in times.values())


def test_series_exposes_file_time_even_without_results(tmp_path):
    client=_client(tmp_path)
    for start,end in (("2026-07-01","2026-07-02"),("2025-01-01","2025-01-02")):
        response=client.get("/api/series",params={"kind":"test","start":start,"end":end})
        assert response.status_code==200
        obj=response.json()
        assert datetime.fromisoformat(obj["source_file_modified_utc"]).tzinfo is not None


def test_dashboard_has_source_aware_indicators_and_no_automatic_verification(tmp_path):
    client=TestClient(create_app(tmp_path))
    body=client.get("/").text
    for name in ("summary","recent","forecast","weather","verification","official","explore"):
        assert f'id="{name}-updated"' in body
    js=client.get("/static/app.js").text
    assert 'verified_at_utc' in js
    assert 'setInterval(' in js
    assert 'cache:"no-store"' in js
    assert 'python -m gridwise.official verify' in js
