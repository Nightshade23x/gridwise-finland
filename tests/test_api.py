"""Phase 9 API regression tests with synthetic files, no network required."""
import pandas as pd
from fastapi.testclient import TestClient

from gridwise.api import create_app


def _client(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/model_outputs").mkdir(parents=True)
    times = pd.date_range("2026-07-01", periods=96 * 3, freq="15min", tz="UTC")
    demand = [float(8000+i % 96) for i in range(len(times))]
    prepared = pd.DataFrame({"timestamp_utc":times,"demand_mw":demand})
    prepared.loc[2,"demand_mw"] = float("nan")
    prepared.to_csv(tmp_path / "data/processed/consumption_2024_2026.csv", index=False)
    test = pd.DataFrame({
        "timestamp_utc":times,
        "actual_mw":demand,
        "model_mw":[v+10 for v in demand],
        "selected_baseline_mw":[v+30 for v in demand],
    })
    test.to_csv(tmp_path / "data/model_outputs/lightgbm_predictions_2026.csv",index=False)
    baseline = pd.DataFrame({
        "timestamp_utc":times,
        "pred_yesterday_mw":[v+30 for v in demand],
        "pred_last_week_mw":[v+50 for v in demand],
    })
    baseline.to_csv(tmp_path / "data/model_outputs/baseline_predictions_2026.csv",index=False)
    return TestClient(create_app(tmp_path))


def test_root_and_health(tmp_path):
    client = _client(tmp_path)
    assert client.get("/").status_code == 200
    assert "Historical research dashboard" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/api/health").json()["status"] == "ready"


def test_missing_files_are_friendly(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").json()["status"] == "missing_data"
    response = client.get("/api/summary")
    assert response.status_code == 503
    assert "Missing" in response.json()["detail"]


def test_summary_uses_identical_timestamps(tmp_path):
    client = _client(tmp_path)
    summary = client.get("/api/summary").json()
    assert summary["live_forecast"] is False
    assert summary["history"]["intervals"] == 288
    assert summary["history"]["observed_intervals"] == 287
    assert summary["evaluation"]["paired_rows"] == 288
    assert summary["evaluation"]["metrics"]["lightgbm"]["mae_mw"] == 10
    assert summary["evaluation"]["metrics"]["blend_50_50"]["mae_mw"] == 40


def test_series_and_missing_values(tmp_path):
    client = _client(tmp_path)
    params={"kind":"history","start":"2026-07-01","end":"2026-07-02","interval":"15min"}
    response=client.get("/api/series",params=params)
    assert response.status_code == 200
    points=response.json()["points"]
    assert len(points)==96
    assert points[2]["demand_mw"] is None
    params.update(interval="hour")
    hourly=client.get("/api/series",params=params).json()
    assert hourly["count"]==24
    assert hourly["points"][0]["demand_mw"] is not None


def test_test_series_and_validation(tmp_path):
    client=_client(tmp_path)
    params={"kind":"test","start":"2026-07-01","end":"2026-07-04","interval":"day"}
    result=client.get("/api/series",params=params)
    assert result.status_code==200
    assert result.json()["count"]==3
    assert "model_mw" in result.json()["points"][0]
    params.update(start="2026-07-04",end="2026-07-01")
    assert client.get("/api/series",params=params).status_code==422


def test_too_many_points_and_no_data(tmp_path):
    client=_client(tmp_path)
    params={"kind":"test","start":"2025-01-01","end":"2025-01-02","interval":"hour"}
    assert client.get("/api/series",params=params).json()["points"]==[]
