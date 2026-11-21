"""Selected-period explanations are data-driven and never confuse chart aggregation with MAE."""
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from gridwise.api import create_app


def client_with_series(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/model_outputs").mkdir(parents=True)
    times = pd.date_range("2026-07-01", periods=8, freq="15min", tz="UTC")
    pd.DataFrame({"timestamp_utc":times, "demand_mw":[100., 150., None, 120., 200., 140., 160., 80.]}).to_csv(
        tmp_path / "data/processed/consumption_2024_2026.csv", index=False
    )
    pd.DataFrame({
        "timestamp_utc": times,
        "actual_mw":[100,200,300,400,500,600,700,800],
        "model_mw":[110,190,290,410,510,590,690,790],
        "selected_baseline_mw":[130,170,270,430,530,570,730,770],
    }).to_csv(tmp_path / "data/model_outputs/lightgbm_predictions_2026.csv",index=False)
    return TestClient(create_app(tmp_path))


def test_history_explains_selected_range(tmp_path):
    client = client_with_series(tmp_path)
    data = client.get("/api/insights",params={"kind":"history","start":"2026-07-01","end":"2026-07-02"}).json()
    assert data["status"] == "ok"
    assert data["total_intervals"] == 8
    assert "135.7 MW" in data["headline"] # valid demand mean: 950 / 7
    assert data["metrics"][2]["value"] == "87.5%"
    assert "1 are missing" in data["details"][2]
    assert "2026-07-01 01:00 UTC" in data["details"][0]
    assert "not the cause" in data["implication"]


def test_forecast_insights_use_raw_paired_rows_and_bias(tmp_path):
    client = client_with_series(tmp_path)
    data=client.get("/api/insights",params={"kind":"test","start":"2026-07-01","end":"2026-07-02"}).json()
    assert data["status"]=="ok"
    assert data["paired_intervals"]==8
    assert data["metrics"][0]["value"]=="10.0 MW"
    assert data["metrics"][1]["value"]=="30.0 MW"
    assert data["metrics"][2]["value"]=="+66.7%"
    assert "20.0 MW lower MAE" in data["headline"]
    assert "15-minute" in data["details"][0]
    assert "backtest" in data["implication"]
    # The chart's hourly mean averages signed errors to zero for the 1st hour.
    # Error computed after resampling would incorrectly appear to be zero.
    chart=client.get("/api/series",params={"kind":"test","start":"2026-07-01","end":"2026-07-02","interval":"hour"}).json()
    assert chart["count"]==2
    assert chart["points"][0]["actual_mw"] == chart["points"][0]["model_mw"]


def test_insights_no_data_and_bad_date(tmp_path):
    client = client_with_series(tmp_path)
    query={"kind":"test","start":"2026-08-01","end":"2026-08-02"}
    result=client.get("/api/insights",params=query)
    assert result.status_code==200
    assert result.json()["status"]=="no_data"
    query.update(end="2026-07-01")
    assert client.get("/api/insights",params=query).status_code==422


def test_insights_handles_empty_measures_and_no_pairs(tmp_path):
    client = client_with_series(tmp_path)
    project=tmp_path
    # Avoid relying on cache invalidation by creating a new file path/root.
    alt=tmp_path / 'alt'
    (alt / "data/processed").mkdir(parents=True)
    (alt / "data/model_outputs").mkdir(parents=True)
    times=pd.date_range("2026-07-01",periods=2,freq="15min",tz="UTC")
    pd.DataFrame({"timestamp_utc":times,"demand_mw":[np.nan,np.nan]}).to_csv(alt/"data/processed/consumption_2024_2026.csv",index=False)
    pd.DataFrame({"timestamp_utc":times,"actual_mw":[np.nan,np.nan],"model_mw":[10,20],"selected_baseline_mw":[30,40]}).to_csv(alt/"data/model_outputs/lightgbm_predictions_2026.csv",index=False)
    c=TestClient(create_app(alt))
    query={"start":"2026-07-01","end":"2026-07-02"}
    assert c.get("/api/insights",params={**query,"kind":"history"}).json()["status"]=="no_measurements"
    assert c.get("/api/insights",params={**query,"kind":"test"}).json()["status"]=="no_pairs"


def test_dashboard_includes_explainer_and_script_fetch(tmp_path):
    client = client_with_series(tmp_path)
    html = client.get("/").text
    assert 'id="insights-headline"' in html
    assert 'id="insights-details"' in html
    assert 'id="insights-implication"' in html
    assert "/api/insights" in client.get("/static/app.js").text
