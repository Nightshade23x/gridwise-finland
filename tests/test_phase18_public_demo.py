"""Static demo must be sourced from real files, never generate fake measurements."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from gridwise.api import create_app

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location("gridwise_demo_export", ROOT / "scripts/export_public_demo.py")
exporter = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(exporter)


def _data(root):
    start = pd.date_range("2026-07-01", periods=1200, freq="15min", tz="UTC")
    history = pd.DataFrame({"timestamp_utc":start, "demand_mw":range(1200)})
    test = pd.DataFrame({"timestamp_utc":start, "actual_mw":range(1200),
                         "model_mw":range(1200), "selected_baseline_mw":range(1200)})
    baseline = pd.DataFrame({"timestamp_utc":start, "pred_yesterday_mw":range(1200),
                             "pred_last_week_mw":range(1200)})
    files={"history":history,"test":test,"baselines":baseline}
    for kind,relative in exporter.SOURCES.items():
        target=root/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        files[kind].to_csv(target,index=False)
    (root/".env").write_text("FINGRID_API_KEY=DO_NOT_PUBLISH",encoding="utf-8")


def test_exporter_copies_only_whitelisted_historical_data(tmp_path):
    _data(tmp_path)
    dest=tmp_path/"demo_bundle"
    meta=exporter.export_demo(tmp_path,dest)
    assert meta["row_counts"]=={"history":1200,"test":1200,"baselines":1200}
    assert len(pd.read_csv(dest/"data/model_outputs/lightgbm_predictions_2026.csv"))==1200
    assert not (dest/".env").exists()
    assert not (dest/"data/live").exists()
    assert not (dest/"data/raw").exists()
    assert "DO_NOT_PUBLISH" not in (dest/"manifest.json").read_text()


def test_exporter_refuses_missing_or_misleading_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        exporter.export_demo(tmp_path,tmp_path/"demo_bundle")
    _data(tmp_path)
    path=tmp_path/exporter.SOURCES["test"]
    frame=pd.read_csv(path)
    frame=frame.head(5)
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError,match="Too few test predictions"):
        exporter.export_demo(tmp_path,tmp_path/"demo_bundle")


def test_demo_api_provides_label_and_no_private_data(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDWISE_DEMO_MODE","1")
    (tmp_path/"demo_manifest.json").write_text(json.dumps({
        "schema_version":1,"kind":"static_historical_demo",
        "generated_at_utc":"2026-11-25T12:00:00+00:00","source":"Fingrid dataset 124"}))
    client=TestClient(create_app(tmp_path))
    result=client.get('/api/deployment-info')
    assert result.status_code==200
    assert result.json()['mode']=='demo'
    assert result.json()['live_updates_enabled'] is False
    assert 'FINGRID_API_KEY' not in result.text
    assert 'PUBLIC ARCHIVED DEMO' in client.get('/static/app.js').text
    assert 'id="demo-banner"' in client.get('/').text


def test_deployment_files_configured_for_port_and_read_only():
    docker=(ROOT/"Dockerfile.demo").read_text()
    render=(ROOT/"render.yaml").read_text()
    assert 'COPY demo_bundle/data ./data' in docker
    assert '${PORT:-10000}' in docker
    assert 'GRIDWISE_DEMO_MODE=1' in docker
    assert 'dockerfilePath: ./Dockerfile.demo' in render
    assert 'healthCheckPath: /api/health' in render
    assert 'plan: free' in render
    assert '!demo_bundle/data/**' in (ROOT / ".dockerignore").read_text()
    assert '.env' not in docker
