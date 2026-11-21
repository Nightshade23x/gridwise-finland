"""Phase 10 regression tests for environment-configured data root and UI assets."""
from pathlib import Path
from fastapi.testclient import TestClient
from gridwise.api import create_app


def test_env_root_used_by_packaged_app(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDWISE_PROJECT_ROOT", str(tmp_path))
    response = TestClient(create_app()).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "missing_data"
    assert response.json()["files"]["history"] is False


def test_explicit_root_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("GRIDWISE_PROJECT_ROOT", "/invalid/path/for/test")
    response = TestClient(create_app(tmp_path)).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "missing_data"


def test_assets_expose_time_axis_and_export():
    root=Path(__file__).resolve().parents[1]
    dashboard=root/"src/gridwise/dashboard"
    html=(dashboard/"index.html").read_text()
    js=(dashboard/"app.js").read_text()
    assert 'id="download-csv"' in html
    assert "function axisTickLabel(" in js
    assert "const idx=tickCount===1?0:" in js
    assert "function exportChartCSV()" in js
    assert "drawChart(data.points,data.kind,data.interval)" in js


def test_docker_build_context_never_copies_data_or_secrets():
    root=Path(__file__).resolve().parents[1]
    ignore=(root/".dockerignore").read_text().splitlines()
    assert "data" in ignore
    assert ".env" in ignore
    assert ".venv" in ignore
    docker=(root/"Dockerfile").read_text()
    assert "COPY src ./src" in docker
    assert "USER gridwise" in docker
    assert "COPY . ." not in docker
    compose=(root/"compose.yaml").read_text()
    assert "GRIDWISE_PROJECT_ROOT: /app" in compose
    assert "./data/processed:/app/data/processed:ro" in compose
    assert "127.0.0.1:8000:8000" in compose
