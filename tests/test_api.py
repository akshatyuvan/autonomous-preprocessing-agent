"""
tests/test_api.py -- the HTTP API, with every LLM faked. Environment: LOCAL (Mac).
"""
from fastapi.testclient import TestClient

from api.main import app
from tests.test_cleaner import _write_messy_csv

client = TestClient(app)


def _raw_dir(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr("config.RAW_DIR", str(raw))
    return raw


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_run_pipeline_on_a_file_in_the_raw_dir(tmp_path, monkeypatch):
    csv = _raw_dir(tmp_path, monkeypatch) / "messy.csv"
    _write_messy_csv(csv)
    response = client.post("/pipeline/run", json={"dataset_path": str(csv)})
    assert response.status_code == 200
    body = response.json()
    assert body["steps_run"] == ["cleaner", "encoder", "scaler", "feature_selector"]
    assert body["completed"] is True and body["halted"] is False
    assert body["output_path"].endswith(".parquet")
    assert body["decisions"] and body["verdicts"]


def test_paths_outside_the_raw_dir_are_refused(tmp_path, monkeypatch):
    raw = _raw_dir(tmp_path, monkeypatch)
    response = client.post("/pipeline/run", json={"dataset_path": str(raw / ".." / "secret.env")})
    assert response.status_code == 400


def test_missing_file_is_404(tmp_path, monkeypatch):
    raw = _raw_dir(tmp_path, monkeypatch)
    response = client.post("/pipeline/run", json={"dataset_path": str(raw / "nope.csv")})
    assert response.status_code == 404