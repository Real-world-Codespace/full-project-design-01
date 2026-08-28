from fastapi.testclient import TestClient

from cooling_load.api.app import app


def test_liveness_and_missing_model_readiness(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing-model.joblib"))
    monkeypatch.delenv("MODEL_S3_URI", raising=False)
    with TestClient(app) as client:
        assert client.get("/health/live").json() == {"status": "alive"}
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"] == "Model artifact is not loaded."
