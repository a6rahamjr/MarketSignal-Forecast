from app.api import create_app


def test_api_health_and_prediction(trained_result, synthetic_market):
    import pytest

    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    client = TestClient(create_app(trained_result.model_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ready", "model_loaded": True}

    candles = (
        synthetic_market.reset_index()
        .assign(date=lambda frame: frame["date"].astype(str))
        .to_dict(orient="records")
    )
    response = client.post("/predict", json={"candles": candles})
    assert response.status_code == 200
    payload = response.json()
    assert payload["direction"] in {"up", "down", "uncertain"}
    assert 0 <= payload["probability_up"] <= 1


def test_api_key_is_enforced(monkeypatch, trained_result, synthetic_market):
    import pytest

    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MARKETSIGNAL_API_KEY", "test-secret")
    client = TestClient(create_app(trained_result.model_path))
    candles = (
        synthetic_market.reset_index()
        .assign(date=lambda frame: frame["date"].astype(str))
        .to_dict(orient="records")
    )
    assert client.post("/predict", json={"candles": candles}).status_code == 401
    response = client.post(
        "/predict",
        headers={"X-API-Key": "test-secret"},
        json={"candles": candles},
    )
    assert response.status_code == 200


def test_production_requires_api_key(monkeypatch, trained_result):
    import pytest

    monkeypatch.setenv("MARKETSIGNAL_ENV", "production")
    monkeypatch.delenv("MARKETSIGNAL_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="required"):
        create_app(trained_result.model_path)
