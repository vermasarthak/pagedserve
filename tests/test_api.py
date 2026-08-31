import pytest
from fastapi.testclient import TestClient
from pagedserve.config import EngineConfig
from pagedserve.engine.engine import PagedServeEngine
from pagedserve.server.api import create_app


@pytest.fixture
def api_client(cpu_loaded_model):
    config = EngineConfig(num_blocks=100)
    engine = PagedServeEngine(config=config, loaded_model=cpu_loaded_model)
    app = create_app(engine_instance=engine)
    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data


def test_metrics_endpoint(api_client):
    response = api_client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "requests" in data
    assert "kv_cache" in data


def test_completions_endpoint(api_client):
    payload = {
        "model": "pagedserve",
        "prompt": "Hello world",
        "max_tokens": 5,
        "temperature": 0.0,
    }
    response = api_client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert "text" in data["choices"][0]


def test_chat_completions_endpoint(api_client):
    payload = {
        "model": "pagedserve",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }
    response = api_client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_completions_streaming_endpoint(api_client):
    payload = {
        "model": "pagedserve",
        "prompt": "Hello world",
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": True,
    }
    response = api_client.post("/v1/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = response.text.split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) > 0


def test_validation_errors(api_client):
    # Invalid temperature
    payload = {"prompt": "test", "temperature": -1.0}
    response = api_client.post("/v1/completions", json=payload)
    assert response.status_code == 422

    # Empty prompt
    payload = {"prompt": ""}
    response = api_client.post("/v1/completions", json=payload)
    assert response.status_code == 422
