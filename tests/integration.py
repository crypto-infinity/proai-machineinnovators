import pytest
import subprocess
import time
import requests

API_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session", autouse=True)
def start_uvicorn():
    proc = subprocess.Popen(
        ["uvicorn", "api.backend:app"]
    )

    # Check for server startup actively
    timeout = 30
    start = time.time()
    while True:
        try:
            requests.get(f"{API_URL}/health")
            break
        except Exception:
            if time.time() - start > timeout:
                proc.terminate()
                proc.wait()
                raise RuntimeError("Uvicorn server didn't load in time.")
            time.sleep(0.5)

    yield  # Tests execution
    proc.terminate()
    proc.wait()


def test_main_page_redirect():
    response = requests.get(f"{API_URL}/", allow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "/health"


def test_inference_valid():
    payload = {
        "input_string": "I love this product!",
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    }
    response = requests.post(f"{API_URL}/inference", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert isinstance(data["score"], float)


def test_inference_valid_custom_model():
    payload = {
        "input_string": "I love this product!",
        "model": "infinitydreams/roberta-base-sentiment-finetuned",
    }
    response = requests.post(f"{API_URL}/inference", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "label" in data
    assert "score" in data
    assert isinstance(data["score"], float)


def test_performance_compute():
    response = requests.post(f"{API_URL}/performance")
    assert response.status_code == 200

    data = response.json()
    assert "accuracy" in data
    assert "precision" in data
    assert "recall" in data
    assert "f1" in data

    assert data["accuracy"] > 0
    assert data["precision"] > 0
    assert data["recall"] > 0
    assert data["f1"] > 0


def test_inference_invalid_model():
    payload = {"input_string": "Test", "model": "modello-invalido"}
    response = requests.post(f"{API_URL}/inference", json=payload)
    assert response.status_code == 500
    assert "detail" in response.json()
