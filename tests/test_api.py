"""API tests. Model loading is stubbed directly via model_state so these
run without a trained model, Kaggle credentials, or network access
(important for CI, which has none of those)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, model_state

SAMPLE_PAYLOAD = {
    "Item_Weight": 9.3,
    "Item_Visibility": 0.016,
    "Item_MRP": 249.8,
    "Outlet_Age": 14,
    "Item_Fat_Content": "Low Fat",
    "Item_Type": "Dairy",
    "Item_Category": "Food",
    "Outlet_Size": "Medium",
    "Outlet_Location_Type": "Tier 1",
    "Outlet_Type": "Supermarket Type1",
}


class StubModel:
    def predict(self, df):
        return np.array([1234.56])


@pytest.fixture
def client():
    # TestClient(app) without `with` does not trigger the real lifespan
    # (which loads from the MLflow registry), so we populate model_state
    # directly instead.
    model_state["model"] = StubModel()
    model_state["model_name"] = "stub-model"
    model_state["model_version"] = "1"
    yield TestClient(app)
    model_state.clear()


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "stub-model"


def test_health_when_model_missing():
    model_state.clear()
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "model not loaded"


def test_predict_returns_expected_shape(client):
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_sales"] == 1234.56
    assert body["model_name"] == "stub-model"
    assert "latency_ms" in body


def test_predict_rejects_invalid_category(client):
    bad_payload = {**SAMPLE_PAYLOAD, "Outlet_Type": "Not A Real Type"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_without_model_returns_503():
    model_state.clear()
    response = TestClient(app).post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 503
