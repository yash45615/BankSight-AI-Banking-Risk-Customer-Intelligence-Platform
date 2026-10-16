"""Tests for BankSight AI's prediction API."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from banksight import api
from banksight.ml import FEATURE_COLUMNS


class FakeModel:
    """Small deterministic model for endpoint tests."""

    def predict_proba(self, frame) -> np.ndarray:
        assert list(frame.columns) == FEATURE_COLUMNS
        return np.array([[0.25, 0.75]])


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[TestClient]:
    """Provide an isolated API test client."""
    monkeypatch.setattr(
        api,
        "PREDICTION_LOG_PATH",
        tmp_path / "events.jsonl",
    )

    with TestClient(api.app) as test_client:
        test_client.app.state.model_bundle = {
            "model": FakeModel(),
            "feature_columns": FEATURE_COLUMNS.copy(),
            "threshold": 0.50,
            "selected_model": "test-model",
        }
        yield test_client


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "age": 35,
        "annual_income": 75000,
        "employment_years": 8,
        "loan_amount": 15000,
        "interest_rate": 10.5,
        "existing_debt": 5000,
        "credit_history_years": 10,
        "late_payments": 1,
        "loan_purpose": "education",
    }


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["selected_model"] == "test-model"


def test_predict_returns_probability(
    client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["default_probability"] == 0.75
    assert body["predicted_default"] == 1
    assert body["threshold"] == 0.5
    assert body["request_id"]


def test_prediction_creates_metadata_log(
    client: TestClient,
    valid_payload: dict[str, Any],
    tmp_path: Path,
) -> None:
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200
    log_path = tmp_path / "events.jsonl"
    assert log_path.exists()

    content = log_path.read_text(encoding="utf-8")
    assert "default_probability" in content
    assert "annual_income" not in content


def test_invalid_age_is_rejected(
    client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["age"] = 12

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_extra_fields_are_rejected(
    client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    valid_payload["customer_name"] = "Example"

    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 422


def test_missing_model_returns_service_unavailable(
    client: TestClient,
    valid_payload: dict[str, Any],
) -> None:
    client.app.state.model_bundle = None
    original_path = api.MODEL_PATH

    try:
        api.MODEL_PATH = Path("missing-model-file.joblib")
        response = client.post("/predict", json=valid_payload)
    finally:
        api.MODEL_PATH = original_path

    assert response.status_code == 503
