"""Tests for BankSight AI machine-learning workflow."""

import numpy as np
import pandas as pd
import pytest

from banksight.ml import (
    FEATURE_COLUMNS,
    train_and_evaluate,
    validate_training_data,
)


@pytest.fixture
def training_dataset() -> pd.DataFrame:
    """Build a deterministic synthetic dataset for tests."""
    rng = np.random.default_rng(42)
    rows = 100

    late_payments = rng.integers(0, 5, size=rows)
    loan_amount = rng.integers(2_000, 30_000, size=rows)
    defaults = ((late_payments >= 3) | (loan_amount >= 25_000)).astype(int)

    return pd.DataFrame(
        {
            "customer_id": range(1, rows + 1),
            "age": rng.integers(21, 66, size=rows),
            "annual_income": rng.integers(20_000, 150_000, size=rows),
            "employment_years": rng.integers(0, 35, size=rows),
            "loan_amount": loan_amount,
            "interest_rate": rng.uniform(5, 20, size=rows),
            "existing_debt": rng.integers(0, 40_000, size=rows),
            "credit_history_years": rng.integers(0, 30, size=rows),
            "late_payments": late_payments,
            "loan_purpose": rng.choice(
                ["home", "auto", "education", "personal"],
                size=rows,
            ),
            "default": defaults,
        }
    )


def test_validate_training_data_accepts_valid_data(
    training_dataset: pd.DataFrame,
) -> None:
    validate_training_data(training_dataset)


def test_validate_training_data_rejects_missing_features(
    training_dataset: pd.DataFrame,
) -> None:
    training_dataset = training_dataset.drop(columns=["loan_amount"])

    with pytest.raises(ValueError, match="loan_amount"):
        validate_training_data(training_dataset)


def test_validate_training_data_rejects_invalid_target(
    training_dataset: pd.DataFrame,
) -> None:
    training_dataset.loc[0, "default"] = 3

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_training_data(training_dataset)


def test_training_returns_model_and_valid_metrics(
    training_dataset: pd.DataFrame,
) -> None:
    model, result, predictions = train_and_evaluate(training_dataset)

    assert result.selected_model in {
        "logistic_regression",
        "random_forest",
    }
    assert 0 <= result.test_result["roc_auc"] <= 1
    assert 0 <= result.test_result["average_precision"] <= 1
    assert result.training_rows + result.validation_rows + result.test_rows == (
        result.row_count
    )
    assert len(predictions) == result.test_rows
    assert set(FEATURE_COLUMNS).issubset(predictions.columns) is False
    assert hasattr(model, "predict_proba")


def test_training_rejects_invalid_threshold(
    training_dataset: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="threshold"):
        train_and_evaluate(training_dataset, threshold=1.0)
