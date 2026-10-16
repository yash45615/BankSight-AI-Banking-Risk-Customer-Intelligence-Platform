"""Tests for BankSight AI exploratory data analysis."""

import pandas as pd
import pytest

from banksight.eda import analyze_dataset


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(1, 9),
            "age": [22, 25, 30, 35, 40, 45, 50, 55],
            "annual_income": [
                30000,
                40000,
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
            ],
            "employment_years": [1, 2, 5, 8, 10, 12, 15, 20],
            "loan_amount": [
                5000,
                7000,
                10000,
                12000,
                15000,
                18000,
                20000,
                25000,
            ],
            "interest_rate": [8, 9, 10, 11, 12, 13, 14, 15],
            "existing_debt": [
                1000,
                2000,
                3000,
                4000,
                5000,
                6000,
                7000,
                8000,
            ],
            "credit_history_years": [1, 2, 3, 5, 7, 9, 12, 15],
            "late_payments": [0, 0, 1, 0, 2, 1, 3, 4],
            "loan_purpose": [
                "education",
                "home",
                "auto",
                "personal",
                "education",
                "home",
                "auto",
                "personal",
            ],
            "default": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )


def test_analyze_dataset_returns_expected_dimensions(
    sample_dataset: pd.DataFrame,
) -> None:
    result = analyze_dataset(sample_dataset)

    assert result.row_count == 8
    assert result.column_count == 11
    assert result.duplicate_rows == 0


def test_default_rate_is_calculated_correctly(
    sample_dataset: pd.DataFrame,
) -> None:
    result = analyze_dataset(sample_dataset)

    assert result.overall_default_rate == 0.5
    assert len(result.default_rate_ci_95) == 2
    assert 0 <= result.default_rate_ci_95[0] <= 0.5
    assert 0.5 <= result.default_rate_ci_95[1] <= 1


def test_missing_values_are_counted(
    sample_dataset: pd.DataFrame,
) -> None:
    sample_dataset.loc[0, "annual_income"] = float("nan")

    result = analyze_dataset(sample_dataset)

    assert result.missing_values["annual_income"] == 1


def test_grouped_default_rates_are_returned(
    sample_dataset: pd.DataFrame,
) -> None:
    result = analyze_dataset(sample_dataset)

    assert len(result.default_rates_by_purpose) == 4
    assert all(
        0 <= item["default_rate"] <= 1 for item in result.default_rates_by_purpose
    )


def test_statistical_tests_are_included(
    sample_dataset: pd.DataFrame,
) -> None:
    result = analyze_dataset(sample_dataset)

    assert "loan_purpose_vs_default" in result.statistical_tests
    assert "late_payments_by_default" in result.statistical_tests


def test_invalid_target_is_rejected(
    sample_dataset: pd.DataFrame,
) -> None:
    sample_dataset.loc[0, "default"] = 2

    with pytest.raises(ValueError, match="only 0 and 1"):
        analyze_dataset(sample_dataset)


def test_missing_required_column_is_rejected(
    sample_dataset: pd.DataFrame,
) -> None:
    sample_dataset = sample_dataset.drop(columns=["loan_amount"])

    with pytest.raises(ValueError, match="loan_amount"):
        analyze_dataset(sample_dataset)
