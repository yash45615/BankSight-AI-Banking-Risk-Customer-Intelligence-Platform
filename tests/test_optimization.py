"""Tests for BankSight AI threshold and subgroup diagnostics."""

import numpy as np
import pandas as pd
import pytest

from banksight.optimization import choose_threshold, subgroup_diagnostics


def test_threshold_optimizer_returns_valid_threshold() -> None:
    actual = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.20, 0.35, 0.45, 0.75, 0.90])

    result = choose_threshold(actual, probabilities)

    assert 0.10 <= result["threshold"] <= 0.90
    assert result["total_cost"] >= 0
    assert result["cost_per_record"] >= 0


def test_false_negative_cost_changes_threshold_selection() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.30, 0.40, 0.45, 0.60])

    low_fn_cost = choose_threshold(
        actual,
        probabilities,
        false_negative_cost=1,
        false_positive_cost=5,
    )
    high_fn_cost = choose_threshold(
        actual,
        probabilities,
        false_negative_cost=10,
        false_positive_cost=1,
    )

    assert high_fn_cost["threshold"] <= low_fn_cost["threshold"]


def test_threshold_optimizer_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        choose_threshold(
            pd.Series([0, 1]),
            np.array([0.2, 1.2]),
        )


def test_subgroup_diagnostics_returns_age_and_income_groups() -> None:
    df = pd.DataFrame(
        {
            "age": [25, 32, 42, 52, 62] * 8,
            "annual_income": [30000, 45000, 60000, 80000, 100000] * 8,
        }
    )
    actual = pd.Series([0, 1, 0, 1, 0] * 8)
    probabilities = np.array([0.1, 0.8, 0.2, 0.7, 0.3] * 8)

    result = subgroup_diagnostics(
        df,
        actual,
        probabilities,
        threshold=0.5,
        minimum_group_size=5,
    )

    assert "age_group" in result
    assert "income_group" in result
    assert len(result["age_group"]) > 0
    assert len(result["income_group"]) == 4


def test_subgroup_diagnostics_rejects_misaligned_data() -> None:
    df = pd.DataFrame({"age": [25, 35], "annual_income": [30000, 50000]})

    with pytest.raises(ValueError, match="must align"):
        subgroup_diagnostics(
            df,
            pd.Series([0]),
            np.array([0.2]),
            threshold=0.5,
        )
