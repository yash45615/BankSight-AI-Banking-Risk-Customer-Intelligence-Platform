"""Threshold optimization and subgroup diagnostics for BankSight AI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def choose_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> dict[str, float]:
    """Choose the threshold with the lowest validation error cost."""
    if false_negative_cost <= 0 or false_positive_cost <= 0:
        raise ValueError("Both error costs must be positive.")

    if len(y_true) != len(probabilities) or len(y_true) == 0:
        raise ValueError("Labels and probabilities must have equal nonzero length.")

    if not set(pd.Series(y_true).unique()).issubset({0, 1}):
        raise ValueError("Labels must contain only 0 and 1.")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be between 0 and 1.")

    best: dict[str, float] | None = None

    # Fixed grid makes the selection process deterministic and reproducible.
    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        total_cost = float(fn) * false_negative_cost + float(fp) * false_positive_cost

        candidate = {
            "threshold": round(float(threshold), 2),
            "total_cost": float(total_cost),
            "cost_per_record": float(total_cost / len(y_true)),
            "precision": float(tp / (tp + fp) if tp + fp else 0.0),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "true_negatives": float(tn),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_positives": float(tp),
        }

        if best is None or candidate["total_cost"] < best["total_cost"]:
            best = candidate

    if best is None:
        raise RuntimeError("No candidate threshold was evaluated.")

    return best


def subgroup_diagnostics(
    df: pd.DataFrame,
    actual: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    minimum_group_size: int = 20,
) -> dict[str, Any]:
    """Report classification metrics by descriptive demographic groups."""
    if len(df) != len(actual) or len(actual) != len(probabilities):
        raise ValueError("Group data, labels, and probabilities must align.")

    if minimum_group_size < 1:
        raise ValueError("minimum_group_size must be at least 1.")

    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    predictions = (probabilities >= threshold).astype(int)
    working = pd.DataFrame(
        {
            "age": df["age"].to_numpy(),
            "annual_income": df["annual_income"].to_numpy(),
            "actual": np.asarray(actual),
            "predicted": predictions,
        }
    )

    working["age_group"] = pd.cut(
        working["age"],
        bins=[-np.inf, 29, 39, 49, 59, np.inf],
        labels=["29_and_under", "30_to_39", "40_to_49", "50_to_59", "60_plus"],
    )

    working["income_group"] = pd.qcut(
        working["annual_income"].rank(method="first"),
        q=4,
        labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"],
    )

    output: dict[str, Any] = {}

    for column in ["age_group", "income_group"]:
        groups: list[dict[str, Any]] = []

        for group_name, group in working.groupby(
            column,
            observed=True,
            dropna=False,
        ):
            count = len(group)
            actual_values = group["actual"].to_numpy()
            predicted_values = group["predicted"].to_numpy()

            tn, fp, fn, tp = confusion_matrix(
                actual_values,
                predicted_values,
                labels=[0, 1],
            ).ravel()

            if count < minimum_group_size:
                groups.append(
                    {
                        "group": str(group_name),
                        "count": count,
                        "status": "Insufficient sample size for interpretation",
                    }
                )
                continue

            positive_actuals = int(actual_values.sum())

            groups.append(
                {
                    "group": str(group_name),
                    "count": count,
                    "actual_default_rate": float(actual_values.mean()),
                    "predicted_default_rate": float(predicted_values.mean()),
                    "recall": float(tp / (tp + fn) if tp + fn else 0.0),
                    "false_positive_rate": float(fp / (fp + tn) if fp + tn else 0.0),
                    "actual_defaults": positive_actuals,
                    "status": "Descriptive diagnostic only",
                }
            )

        output[column] = groups

    return output
