"""Exploratory data analysis and statistical tests for BankSight AI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

TARGET = "default"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "interest_rate",
    "existing_debt",
    "credit_history_years",
    "late_payments",
]

CATEGORICAL_FEATURES = ["loan_purpose"]


@dataclass
class AnalysisResult:
    """Structured output from a BankSight EDA run."""

    row_count: int
    column_count: int
    missing_values: dict[str, int]
    duplicate_rows: int
    overall_default_rate: float
    default_rate_ci_95: list[float]
    numeric_summary: dict[str, dict[str, float]]
    default_rates_by_purpose: list[dict[str, Any]]
    default_rates_by_late_payments: list[dict[str, Any]]
    statistical_tests: dict[str, Any]


def _require_valid_dataset(df: pd.DataFrame) -> None:
    """Reject missing required columns and invalid target values."""
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset contains no rows.")

    if df[TARGET].isna().any():
        raise ValueError("The default target contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The default target must contain only 0 and 1.")

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Expected numeric values in column: {column}")

        if not np.isfinite(df[column].dropna()).all():
            raise ValueError(f"Column {column} contains infinite values.")


def _proportion_confidence_interval(
    successes: int,
    total: int,
    z_score: float = 1.96,
) -> list[float]:
    """Calculate a 95% Wilson confidence interval for a proportion."""
    if total == 0:
        return [0.0, 0.0]

    proportion = successes / total
    denominator = 1 + z_score**2 / total
    center = (proportion + z_score**2 / (2 * total)) / denominator
    margin = (
        z_score
        * np.sqrt(proportion * (1 - proportion) / total + z_score**2 / (4 * total**2))
        / denominator
    )

    return [
        round(float(max(0, center - margin)), 6),
        round(float(min(1, center + margin)), 6),
    ]


def _group_default_rates(
    df: pd.DataFrame,
    group_column: str,
) -> list[dict[str, Any]]:
    """Calculate count and default rate for each group."""
    grouped = (
        df.groupby(group_column, dropna=False)[TARGET]
        .agg(["count", "mean"])
        .reset_index()
    )

    results: list[dict[str, Any]] = []

    for _, row in grouped.iterrows():
        group_value = row[group_column]
        results.append(
            {
                "group": (str(group_value) if pd.notna(group_value) else "Missing"),
                "count": int(row["count"]),
                "default_rate": round(float(row["mean"]), 6),
            }
        )

    return results


def _numeric_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return selected descriptive statistics for numeric features."""
    summary: dict[str, dict[str, float]] = {}

    for column in NUMERIC_FEATURES:
        values = df[column].dropna()

        if values.empty:
            summary[column] = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            continue

        summary[column] = {
            "mean": round(float(values.mean()), 4),
            "median": round(float(values.median()), 4),
            "std": round(float(values.std(ddof=1) or 0), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
        }

    return summary


def _statistical_tests(df: pd.DataFrame) -> dict[str, Any]:
    """Test associations between default and selected features."""
    results: dict[str, Any] = {}

    # H0: loan purpose and default status are independent.
    purpose_table = pd.crosstab(df["loan_purpose"], df[TARGET])

    if purpose_table.shape[0] >= 2 and purpose_table.shape[1] >= 2:
        chi2, p_value, degrees_freedom, _ = chi2_contingency(purpose_table)
        results["loan_purpose_vs_default"] = {
            "test": "chi-square test of independence",
            "null_hypothesis": ("Loan purpose and default status are independent."),
            "statistic": round(float(chi2), 6),
            "degrees_of_freedom": int(degrees_freedom),
            "p_value": round(float(p_value), 8),
            "significant_at_0_05": bool(p_value < 0.05),
        }
    else:
        results["loan_purpose_vs_default"] = {
            "test": "chi-square test of independence",
            "status": "Not enough variation in the data.",
        }

    # H0: the late-payment distributions are the same in both groups.
    non_default = df.loc[df[TARGET] == 0, "late_payments"].dropna()
    default = df.loc[df[TARGET] == 1, "late_payments"].dropna()

    if len(non_default) > 0 and len(default) > 0:
        test = mannwhitneyu(
            non_default,
            default,
            alternative="two-sided",
        )
        results["late_payments_by_default"] = {
            "test": "Mann-Whitney U test",
            "null_hypothesis": (
                "Late-payment distributions are the same for "
                "default and non-default records."
            ),
            "non_default_median": round(float(non_default.median()), 4),
            "default_median": round(float(default.median()), 4),
            "statistic": round(float(test.statistic), 6),
            "p_value": round(float(test.pvalue), 8),
            "significant_at_0_05": bool(test.pvalue < 0.05),
        }
    else:
        results["late_payments_by_default"] = {
            "test": "Mann-Whitney U test",
            "status": "Both target groups need observations.",
        }

    return results


def analyze_dataset(df: pd.DataFrame) -> AnalysisResult:
    """Run data profiling, grouped analysis, and statistical tests."""
    _require_valid_dataset(df)

    default_count = int(df[TARGET].sum())
    row_count = len(df)

    return AnalysisResult(
        row_count=row_count,
        column_count=len(df.columns),
        missing_values={
            str(column): int(count) for column, count in df.isna().sum().items()
        },
        duplicate_rows=int(df.duplicated().sum()),
        overall_default_rate=round(default_count / row_count, 6),
        default_rate_ci_95=_proportion_confidence_interval(
            default_count,
            row_count,
        ),
        numeric_summary=_numeric_summary(df),
        default_rates_by_purpose=_group_default_rates(
            df,
            "loan_purpose",
        ),
        default_rates_by_late_payments=_group_default_rates(
            df,
            "late_payments",
        ),
        statistical_tests=_statistical_tests(df),
    )


def write_markdown_report(
    result: AnalysisResult,
    output_path: Path,
) -> None:
    """Write a readable summary report."""
    purpose_lines = [
        "| Loan purpose | Records | Default rate |",
        "|---|---:|---:|",
    ]
    for item in result.default_rates_by_purpose:
        purpose_lines.append(
            f"| {item['group']} | {item['count']} | {item['default_rate']:.2%} |"
        )

    numeric_lines = [
        "| Feature | Mean | Median | Std. dev. | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, values in result.numeric_summary.items():
        numeric_lines.append(
            f"| {feature} | {values['mean']:.2f} | "
            f"{values['median']:.2f} | {values['std']:.2f} | "
            f"{values['min']:.2f} | {values['max']:.2f} |"
        )

    test_lines: list[str] = []
    for name, test in result.statistical_tests.items():
        test_lines.append(f"### {name.replace('_', ' ').title()}")
        test_lines.append(f"- Test: {test['test']}")

        if "p_value" in test:
            test_lines.append(f"- p-value: {test['p_value']}")
            test_lines.append(f"- Significant at 0.05: {test['significant_at_0_05']}")
        else:
            test_lines.append(f"- Status: {test.get('status', 'Not available')}")

        test_lines.append("")

    content = (
        f"""# BankSight AI — Exploratory Data Analysis

## Dataset overview

- Records: {result.row_count}
- Columns: {result.column_count}
- Duplicate rows: {result.duplicate_rows}
- Overall default rate: {result.overall_default_rate:.2%}
- Approximate 95% Wilson confidence interval for default rate:
  {result.default_rate_ci_95[0]:.2%}–{result.default_rate_ci_95[1]:.2%}

## Missing values

| Column | Missing values |
|---|---:|
"""
        + "\n".join(
            f"| {column} | {count} |" for column, count in result.missing_values.items()
        )
        + """

## Numeric feature summary

"""
        + "\n".join(numeric_lines)
        + """

## Default rate by loan purpose

"""
        + "\n".join(purpose_lines)
        + """

## Statistical tests

"""
        + "\n".join(test_lines)
        + """
## Interpretation notes

- A p-value below 0.05 is evidence against the relevant null hypothesis;
  it does not prove causation or practical importance.
- Group rates should be interpreted alongside group sizes.
- The dataset is synthetic. Results are for demonstrating an analytical
  workflow and must not be presented as real banking findings.
"""
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def result_to_dict(result: AnalysisResult) -> dict[str, Any]:
    """Convert the result into a JSON-serializable dictionary."""
    return asdict(result)
