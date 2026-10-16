"""Validation utilities for BankSight AI datasets."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = {
    "customer_id",
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "interest_rate",
    "existing_debt",
    "credit_history_years",
    "late_payments",
    "loan_purpose",
    "default",
}

NUMERIC_RANGES = {
    "age": (18, 100),
    "annual_income": (0, 100_000_000),
    "employment_years": (0, 82),
    "loan_amount": (0, 100_000_000),
    "interest_rate": (0, 100),
    "existing_debt": (0, 100_000_000),
    "credit_history_years": (0, 82),
    "late_payments": (0, 10_000),
}


@dataclass
class ValidationReport:
    """Contains validation results for a dataset."""

    row_count: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Validate a credit-risk dataset without changing its contents."""
    report = ValidationReport(row_count=len(df))

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        report.errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        return report

    if df.empty:
        report.errors.append("Dataset contains no records.")
        return report

    if df["customer_id"].isna().any():
        report.errors.append("customer_id contains missing values.")

    if df["customer_id"].duplicated().any():
        report.errors.append("customer_id contains duplicate values.")

    if df["customer_id"].astype("string").str.strip().eq("").any():
        report.errors.append("customer_id contains blank values.")

    if df["loan_purpose"].isna().any():
        report.errors.append("loan_purpose contains missing values.")

    if df["loan_purpose"].astype("string").str.strip().eq("").any():
        report.errors.append("loan_purpose contains blank values.")

    for column in NUMERIC_RANGES:
        numeric = pd.to_numeric(df[column], errors="coerce")

        if numeric.isna().any():
            report.errors.append(f"{column} contains missing or non-numeric values.")
            continue

        lower, upper = NUMERIC_RANGES[column]
        if ((numeric < lower) | (numeric > upper)).any():
            report.errors.append(f"{column} must be between {lower} and {upper}.")

        if not numeric.map(
            lambda value: float(value).is_integer()
        ).all() and column in {
            "age",
            "employment_years",
            "credit_history_years",
            "late_payments",
        }:
            report.errors.append(f"{column} must contain whole-number values.")

    target = pd.to_numeric(df["default"], errors="coerce")

    if target.isna().any() or not target.isin([0, 1]).all():
        report.errors.append("default must contain only 0 or 1.")

    if df.isna().any().any():
        report.warnings.append(
            "Dataset contains missing values in one or more columns."
        )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        report.warnings.append(f"Dataset contains {duplicate_rows} duplicate rows.")

    return report
