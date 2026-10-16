"""CSV ingestion functions for BankSight AI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from banksight.data_validation import ValidationReport, validate_dataset


class DataValidationError(ValueError):
    """Raised when an input dataset fails validation."""


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file and validate its schema and values."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported.")

    df = pd.read_csv(path)

    report = validate_dataset(df)
    if not report.is_valid:
        details = "; ".join(report.errors)
        raise DataValidationError(f"Dataset validation failed: {details}")

    return df


def save_validation_report(
    report: ValidationReport,
    path: Path,
) -> None:
    """Save a machine-readable validation report."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "row_count": report.row_count,
        "is_valid": report.is_valid,
        "errors": report.errors,
        "warnings": report.warnings,
    }

    import json

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
