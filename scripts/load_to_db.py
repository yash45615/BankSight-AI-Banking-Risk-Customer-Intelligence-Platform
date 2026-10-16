"""Validate and load a CSV dataset into PostgreSQL."""

from __future__ import annotations

import argparse
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from banksight.data_ingestion import load_csv, save_validation_report
from banksight.db import get_engine, initialize_database
from banksight.models import CreditRiskRecord

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "credit_risk.csv"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "validation.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load validated credit-risk data into PostgreSQL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    df = load_csv(args.input)

    from banksight.data_validation import validate_dataset

    report = validate_dataset(df)
    save_validation_report(report, args.report)

    if report.warnings:
        for warning in report.warnings:
            print(f"WARNING: {warning}")

    initialize_database()
    records = df.to_dict(orient="records")

    with Session(get_engine()) as session:
        existing_count = (
            session.scalar(select(func.count()).select_from(CreditRiskRecord)) or 0
        )

        if existing_count:
            raise RuntimeError(
                f"Database already contains {existing_count} records. "
                "Use a fresh database for this initial loader to avoid "
                "accidental duplicate inserts."
            )

        session.add_all([CreditRiskRecord(**record) for record in records])
        session.commit()

        total = session.scalar(select(func.count()).select_from(CreditRiskRecord))

    print(f"Validated {len(df):,} records.")
    print(f"Loaded {total:,} records into credit_risk_records.")
    print(f"Validation report: {args.report}")


if __name__ == "__main__":
    main()
