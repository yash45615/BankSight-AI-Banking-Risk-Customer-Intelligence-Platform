"""Generate a reproducible synthetic credit-risk dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "credit_risk.csv"


def generate_dataset(
    rows: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic customer and loan attributes."""
    if rows < 100:
        raise ValueError("rows must be at least 100.")

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    rng = np.random.default_rng(seed)

    age = rng.integers(21, 71, size=rows)
    annual_income = np.clip(
        rng.lognormal(mean=10.8, sigma=0.45, size=rows),
        12000,
        500000,
    ).round(2)
    employment_years = np.minimum(
        rng.integers(0, 36, size=rows),
        age - 18,
    )
    loan_amount = np.clip(
        rng.lognormal(mean=9.5, sigma=0.65, size=rows),
        1000,
        150000,
    ).round(2)
    interest_rate = rng.uniform(4.0, 29.0, size=rows).round(2)
    existing_debt = np.clip(
        rng.lognormal(mean=8.7, sigma=0.8, size=rows),
        0,
        200000,
    ).round(2)
    credit_history_years = np.minimum(
        rng.integers(0, 31, size=rows),
        age - 18,
    )
    late_payments = rng.poisson(lam=1.0, size=rows)
    loan_purpose = rng.choice(
        ["home", "vehicle", "personal", "education", "business"],
        size=rows,
        p=[0.24, 0.20, 0.28, 0.12, 0.16],
    )

    debt_to_income = (existing_debt + loan_amount * 0.15) / np.maximum(annual_income, 1)

    # These coefficients create a plausible synthetic pattern.
    # They are not a real banking risk model.
    log_odds = (
        -2.0
        + 2.2 * debt_to_income
        + 0.32 * late_payments
        + 0.055 * (interest_rate - 12)
        - 0.035 * credit_history_years
        - 0.012 * employment_years
    )

    probability = 1 / (1 + np.exp(-np.clip(log_odds, -20, 20)))
    default = rng.binomial(1, probability)

    customer_ids = [f"CUST-{i:06d}" for i in range(1, rows + 1)]

    return pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "interest_rate": interest_rate,
            "existing_debt": existing_debt,
            "credit_history_years": credit_history_years,
            "late_payments": late_payments,
            "loan_purpose": loan_purpose,
            "default": default.astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic credit-risk data.")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    dataset = generate_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"Created {len(dataset):,} synthetic records.")
    print(f"Output: {args.output}")
    print(f"Default rate: {dataset['default'].mean():.2%}")


if __name__ == "__main__":
    main()
