"""Run BankSight AI exploratory analysis and save reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from banksight.eda import analyze_dataset, result_to_dict, write_markdown_report

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "raw" / "credit_risk.csv"
REPORT_DIR = ROOT / "data" / "reports"
CHART_DIR = REPORT_DIR / "charts"


def save_charts(df: pd.DataFrame) -> None:
    """Generate useful exploratory charts."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Overall target distribution.
    target_counts = (
        df["default"]
        .value_counts()
        .reindex(
            [0, 1],
            fill_value=0,
        )
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["No default", "Default"],
        target_counts.values,
    )
    ax.set_title("Credit Default Class Distribution")
    ax.set_ylabel("Number of records")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "default_distribution.png", dpi=140)
    plt.close(fig)

    # Default rates by loan purpose.
    purpose_rates = (
        df.groupby("loan_purpose", observed=False)["default"]
        .mean()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(purpose_rates.index.astype(str), purpose_rates.values)
    ax.set_title("Default Rate by Loan Purpose")
    ax.set_ylabel("Default rate")
    ax.set_ylim(0, min(1, max(0.1, float(purpose_rates.max()) * 1.2)))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "default_rate_by_purpose.png", dpi=140)
    plt.close(fig)

    # Numeric correlations.
    numeric = df.select_dtypes(include="number")
    correlation = numeric.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title("Numeric Feature Correlation Matrix")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "correlation_matrix.png", dpi=140)
    plt.close(fig)

    # Late-payment distribution by default status.
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [
        df.loc[df["default"] == 0, "late_payments"].dropna(),
        df.loc[df["default"] == 1, "late_payments"].dropna(),
    ]
    ax.boxplot(
        groups,
        tick_labels=["No default", "Default"],
    )
    ax.set_title("Late Payments by Default Status")
    ax.set_ylabel("Number of late payments")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "late_payments_by_default.png", dpi=140)
    plt.close(fig)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Run scripts/generate_dataset.py first."
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET_PATH)

    result = analyze_dataset(df)

    json_path = REPORT_DIR / "eda_report.json"
    markdown_path = REPORT_DIR / "eda_report.md"

    json_path.write_text(
        json.dumps(result_to_dict(result), indent=2),
        encoding="utf-8",
    )
    write_markdown_report(result, markdown_path)
    save_charts(df)

    print("BankSight AI EDA completed successfully.")
    print(f"Rows analyzed: {result.row_count}")
    print(f"Overall default rate: {result.overall_default_rate:.2%}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Charts: {CHART_DIR}")


if __name__ == "__main__":
    main()
