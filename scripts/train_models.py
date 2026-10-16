"""Train and evaluate BankSight AI credit-risk models."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from banksight.ml import train_and_evaluate

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "raw" / "credit_risk.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "data" / "reports"
CHART_DIR = REPORT_DIR / "ml_charts"


def save_evaluation_charts(
    model,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save test-set confusion matrix and ROC curve."""
    output_dir.mkdir(parents=True, exist_ok=True)

    actual = predictions["actual_default"]
    predicted = predictions["predicted_default_at_threshold"]

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=[0, 1],
        display_labels=["No default", "Default"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title("Final Test Set — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        actual,
        predictions["predicted_default_probability"],
        ax=ax,
        name="Selected model",
    )
    ax.set_title("Final Test Set — ROC Curve")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=140)
    plt.close(fig)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run scripts/generate_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)
    model, result, predictions = train_and_evaluate(df)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "banksight_credit_risk.joblib"
    report_path = ARTIFACT_DIR / "ml_report.json"
    comparison_path = ARTIFACT_DIR / "validation_model_comparison.csv"
    predictions_path = REPORT_DIR / "ml_test_predictions.csv"

    # Keep the feature schema and threshold alongside the fitted pipeline.
    joblib.dump(
        {
            "model": model,
            "feature_columns": result.feature_columns,
            "threshold": result.threshold,
            "target": "default",
            "selected_model": result.selected_model,
        },
        model_path,
    )

    report_path.write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(result.validation_results).to_csv(
        comparison_path,
        index=False,
    )
    predictions.to_csv(predictions_path, index=False)

    save_evaluation_charts(
        model,
        predictions,
        CHART_DIR,
    )

    test = result.test_result

    print("\nBankSight AI — ML training completed")
    print("=" * 44)
    print(f"Rows: {result.row_count}")
    print(f"Training rows: {result.training_rows}")
    print(f"Validation rows: {result.validation_rows}")
    print(f"Test rows: {result.test_rows}")
    print(f"Selected model: {result.selected_model}")
    print(f"Test ROC-AUC: {test['roc_auc']:.4f}")
    print(f"Test average precision: {test['average_precision']:.4f}")
    print(f"Test precision: {test['precision']:.4f}")
    print(f"Test recall: {test['recall']:.4f}")
    print(f"Test F1: {test['f1']:.4f}")
    print(f"\nSaved model: {model_path}")
    print(f"Evaluation report: {report_path}")
    print(f"Validation comparison: {comparison_path}")
    print(f"Test predictions: {predictions_path}")
    print(f"Charts: {CHART_DIR}")


if __name__ == "__main__":
    main()
