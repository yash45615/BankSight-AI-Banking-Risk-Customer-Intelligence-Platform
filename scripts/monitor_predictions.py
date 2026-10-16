"""Summarize prediction events logged by BankSight AI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "data" / "reports" / "prediction_events.jsonl"
REPORT_PATH = ROOT / "data" / "reports" / "prediction_monitoring.json"


def main() -> None:
    if not LOG_PATH.exists():
        raise FileNotFoundError(
            f"No prediction log found at {LOG_PATH}. "
            "Start the API and submit at least one valid prediction."
        )

    events = []
    with LOG_PATH.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in prediction log at line {line_number}."
                ) from exc

            events.append(event)

    if not events:
        raise ValueError("Prediction log contains no events.")

    df = pd.DataFrame(events)

    required = {
        "timestamp",
        "request_id",
        "model_version",
        "default_probability",
        "predicted_default",
        "threshold",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Prediction log is missing fields: {missing}")

    probabilities = pd.to_numeric(
        df["default_probability"],
        errors="coerce",
    )
    predictions = pd.to_numeric(
        df["predicted_default"],
        errors="coerce",
    )

    if (
        probabilities.isna().any()
        or not np.isfinite(probabilities.to_numpy()).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("Prediction log contains invalid probabilities.")

    if predictions.isna().any() or not predictions.isin([0, 1]).all():
        raise ValueError("Prediction log contains invalid predicted labels.")

    report = {
        "event_count": int(len(df)),
        "unique_model_versions": sorted(
            df["model_version"].astype(str).unique().tolist()
        ),
        "first_event_timestamp": str(df["timestamp"].iloc[0]),
        "last_event_timestamp": str(df["timestamp"].iloc[-1]),
        "mean_predicted_default_probability": float(probabilities.mean()),
        "median_predicted_default_probability": float(probabilities.median()),
        "predicted_default_rate": float(predictions.mean()),
        "probability_quantiles": {
            "p10": float(probabilities.quantile(0.10)),
            "p50": float(probabilities.quantile(0.50)),
            "p90": float(probabilities.quantile(0.90)),
        },
        "limitations": [
            "These are model predictions, not observed default outcomes.",
            "Prediction distributions alone do not establish data drift.",
            "Monitoring requires sufficient events and context for interpretation.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("Prediction monitoring report generated.")
    print(f"Events analyzed: {report['event_count']}")
    print(
        "Mean predicted default probability: "
        f"{report['mean_predicted_default_probability']:.4f}"
    )
    print(f"Predicted default rate: {report['predicted_default_rate']:.2%}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
