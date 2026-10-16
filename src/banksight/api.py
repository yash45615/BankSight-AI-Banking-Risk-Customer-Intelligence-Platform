"""FastAPI application for BankSight AI."""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from banksight.ml import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv(
        "BANKSIGHT_MODEL_PATH",
        str(ROOT / "models" / "banksight_optimized.joblib"),
    )
)
PREDICTION_LOG_PATH = Path(
    os.getenv(
        "BANKSIGHT_PREDICTION_LOG",
        str(ROOT / "data" / "reports" / "prediction_events.jsonl"),
    )
)

MODEL_VERSION = os.getenv("BANKSIGHT_MODEL_VERSION", "banksight-optimized-v1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("banksight.api")


class PredictionRequest(BaseModel):
    """Validated input features for one credit-risk prediction."""

    model_config = ConfigDict(extra="forbid")

    age: Annotated[int, Field(ge=18, le=100)]
    annual_income: Annotated[float, Field(gt=0, le=100_000_000)]
    employment_years: Annotated[float, Field(ge=0, le=80)]
    loan_amount: Annotated[float, Field(gt=0, le=100_000_000)]
    interest_rate: Annotated[float, Field(ge=0, le=100)]
    existing_debt: Annotated[float, Field(ge=0, le=100_000_000)]
    credit_history_years: Annotated[float, Field(ge=0, le=100)]
    late_payments: Annotated[int, Field(ge=0, le=1000)]
    loan_purpose: Annotated[
        str,
        Field(
            min_length=1,
            max_length=40,
            pattern=r"^[a-zA-Z][a-zA-Z _-]*$",
        ),
    ]


class PredictionResponse(BaseModel):
    """Response returned by the prediction endpoint."""

    request_id: str
    model_version: str
    default_probability: float
    predicted_default: int
    threshold: float
    disclaimer: str


def load_model_bundle() -> dict[str, Any]:
    """Load and validate the saved model artifact."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Run scripts/optimize_models.py first."
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Model artifact has an unexpected format.")

    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match the current API schema.")

    threshold = bundle.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("Model artifact contains an invalid threshold.")

    if not callable(getattr(bundle["model"], "predict_proba", None)):
        raise ValueError("Loaded model does not support predict_proba.")

    return bundle


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the model at startup, but allow health checks without it."""
    try:
        application.state.model_bundle = load_model_bundle()
        logger.info("BankSight model loaded successfully.")
    except (FileNotFoundError, ValueError, OSError) as exc:
        application.state.model_bundle = None
        logger.error("BankSight model unavailable: %s", exc)

    yield


app = FastAPI(
    title="BankSight AI",
    description=(
        "Demonstration credit-risk prediction API using synthetic data. "
        "Not intended for real lending decisions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def write_prediction_event(
    *,
    request_id: str,
    probability: float,
    predicted_default: int,
    threshold: float,
) -> None:
    """Append minimal prediction metadata to a JSON Lines log."""
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "model_version": MODEL_VERSION,
        "default_probability": probability,
        "predicted_default": predicted_default,
        "threshold": threshold,
    }

    try:
        PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")
    except OSError:
        # Do not silently discard monitoring failures.
        logger.exception("Failed to write prediction event.")


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Return service and model readiness."""
    bundle = getattr(request.app.state, "model_bundle", None)

    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "model_version": MODEL_VERSION,
    }


@app.get("/model-info")
def model_info(request: Request) -> dict[str, Any]:
    """Expose non-sensitive model metadata."""
    bundle = getattr(request.app.state, "model_bundle", None)

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model unavailable. Train and save a model first.",
        )

    return {
        "model_version": MODEL_VERSION,
        "selected_model": bundle.get("selected_model", "unknown"),
        "feature_columns": bundle["feature_columns"],
        "threshold": float(bundle["threshold"]),
        "training_data_type": "synthetic",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Estimate default probability for one input record."""
    bundle = getattr(request.app.state, "model_bundle", None)

    if bundle is None:
        try:
            bundle = load_model_bundle()
            request.app.state.model_bundle = bundle
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Model unavailable: %s", exc)
            raise HTTPException(
                status_code=503,
                detail="Model unavailable. Train and save a model first.",
            ) from exc

    row = pd.DataFrame(
        [payload.model_dump()],
        columns=bundle["feature_columns"],
    )

    try:
        probabilities = bundle["model"].predict_proba(row)
        probability = float(probabilities[0][1])
    except (ValueError, TypeError, IndexError) as exc:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed.",
        ) from exc

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        logger.error("Model returned an invalid probability.")
        raise HTTPException(
            status_code=500,
            detail="Model returned an invalid probability.",
        )

    threshold = float(bundle["threshold"])
    predicted_default = int(probability >= threshold)
    request_id = str(uuid.uuid4())

    write_prediction_event(
        request_id=request_id,
        probability=probability,
        predicted_default=predicted_default,
        threshold=threshold,
    )

    return PredictionResponse(
        request_id=request_id,
        model_version=MODEL_VERSION,
        default_probability=round(probability, 6),
        predicted_default=predicted_default,
        threshold=threshold,
        disclaimer=(
            "Synthetic-data demonstration only; not suitable for real credit decisions."
        ),
    )
