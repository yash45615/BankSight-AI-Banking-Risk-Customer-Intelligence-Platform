"""Leakage-aware machine-learning workflow for BankSight AI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class ModelEvaluation:
    """Evaluation metrics for one trained model."""

    model_name: str
    roc_auc: float
    average_precision: float
    precision: float
    recall: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int


@dataclass
class TrainingResult:
    """Serializable summary of the training run."""

    row_count: int
    training_rows: int
    validation_rows: int
    test_rows: int
    positive_class_rate: float
    selected_model: str
    validation_results: list[dict[str, Any]]
    test_result: dict[str, Any]
    threshold: float
    feature_columns: list[str]


def validate_training_data(df: pd.DataFrame) -> None:
    """Validate target and model feature columns."""
    required = set(FEATURE_COLUMNS + [TARGET])
    missing = sorted(required.difference(df.columns))

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if df.empty:
        raise ValueError("Cannot train on an empty dataset.")

    if df[TARGET].isna().any():
        raise ValueError("The target contains missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")

    if df[TARGET].nunique() != 2:
        raise ValueError("Training requires both target classes.")

    if df[TARGET].value_counts().min() < 4:
        raise ValueError(
            "At least four records per target class are needed "
            "for stratified training, validation, and test splits."
        )

    for column in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Feature {column} must be numeric.")

        values = df[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Feature {column} contains infinite values.")


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing steps fitted only on the training split."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, Pipeline]:
    """Return candidate models with independent preprocessing."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> ModelEvaluation:
    """Evaluate class predictions and predicted default probabilities."""
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    return ModelEvaluation(
        model_name=model_name,
        roc_auc=float(roc_auc_score(y, probabilities)),
        average_precision=float(average_precision_score(y, probabilities)),
        precision=float(precision_score(y, predictions, zero_division=0)),
        recall=float(recall_score(y, predictions, zero_division=0)),
        f1=float(f1_score(y, predictions, zero_division=0)),
        true_negatives=int(tn),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
    )


def train_and_evaluate(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[Pipeline, TrainingResult, pd.DataFrame]:
    """Train candidates, select by validation ROC-AUC, and test once."""
    validate_training_data(df)

    if not 0 < threshold < 1:
        raise ValueError("threshold must be strictly between 0 and 1.")

    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int).copy()

    # 70% training, 15% validation, 15% untouched final test.
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    validation_results: list[ModelEvaluation] = []
    candidates: dict[str, Pipeline] = {}

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        candidates[model_name] = model
        validation_results.append(
            evaluate_model(
                model_name,
                model,
                x_validation,
                y_validation,
                threshold,
            )
        )

    # Use validation data for model selection, not the final test set.
    best_result = max(validation_results, key=lambda item: item.roc_auc)
    selected_name = best_result.model_name

    # Refit the selected model on training + validation data.
    x_train_validation = pd.concat([x_train, x_validation])
    y_train_validation = pd.concat([y_train, y_validation])
    final_model = build_models()[selected_name]
    final_model.fit(x_train_validation, y_train_validation)

    test_result = evaluate_model(
        selected_name,
        final_model,
        x_test,
        y_test,
        threshold,
    )

    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    predictions = pd.DataFrame(
        {
            "actual_default": y_test.to_numpy(),
            "predicted_default_probability": test_probabilities,
            "predicted_default_at_threshold": (test_probabilities >= threshold).astype(
                int
            ),
        },
        index=x_test.index,
    ).sort_index()

    result = TrainingResult(
        row_count=len(df),
        training_rows=len(x_train),
        validation_rows=len(x_validation),
        test_rows=len(x_test),
        positive_class_rate=float(y.mean()),
        selected_model=selected_name,
        validation_results=[asdict(item) for item in validation_results],
        test_result=asdict(test_result),
        threshold=threshold,
        feature_columns=FEATURE_COLUMNS.copy(),
    )

    return final_model, result, predictions
