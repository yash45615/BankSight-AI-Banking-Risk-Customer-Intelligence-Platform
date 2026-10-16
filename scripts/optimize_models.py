"""Tune BankSight AI models and produce validation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from banksight.ml import FEATURE_COLUMNS, TARGET, build_models, validate_training_data
from banksight.optimization import choose_threshold, subgroup_diagnostics

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "raw" / "credit_risk.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "data" / "reports"

FALSE_NEGATIVE_COST = 5.0
FALSE_POSITIVE_COST = 1.0
RANDOM_STATE = 42


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run scripts/generate_dataset.py first."
        )

    df = pd.read_csv(DATASET_PATH)
    validate_training_data(df)

    x = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].astype(int).copy()

    # Split into 70% training, 15% validation, and 15% final test.
    x_train, x_temporary, y_train, y_temporary = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temporary,
        y_temporary,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temporary,
    )

    search_spaces = {
        "logistic_regression": {
            "classifier__C": [0.1, 1.0, 10.0],
        },
        "random_forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 8, 15],
            "classifier__min_samples_leaf": [2, 5],
        },
    }

    fitted_searches: dict[str, GridSearchCV] = {}
    validation_scores: dict[str, float] = {}
    search_summary: list[dict[str, object]] = []

    # Cross-validation sees training data only.
    for name, pipeline in build_models().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=search_spaces[name],
            scoring="roc_auc",
            cv=3,
            n_jobs=1,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)

        fitted_searches[name] = search
        validation_scores[name] = float(
            roc_auc_score(
                y_validation,
                search.best_estimator_.predict_proba(x_validation)[:, 1],
            )
        )
        search_summary.append(
            {
                "model": name,
                "best_cv_roc_auc": float(search.best_score_),
                "validation_roc_auc": validation_scores[name],
                "best_parameters": search.best_params_,
            }
        )

    # Select using validation ROC-AUC, never the test set.
    selected_name = max(validation_scores, key=validation_scores.get)
    selected_model = fitted_searches[selected_name].best_estimator_

    validation_probabilities = selected_model.predict_proba(x_validation)[:, 1]

    threshold_result = choose_threshold(
        y_validation,
        validation_probabilities,
        false_negative_cost=FALSE_NEGATIVE_COST,
        false_positive_cost=FALSE_POSITIVE_COST,
    )
    threshold = threshold_result["threshold"]

    # Final evaluation on the untouched test set.
    test_probabilities = selected_model.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "average_precision": float(average_precision_score(y_test, test_probabilities)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "threshold": float(threshold),
        "false_negative_cost": FALSE_NEGATIVE_COST,
        "false_positive_cost": FALSE_POSITIVE_COST,
    }

    # Explain model behavior on validation data, not the final test set.
    importance = permutation_importance(
        selected_model,
        x_validation,
        y_validation,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    # Subgroup diagnostics use the held-out test data.
    subgroup_report = subgroup_diagnostics(
        df.loc[x_test.index],
        y_test,
        test_probabilities,
        threshold,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": selected_model,
            "feature_columns": FEATURE_COLUMNS,
            "threshold": threshold,
            "selected_model": selected_name,
        },
        MODEL_DIR / "banksight_optimized.joblib",
    )

    importance_df.to_csv(
        OUTPUT_DIR / "permutation_importance.csv",
        index=False,
    )

    comparison_df = pd.DataFrame(
        [
            {
                "model": item["model"],
                "best_cv_roc_auc": item["best_cv_roc_auc"],
                "validation_roc_auc": item["validation_roc_auc"],
            }
            for item in search_summary
        ]
    )
    comparison_df.to_csv(
        OUTPUT_DIR / "tuning_comparison.csv",
        index=False,
    )

    test_predictions = pd.DataFrame(
        {
            "actual_default": y_test,
            "predicted_probability": test_probabilities,
            "predicted_default": test_predictions,
        },
        index=x_test.index,
    ).sort_index()
    test_predictions.to_csv(
        REPORT_DIR / "optimized_test_predictions.csv",
        index_label="source_row",
    )

    report = {
        "dataset_rows": len(df),
        "split_sizes": {
            "training": len(x_train),
            "validation": len(x_validation),
            "test": len(x_test),
        },
        "selection_metric": "validation ROC-AUC",
        "selected_model": selected_name,
        "cross_validation_and_validation": search_summary,
        "best_parameters": fitted_searches[selected_name].best_params_,
        "threshold_selection": threshold_result,
        "final_test_metrics": test_metrics,
        "subgroup_diagnostics": subgroup_report,
        "limitations": [
            "The dataset is synthetic and does not establish real-world performance.",
            "The error-cost ratio is an illustrative assumption, not a banking policy.",
            "Subgroup metrics are descriptive and are not proof of fairness.",
            "Permutation importance indicates predictive reliance, not causation.",
        ],
    }

    (OUTPUT_DIR / "optimization_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nBankSight AI — Optimization completed")
    print("=" * 44)
    print(f"Selected model: {selected_name}")
    print(f"Chosen threshold: {threshold:.2f}")
    print(
        f"Validation error cost per record: {threshold_result['cost_per_record']:.4f}"
    )
    print(f"Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"Test average precision: {test_metrics['average_precision']:.4f}")
    print(f"Test precision: {test_metrics['precision']:.4f}")
    print(f"Test recall: {test_metrics['recall']:.4f}")
    print(f"Saved model: {MODEL_DIR / 'banksight_optimized.joblib'}")
    print(f"Report: {OUTPUT_DIR / 'optimization_report.json'}")
    print(f"Feature importance: {OUTPUT_DIR / 'permutation_importance.csv'}")
    print(f"Model comparison: {OUTPUT_DIR / 'tuning_comparison.csv'}")


if __name__ == "__main__":
    main()
