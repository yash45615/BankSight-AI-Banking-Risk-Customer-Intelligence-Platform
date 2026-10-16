# BankSight AI: Reproducibility Guide

## Purpose

This guide describes how to reproduce the synthetic-data modelling workflow and verify its outputs.

## Environment

- Use a Python version supported by `pyproject.toml`.
- Install dependencies in a dedicated virtual environment.
- Use the same dependency versions when comparing results.
- Record the operating system and Python version when reporting results.

## Workflow

1. Generate the synthetic dataset using `scripts/generate_dataset.py`.
2. Run data validation and exploratory analysis.
3. Train the baseline models with `scripts/train_models.py`.
4. Run hyperparameter tuning and threshold optimization with `scripts/optimize_models.py`.
5. Inspect evaluation reports and saved model artifacts.
6. Start the FastAPI service and test its prediction endpoint.
7. Start the Streamlit dashboard.
8. Generate the prediction-monitoring report after API predictions have been recorded.

## Verification

Run:

```bash
python -m pytest -v
ruff check .
ruff format --check .
python -m compileall -q src scripts
```

Check that the saved model, reports, and charts exist before launching components that depend on them.

## Reproducibility notes

- Synthetic data generation uses a random seed where configured.
- Exact metrics can vary if dependency versions, data-generation settings, or model parameters change.
- Model selection and threshold selection must use training and validation data; final test data should remain reserved for final evaluation.
- The final test report should record the selected model, threshold, sample size, and relevant metrics.

## Scope and limitations

This project is a portfolio demonstration using synthetic data. Results do not establish real-world credit-risk performance, fairness, calibration, regulatory compliance, or suitability for lending decisions.
