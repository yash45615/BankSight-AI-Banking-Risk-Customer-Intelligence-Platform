"""Interactive Streamlit dashboard for BankSight AI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
API_BASE_URL = os.getenv("BANKSIGHT_API_URL", "http://127.0.0.1:8000").rstrip("/")

OPTIMIZATION_REPORT = ROOT / "artifacts" / "optimization_report.json"
ML_REPORT = ROOT / "artifacts" / "ml_report.json"
EDA_REPORT = ROOT / "data" / "reports" / "eda_report.json"
MONITORING_REPORT = ROOT / "data" / "reports" / "prediction_monitoring.json"
IMPORTANCE_FILE = ROOT / "artifacts" / "permutation_importance.csv"
TUNING_FILE = ROOT / "artifacts" / "tuning_comparison.csv"
SUBGROUP_FILE = ROOT / "artifacts" / "subgroup_diagnostics.csv"


st.set_page_config(
    page_title="BankSight AI",
    page_icon="ðŸ“Š",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON report without crashing the dashboard."""
    try:
        if path.is_file():
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
            return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def load_csv(path: Path) -> pd.DataFrame | None:
    """Read a CSV report when available."""
    try:
        if path.is_file():
            return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    return None


def get_api_json(endpoint: str) -> tuple[dict[str, Any] | None, str | None]:
    """Request a JSON response from the FastAPI service."""
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            return None, "The API returned an unexpected response."
        return result, None
    except requests.RequestException as exc:
        return None, (
            f"Cannot reach the API at {API_BASE_URL}. "
            f"Start the FastAPI server first. Details: {exc}"
        )
    except ValueError:
        return None, "The API returned invalid JSON."


def metric_value(report: dict[str, Any] | None, *keys: str) -> Any:
    """Get a report field while tolerating missing report files."""
    if not report:
        return None

    for key in keys:
        if key in report:
            return report[key]

    return None


def display_metric(
    label: str,
    value: Any,
    *,
    percentage: bool = False,
) -> None:
    """Display a metric if its value is available."""
    if value is None:
        st.metric(label, "N/A")
        return

    try:
        number = float(value)
        rendered = f"{number:.1%}" if percentage else f"{number:.3f}"
        st.metric(label, rendered)
    except (TypeError, ValueError):
        st.metric(label, str(value))


def render_prediction_page() -> None:
    st.subheader("Applicant risk prediction")
    st.write(
        "Enter synthetic demonstration data to obtain a probability "
        "from the trained model."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35,
                step=1,
            )
            annual_income = st.number_input(
                "Annual income",
                min_value=1.0,
                max_value=100_000_000.0,
                value=75_000.0,
                step=1_000.0,
            )
            employment_years = st.number_input(
                "Years employed",
                min_value=0.0,
                max_value=80.0,
                value=8.0,
                step=1.0,
            )
            loan_amount = st.number_input(
                "Loan amount",
                min_value=1.0,
                max_value=100_000_000.0,
                value=15_000.0,
                step=500.0,
            )
            interest_rate = st.number_input(
                "Interest rate (%)",
                min_value=0.0,
                max_value=100.0,
                value=10.5,
                step=0.5,
            )

        with right:
            existing_debt = st.number_input(
                "Existing debt",
                min_value=0.0,
                max_value=100_000_000.0,
                value=5_000.0,
                step=500.0,
            )
            credit_history_years = st.number_input(
                "Credit history (years)",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
            )
            late_payments = st.number_input(
                "Number of late payments",
                min_value=0,
                max_value=1000,
                value=1,
                step=1,
            )
            loan_purpose = st.selectbox(
                "Loan purpose",
                options=[
                    "education",
                    "home",
                    "medical",
                    "personal",
                    "business",
                    "vehicle",
                    "debt_consolidation",
                ],
            )

        submitted = st.form_submit_button(
            "Calculate demonstration prediction",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    payload = {
        "age": int(age),
        "annual_income": float(annual_income),
        "employment_years": float(employment_years),
        "loan_amount": float(loan_amount),
        "interest_rate": float(interest_rate),
        "existing_debt": float(existing_debt),
        "credit_history_years": float(credit_history_years),
        "late_payments": int(late_payments),
        "loan_purpose": loan_purpose,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()

    except requests.HTTPError:
        try:
            detail = response.json().get("detail", response.text)
        except (ValueError, AttributeError):
            detail = response.text
        st.error(f"Prediction API error: {detail}")
        return

    except requests.Timeout:
        st.error("The prediction request timed out. Try again.")
        return

    except requests.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. Start the API in another terminal and retry."
        )
        return

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return

    except ValueError:
        st.error("The API returned invalid JSON.")
        return

    probability = result.get("default_probability")
    threshold = result.get("threshold")

    if not isinstance(probability, (int, float)):
        st.error("The API response did not contain a valid probability.")
        return

    if not isinstance(threshold, (int, float)):
        st.error("The API response did not contain a valid threshold.")
        return

    st.divider()
    st.subheader("Prediction result")

    first, second, third = st.columns(3)
    first.metric("Estimated default probability", f"{probability:.1%}")
    second.metric("Model threshold", f"{threshold:.1%}")
    second_class = (
        "Above threshold" if result.get("predicted_default") == 1 else "Below threshold"
    )
    third.metric("Threshold comparison", second_class)

    st.caption(f"Request ID: {result.get('request_id', 'N/A')}")
    st.caption(f"Model version: {result.get('model_version', 'N/A')}")
    st.warning(
        result.get(
            "disclaimer",
            "Synthetic portfolio demonstration only; not a real credit decision.",
        )
    )


def render_overview_page() -> None:
    st.subheader("Model overview")

    health, health_error = get_api_json("/health")
    model_info, info_error = get_api_json("/model-info")

    left, right = st.columns(2)

    with left:
        if health:
            if health.get("model_loaded"):
                st.success("API is responding and a model is loaded.")
            else:
                st.warning("API is responding, but no model is loaded.")
            st.json(health)
        else:
            st.error(health_error or "Health check unavailable.")

    with right:
        if model_info:
            st.write("Loaded model information")
            st.json(model_info)
        elif info_error:
            st.info(info_error)

    report = load_json(OPTIMIZATION_REPORT)
    if report is None:
        report = load_json(ML_REPORT)

    st.divider()
    st.subheader("Held-out model evaluation")

    if report is None:
        st.info(
            "No model report found. Run the model training or optimization "
            "script first."
        )
        return

    # Accommodate common report layouts without assuming a fixed schema.
    metrics = report.get("test_metrics", report.get("metrics", report))

    if not isinstance(metrics, dict):
        st.json(report)
        return

    cols = st.columns(4)
    metric_definitions = [
        ("ROC-AUC", ("roc_auc", "test_roc_auc"), False),
        ("Average precision", ("average_precision", "test_average_precision"), False),
        ("Precision", ("precision", "test_precision"), False),
        ("Recall", ("recall", "test_recall"), False),
    ]

    for column, (label, keys, percentage) in zip(cols, metric_definitions, strict=True):
        with column:
            display_metric(label, metric_value(metrics, *keys), percentage=percentage)

    with st.expander("View full evaluation report"):
        st.json(report)


def render_diagnostics_page() -> None:
    st.subheader("Model diagnostics")

    importance = load_csv(IMPORTANCE_FILE)
    tuning = load_csv(TUNING_FILE)
    subgroup = load_csv(SUBGROUP_FILE)
    eda = load_json(EDA_REPORT)

    if importance is not None and not importance.empty:
        st.markdown("#### Permutation feature importance")
        st.caption(
            "This shows how model evaluation changes when individual input "
            "features are shuffled. It is a diagnostic, not proof of causation."
        )

        if "feature" in importance.columns and "importance_mean" in importance.columns:
            chart_data = importance.set_index("feature")[["importance_mean"]]
            st.bar_chart(chart_data.sort_values("importance_mean"))
        else:
            st.dataframe(importance, use_container_width=True)
    else:
        st.info(
            "Feature-importance report not found. Run "
            "`python scripts\\optimize_models.py` first."
        )

    if tuning is not None and not tuning.empty:
        with st.expander("Model tuning comparison"):
            st.dataframe(tuning, use_container_width=True)

    if subgroup is not None and not subgroup.empty:
        st.markdown("#### Exploratory subgroup diagnostics")
        st.caption(
            "Small or synthetic samples can produce unstable results. "
            "These diagnostics do not establish fairness or suitability "
            "for real lending."
        )
        st.dataframe(subgroup, use_container_width=True)

    if eda is not None:
        with st.expander("Exploratory data analysis report"):
            st.json(eda)


def render_monitoring_page() -> None:
    st.subheader("Prediction monitoring")

    report = load_json(MONITORING_REPORT)

    if report is None:
        st.info(
            "No monitoring report is available yet. Make one or more "
            "predictions through the API, then run "
            "`python scripts\\monitor_predictions.py`."
        )
        return

    a, b, c = st.columns(3)
    a.metric("Recorded events", report.get("event_count", "N/A"))
    b.metric(
        "Model versions",
        len(report.get("model_versions", []))
        if isinstance(report.get("model_versions"), list)
        else "N/A",
    )
    rate = report.get("predicted_default_rate")
    c.metric(
        "Predicted default rate",
        f"{rate:.1%}" if isinstance(rate, (int, float)) else "N/A",
    )

    st.markdown("#### Monitoring report")
    st.json(report)

    st.caption(
        "These are prediction-log summaries, not proof of data drift or "
        "model degradation. Measuring real-world drift requires suitable "
        "reference data and, for outcome monitoring, verified labels."
    )


def main() -> None:
    st.title("ðŸ“Š BankSight AI")
    st.write("Credit-risk modelling, model diagnostics, and prediction monitoring")

    st.info(
        "Portfolio prototype using synthetic data. Not financial advice and "
        "not for real-world lending or credit decisions."
    )

    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a page",
            [
                "Risk prediction",
                "Model overview",
                "Diagnostics",
                "Monitoring",
            ],
        )
        st.divider()
        st.caption("API endpoint")
        st.code(API_BASE_URL, language="text")

    if page == "Risk prediction":
        render_prediction_page()
    elif page == "Model overview":
        render_overview_page()
    elif page == "Diagnostics":
        render_diagnostics_page()
    else:
        render_monitoring_page()


if __name__ == "__main__":
    main()
