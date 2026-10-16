# BankSight AI — Credit Risk Analytics & ML Monitoring

**An end-to-end machine learning platform for synthetic credit-risk analysis, model evaluation, API-based predictions, and prediction monitoring.**

BankSight AI demonstrates how a machine learning workflow can be organized into a maintainable Python application, from synthetic data generation and validation to model optimization, REST API serving, and monitoring reports.

> **Disclaimer:** This is an educational portfolio project built around synthetic data. It is not intended for real lending decisions, financial advice, or production credit underwriting.

## Project Highlights

- **Data pipeline:** Generate and validate synthetic credit-risk datasets.
- **Exploratory data analysis:** Produce statistical summaries, reports, and visualizations.
- **Machine learning:** Train and compare baseline classification models.
- **Model optimization:** Tune model parameters, examine classification thresholds, and calculate permutation feature importance.
- **Model diagnostics:** Evaluate test-set performance and inspect prediction behavior across available subgroups.
- **REST API:** Serve model predictions through FastAPI.
- **Prediction monitoring:** Record prediction-event metadata and generate monitoring reports.
- **Interactive dashboard:** Explore predictions, model information, diagnostics, and monitoring through Streamlit.
- **Automated testing and code quality:** Use pytest and Ruff to support reliability and maintainability.

## Architecture

```text
Synthetic Data
      |
      v
Data Validation
      |
      v
Exploratory Data Analysis
      |
      v
Baseline Model Training
      |
      v
Hyperparameter Tuning
      |
      v
Model Evaluation & Diagnostics
      |
      v
Saved Model Artifact
      |
      v
FastAPI Prediction Service
      |
      +----------------------+
      |                      |
      v                      v
Prediction Event Log    Streamlit Dashboard
      |
      v
Monitoring Reports
```

## Technology Stack

| Area | Technologies |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| API development | FastAPI, Pydantic, Uvicorn |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Testing | pytest |
| Code quality | Ruff |
| Model persistence | joblib |
| Database integration | SQLAlchemy, PostgreSQL (optional) |
| Version control | Git, GitHub |

## Project Structure

```text
BankSightAI/
├── artifacts/                 # Model evaluation and optimization reports
├── data/
│   ├── reports/               # EDA and monitoring outputs
│   └── ...                    # Generated datasets, depending on configuration
├── docs/
│   └── REPRODUCIBILITY.md     # Reproducibility notes, if included
├── models/                    # Saved model artifacts
├── scripts/
│   ├── generate_dataset.py
│   ├── run_eda.py
│   ├── train_models.py
│   ├── optimize_models.py
│   ├── monitor_predictions.py
│   └── load_to_db.py          # Optional database workflow
├── src/
│   └── banksight/
│       ├── api.py
│       ├── config.py
│       ├── data_validation.py
│       ├── data_ingestion.py
│       ├── db.py
│       ├── eda.py
│       ├── ml.py
│       ├── models.py
│       ├── optimization.py
│       └── dashboard.py
├── tests/
├── pyproject.toml
├── README.md
└── .gitignore
```

*The tree is representative. Keep only filenames and directories that actually exist in your repository.*

## Getting Started

### 1. Requirements

- Windows, macOS, or Linux
- Python 3.11 or a compatible version allowed by `pyproject.toml`
- Git
- VS Code (recommended)

PostgreSQL is optional if you are not using the database integration. Docker is not required for the local workflow described here.

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/BankSight-AI.git
cd BankSight-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### 3. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, use the environment's Python executable directly or follow your organization's PowerShell execution policy. Avoid changing system-wide security settings just to activate a virtual environment.

### 4. Install the project

Install the base project and the optional dependencies needed for the workflows you plan to run. Check `pyproject.toml` for the exact extra names supported by this version.

For example, if the project defines these extras:

```powershell
pip install -e ".[dev,ml,api,dashboard]"
```

If you want the optional database integration, install its dependencies too, using the extra defined in `pyproject.toml`.

## Running the Pipeline

Run commands from the repository root with the virtual environment active.

### 1. Generate synthetic data

```powershell
python scripts/generate_dataset.py
```

This generates the dataset used by the project. Check the script or `src/banksight/config.py` for the configured output location and dataset size.

### 2. Run exploratory data analysis

```powershell
python scripts/run_eda.py
```

The EDA workflow generates reports and visualizations from the configured dataset.

### 3. Train baseline models

```powershell
python scripts/train_models.py
```

This workflow trains and evaluates the baseline models and saves the configured model and evaluation artifacts.

### 4. Optimize the model

```powershell
python scripts/optimize_models.py
```

The optimization workflow performs the tuning and diagnostic steps implemented in the project, which may include threshold selection and permutation feature importance.

### 5. Start the prediction API

Run this after the optimized model artifact has been generated:

```powershell
python -m uvicorn banksight.api:app --app-dir src --host 127.0.0.1 --port 8000
```

Open the interactive API documentation in your browser:

- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health
- Model information: http://127.0.0.1:8000/model-info

The prediction endpoint is:

```text
POST /predict
```

Use the request schema displayed in `/docs` to construct a valid request. Do not submit real personal or financial information.

### 6. Run the dashboard

Open a **second PowerShell terminal**, activate the virtual environment, and run:

```powershell
python -m streamlit run src/banksight/dashboard.py
```

The dashboard communicates with the API using its configured base URL. Start the API first and check the configuration if the dashboard cannot connect.

### 7. Generate monitoring reports

After making sample predictions through the API, stop the server if necessary and run:

```powershell
python scripts/monitor_predictions.py
```

The script processes the available prediction-event log and generates the monitoring output configured by the project.

### 8. Run tests and code-quality checks

```powershell
python -m pytest -v
ruff check .
ruff format --check .
python -m compileall -q src scripts
```

These commands check the automated tests, lint rules, formatting, and Python syntax. A command should be reported as passing only when it completes successfully.

## Model Evaluation and Diagnostics

BankSight AI is designed to make model assessment more transparent by examining several aspects of the workflow:

- Classification metrics on held-out test data
- Comparison of baseline and optimized models
- Confusion matrix and ROC visualization, where generated by the evaluation scripts
- Hyperparameter tuning results
- Permutation-based feature importance
- Subgroup diagnostics, where supported by the current implementation
- Prediction-event monitoring reports

The exact metrics, selected model, thresholds, and reported results depend on the generated dataset, random seed, configuration, and evaluation procedure. Refer to the generated reports for actual values rather than assuming a particular model score.

## API Overview

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check API health |
| `GET /model-info` | Inspect available model metadata |
| `POST /predict` | Request a prediction using the supported input schema |

The API is intended to demonstrate model serving and request validation. Prediction logs should avoid storing raw applicant inputs or unnecessary personal information.

## Reproducibility

For repeatable experiments:

1. Use the configured random seeds where supported.
2. Keep the dependency versions and Python environment consistent.
3. Generate data before running downstream workflows.
4. Train and optimize models before starting the prediction API.
5. Run evaluation on held-out data and inspect the saved reports.
6. Record configuration changes when comparing experiments.

See `docs/REPRODUCIBILITY.md` if this file is present in the repository.

## Limitations and Responsible Use

- The dataset is synthetic and may not represent real-world credit behavior.
- Evaluation results on synthetic data do not establish real-world predictive performance.
- Feature importance indicates model behavior, not causation.
- Subgroup results may be unstable or misleading when sample sizes are small.
- The project does not establish regulatory compliance, fairness, or suitability for actual lending decisions.
- The system is an educational demonstration, not a production financial service.

## Future Improvements

- Add continuous integration for automated tests and lint checks.
- Track model versions and compare experiments over time.
- Add data-drift and prediction-distribution alerts.
- Improve API load testing and performance measurement.
- Add more robust validation and reproducible model evaluation.
- Expand monitoring tests and document known limitations.

## Author

**Yash Kalbhile**

Software engineering and machine learning portfolio project.

- GitHub: https://github.com/yash45615
- LinkedIn: https://www.linkedin.com/in/yash-kalbhile-697727231/

## License

Add a `LICENSE` file before describing the project as open source under a particular license. If no license is included, normal copyright restrictions still apply.
