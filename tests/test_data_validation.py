"""Tests for credit-risk data validation."""

from scripts.generate_dataset import generate_dataset

from banksight.data_validation import validate_dataset


def test_generated_dataset_passes_validation() -> None:
    df = generate_dataset(rows=200, seed=42)

    report = validate_dataset(df)

    assert report.is_valid
    assert report.row_count == 200
    assert report.errors == []


def test_missing_column_is_rejected() -> None:
    df = generate_dataset(rows=100, seed=42).drop(columns=["annual_income"])

    report = validate_dataset(df)

    assert not report.is_valid
    assert any("annual_income" in error for error in report.errors)


def test_duplicate_customer_ids_are_rejected() -> None:
    df = generate_dataset(rows=100, seed=42)
    df.loc[df.index[1], "customer_id"] = df.loc[df.index[0], "customer_id"]

    report = validate_dataset(df)

    assert not report.is_valid
    assert any("duplicate" in error for error in report.errors)


def test_invalid_default_label_is_rejected() -> None:
    df = generate_dataset(rows=100, seed=42)
    df["default"] = df["default"].astype(int)
    df.loc[df.index[0], "default"] = 2

    report = validate_dataset(df)

    assert not report.is_valid
    assert any("default" in error for error in report.errors)


def test_negative_loan_amount_is_rejected() -> None:
    df = generate_dataset(rows=100, seed=42)
    df.loc[df.index[0], "loan_amount"] = -100

    report = validate_dataset(df)

    assert not report.is_valid
    assert any("loan_amount" in error for error in report.errors)


def test_missing_numeric_value_is_rejected() -> None:
    df = generate_dataset(rows=100, seed=42)
    df["annual_income"] = df["annual_income"].astype(float)
    df.loc[df.index[0], "annual_income"] = float("nan")

    report = validate_dataset(df)

    assert not report.is_valid
    assert any("annual_income" in error for error in report.errors)


def test_empty_dataset_is_rejected() -> None:
    df = generate_dataset(rows=100, seed=42).iloc[0:0]

    report = validate_dataset(df)

    assert not report.is_valid


def test_csv_ingestion_rejects_missing_file(tmp_path) -> None:
    from banksight.data_ingestion import load_csv

    missing_file = tmp_path / "missing.csv"

    try:
        load_csv(missing_file)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
