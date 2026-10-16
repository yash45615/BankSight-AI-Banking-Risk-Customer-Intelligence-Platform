"""Tests for BankSight AI configuration."""

from pathlib import Path

import pytest

from banksight.config import Settings


def test_default_settings_are_valid() -> None:
    settings = Settings.from_env()

    assert settings.app_name == "BankSight AI"
    assert settings.environment == "development"
    assert settings.random_seed == 42
    assert settings.enable_demo_mode is True


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BANKSIGHT_APP_NAME", "BankSight Test")
    monkeypatch.setenv("BANKSIGHT_ENVIRONMENT", "testing")
    monkeypatch.setenv("BANKSIGHT_RANDOM_SEED", "123")
    monkeypatch.setenv("BANKSIGHT_DEMO_MODE", "false")

    settings = Settings.from_env()

    assert settings.app_name == "BankSight Test"
    assert settings.environment == "testing"
    assert settings.random_seed == 123
    assert settings.enable_demo_mode is False


def test_invalid_random_seed_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BANKSIGHT_RANDOM_SEED", "not-a-number")

    with pytest.raises(ValueError, match="must be an integer"):
        Settings.from_env()


def test_invalid_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BANKSIGHT_ENVIRONMENT", "unknown")

    with pytest.raises(ValueError, match="must be development"):
        Settings.from_env()


def test_invalid_boolean_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BANKSIGHT_DEMO_MODE", "maybe")

    with pytest.raises(ValueError, match="must be a boolean"):
        Settings.from_env()


def test_ensure_directories_creates_required_paths(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        artifacts_dir=tmp_path / "artifacts",
    )

    settings.ensure_directories()

    assert (tmp_path / "data" / "raw").is_dir()
    assert (tmp_path / "data" / "processed").is_dir()
    assert (tmp_path / "data" / "reports").is_dir()
    assert (tmp_path / "models").is_dir()
    assert (tmp_path / "artifacts").is_dir()
