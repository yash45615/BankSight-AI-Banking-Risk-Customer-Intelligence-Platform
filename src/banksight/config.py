"""Central configuration for BankSight AI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable safely."""
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes", "on"}:
        return True

    if normalized in {"false", "0", "no", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean value such as true or false.")


@dataclass(frozen=True, slots=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_name: str = "BankSight AI"
    environment: str = "development"
    log_level: str = "INFO"
    random_seed: int = 42
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    enable_demo_mode: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings from environment variables."""
        random_seed_value = os.getenv("BANKSIGHT_RANDOM_SEED", "42")

        try:
            random_seed = int(random_seed_value)
        except ValueError as exc:
            raise ValueError("BANKSIGHT_RANDOM_SEED must be an integer.") from exc

        if random_seed < 0:
            raise ValueError("BANKSIGHT_RANDOM_SEED must be non-negative.")

        environment = os.getenv("BANKSIGHT_ENVIRONMENT", "development").strip().lower()

        if environment not in {"development", "testing", "production"}:
            raise ValueError(
                "BANKSIGHT_ENVIRONMENT must be development, testing, or production."
            )

        log_level = os.getenv("BANKSIGHT_LOG_LEVEL", "INFO").strip().upper()

        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("BANKSIGHT_LOG_LEVEL is not a valid logging level.")

        return cls(
            app_name=os.getenv("BANKSIGHT_APP_NAME", "BankSight AI"),
            environment=environment,
            log_level=log_level,
            random_seed=random_seed,
            data_dir=Path(os.getenv("BANKSIGHT_DATA_DIR", str(PROJECT_ROOT / "data"))),
            model_dir=Path(
                os.getenv("BANKSIGHT_MODEL_DIR", str(PROJECT_ROOT / "models"))
            ),
            artifacts_dir=Path(
                os.getenv(
                    "BANKSIGHT_ARTIFACTS_DIR",
                    str(PROJECT_ROOT / "artifacts"),
                )
            ),
            enable_demo_mode=_read_bool("BANKSIGHT_DEMO_MODE", True),
        )

    def ensure_directories(self) -> None:
        """Create application directories when they are missing."""
        directories = (
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "reports",
            self.model_dir,
            self.artifacts_dir,
        )

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()
