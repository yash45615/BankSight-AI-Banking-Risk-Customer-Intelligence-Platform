"""Database connection and schema initialization."""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url

from banksight.models import Base


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create a SQLAlchemy engine from DATABASE_URL."""
    database_url = os.getenv("DATABASE_URL", "").strip()

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Configure the PostgreSQL "
            "connection string in your terminal environment."
        )

    url = make_url(database_url)

    if url.get_backend_name() != "postgresql":
        raise ValueError("DATABASE_URL must use a PostgreSQL driver.")

    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=3,
        max_overflow=2,
    )


def initialize_database() -> None:
    """Create any missing BankSight AI tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def close_database() -> None:
    """Dispose of the cached engine and its connections."""
    if get_engine.cache_info().currsize:
        get_engine().dispose()
        get_engine.cache_clear()
