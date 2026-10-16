"""SQLAlchemy models for BankSight AI."""

from __future__ import annotations

from sqlalchemy import CheckConstraint, Float, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for database models."""


class CreditRiskRecord(Base):
    """A synthetic or approved credit-risk dataset record."""

    __tablename__ = "credit_risk_records"

    __table_args__ = (
        CheckConstraint("age >= 18 AND age <= 100", name="ck_age_range"),
        CheckConstraint("annual_income >= 0", name="ck_income_nonnegative"),
        CheckConstraint("loan_amount > 0", name="ck_loan_positive"),
        CheckConstraint(
            '"default" IN (0, 1)',
            name="ck_default_binary",
        ),
    )

    customer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    annual_income: Mapped[float] = mapped_column(Float, nullable=False)
    employment_years: Mapped[int] = mapped_column(Integer, nullable=False)
    loan_amount: Mapped[float] = mapped_column(Float, nullable=False)
    interest_rate: Mapped[float] = mapped_column(Float, nullable=False)
    existing_debt: Mapped[float] = mapped_column(Float, nullable=False)
    credit_history_years: Mapped[int] = mapped_column(Integer, nullable=False)
    late_payments: Mapped[int] = mapped_column(Integer, nullable=False)
    loan_purpose: Mapped[str] = mapped_column(String(32), nullable=False)
    default: Mapped[int] = mapped_column(Integer, nullable=False)
