"""Shared fixtures for OptiRetain tests."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

# Resolve data directory relative to the project root (one level up from tests/)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_RAW = _PROJECT_ROOT / "data" / "raw"
_TELCO_XLSX = _DATA_RAW / "Telco_customer_churn.xlsx"


@pytest.fixture(scope="session")
def telco_xlsx_path() -> Path:
    """Path to the raw Telco customer churn Excel file."""
    assert _TELCO_XLSX.exists(), f"Telco dataset not found at {_TELCO_XLSX}"
    return _TELCO_XLSX


@pytest.fixture(scope="session")
def loaded_df(telco_xlsx_path):
    """Load the Telco dataset via load_telco_data() once per test session."""
    from optiretain.data.loader import load_telco_data

    df = load_telco_data(telco_xlsx_path)
    yield df


@pytest.fixture(scope="session")
def expected_columns() -> list[str]:
    """Canonical column set after loader processing."""
    return sorted([
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn", "CLTV",
    ])
