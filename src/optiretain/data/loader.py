"""Data loader for the customer churn retention dataset.

This module provides a single public function ``load_telco_data`` that:

1. Reads the raw Excel file from ``data/raw/Telco_customer_churn.xlsx``
2. Validates the expected schema
3. Coerces ``Total Charges`` to numeric (handles blank-space rows)
4. Returns a clean DataFrame with standardised column names for downstream
   modules (risk, uplift, optimization).

Missing-value imputation, encoding, and feature engineering are deferred
to ``features.py`` so that loading remains a thin, auditable step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from optiretain.config import TELECO_FILE


# Columns to drop: geographic leakage, redundant IDs, and pre-computed churn targets
_COLUMNS_TO_DROP = {
    "Count",               # constant column (always 1)
    "Country",             # geo — no modelling value
    "State",               # geo
    "City",                # geo
    "Zip Code",            # geo
    "Lat Long",            # geographic pair
    "Latitude",            # geo component of Lat Long
    "Longitude",           # geo component of Lat Long
    "Churn Value",         # derived from Churn Label (redundant target)
    "Churn Score",         # derived churn score (redundant target)
    "Churn Reason",        # text — 73% null, no structured use in ML pipeline
}

# Mapping from raw column names to canonical names used downstream
_COLUMN_RENAME = {
    "CustomerID":     "customerID",
    "Gender":         "gender",
    "Senior Citizen": "SeniorCitizen",
    "Partner":        "Partner",
    "Dependents":     "Dependents",
    "Tenure Months":  "tenure",
    "Phone Service":  "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup":  "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support":   "TechSupport",
    "Streaming TV":   "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Contract":       "Contract",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method":  "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges":  "TotalCharges",
    "Churn Label":    "Churn",
    # Note: CLTV is kept as-is — the dataset creator's baked-in value
    # is a legitimate ML feature. A transparent proxy (clv_computed) is
    # built in features.py for Layer 3 where auditability matters.
}

# Columns remaining after rename + drop (used for validation)
_EXPECTED_COLUMNS = sorted({
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn", "CLTV",
})


def load_telco_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the customer churn dataset and return a cleaned DataFrame.

    Parameters
    ----------
    path : str or Path, optional
        Explicit path to the Excel file.  Falls back to the default
        ``data/raw/Telco_customer_churn.xlsx`` when *None*.

    Returns
    -------
    pd.DataFrame
        A DataFrame with:

        - Standardised column names (camelCase for feature columns)
        - ``TotalCharges`` coerced to ``float64`` (blanks → NaN)
        - ``Churn`` as a binary target: ``Yes/No`` mapped to ``1/0``
        - ``CLTV`` kept from source — features.py recomputes a transparent proxy
        - Geographic and pre-computed churn columns dropped

    Raises
    ------
    ValueError
        If the loaded file schema does not match expectations.
    FileNotFoundError
        If the target Excel file cannot be found.

    Examples
    --------
    >>> df = load_telco_data()       # uses default path
    >>> assert df.shape == (7043, 21)
    """
    if path is None:
        resolved = TELECO_FILE
    else:
        resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(f"Churn dataset not found at {resolved}")

    df = pd.read_excel(resolved)

    # 1. Drop unwanted columns
    drop_cols = _COLUMNS_TO_DROP & set(df.columns)
    if not drop_cols:
        raise ValueError(
            f"None of the expected columns to drop ({_COLUMNS_TO_DROP}) "
            f"were found in the loaded data. The file may be from a different source."
        )
    df = df.drop(columns=drop_cols)

    # 2. Rename to standardised names
    df.rename(columns=_COLUMN_RENAME, inplace=True)

    # 3. Validate canonical column set
    actual = sorted(df.columns.tolist())
    if actual != _EXPECTED_COLUMNS:
        missing = set(_EXPECTED_COLUMNS) - set(actual)
        extra = set(actual) - set(_EXPECTED_COLUMNS)
        raise ValueError(
            f"Schema mismatch. Expected {len(_EXPECTED_COLUMNS)} columns, "
            f"got {len(df.columns)}. Missing: {missing}; Extra: {extra}"
        )

    # 4. Coerce TotalCharges to numeric (handles ~11 blank-space rows)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 5. Map target variable to binary (handle both string and ArrowStringArray dtypes)
    if "Churn" in df.columns and df["Churn"].dtype.kind == "O":
        df["Churn"] = df["Churn"].astype(str).map({"Yes": 1, "No": 0})

    return df
