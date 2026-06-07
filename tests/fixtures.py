"""Reusable fixture utilities for OptiRetain model tests.

These helpers are intended for *future* test files (risk, uplift, optimize).
Each function produces a small, deterministic DataFrame that exercises the
boundaries of what a full dataset can be tested against — no network calls,
no heavy computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_mock_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Build a small synthetic churn DataFrame matching the loader's schema.

    Parameters
    ----------
    n : int
        Number of rows to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same column names and approximate dtypes as the
        real dataset after loader processing.
    """
    rng = np.random.default_rng(seed)

    churn_prob = 0.26  # matches source Telco rate
    churn = (rng.random(n) < churn_prob).astype(int)
    tenure = rng.integers(0, 73, size=n)
    monthly = rng.uniform(18.0, 120.0, size=n)
    total = monthly * tenure + rng.normal(0, 50, size=n)  # approximate
    cltv = total + monthly * (60 - tenure) + rng.integers(0, 500, size=n)

    cat_cols = {
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["Yes", "No"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    }

    df = pd.DataFrame({
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], size=n),
        "SeniorCitizen": rng.choice([0, 1], size=n),
        "Partner": rng.choice(["Yes", "No"], size=n),
        "Dependents": rng.choice(["Yes", "No"], size=n),
        "tenure": tenure.astype(float),
        "PhoneService": rng.choice(cat_cols["PhoneService"], size=n),
        "MultipleLines": rng.choice(cat_cols["MultipleLines"], size=n),
        "InternetService": rng.choice(cat_cols["InternetService"], size=n),
        "OnlineSecurity": rng.choice(cat_cols["OnlineSecurity"], size=n),
        "OnlineBackup": rng.choice(cat_cols["OnlineBackup"], size=n),
        "DeviceProtection": rng.choice(cat_cols["DeviceProtection"], size=n),
        "TechSupport": rng.choice(cat_cols["TechSupport"], size=n),
        "StreamingTV": rng.choice(cat_cols["StreamingTV"], size=n),
        "StreamingMovies": rng.choice(cat_cols["StreamingMovies"], size=n),
        "Contract": rng.choice(cat_cols["Contract"], size=n),
        "PaperlessBilling": rng.choice(cat_cols["PaperlessBilling"], size=n),
        "PaymentMethod": rng.choice(cat_cols["PaymentMethod"], size=n),
        "MonthlyCharges": monthly,
        "TotalCharges": np.maximum(total, 0.0),
        "Churn": churn,
        "CLTV": cltv.clip(lower=1),
    })

    return df


def make_train_test_split(df: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42):
    """Stratified train/test split using pandas (no sklearn dependency for fixtures)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        df.drop(columns=["customerID"]),
        df["Churn"],
        test_size=test_ratio,
        stratify=df["Churn"],
        random_state=seed,
    )
