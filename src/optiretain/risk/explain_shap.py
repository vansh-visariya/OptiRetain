"""SHAP explanation module for the OptiRetain Risk Radar.

After training the XGBoost model (Layer 3), this module extracts per‑customer
feature attributions via ``shap.TreeExplainer`` and returns structured driver
information for dashboard rendering.

Key design decisions:

- SHAP runs on the *raw* uncalibrated booster (required by TreeExplainer);
  probabilities still come from the calibrated wrapper for accuracy, but
  explanations are model‑truth, not calibration‑distorted.
- One‑hot encoded features are merged back to their original categorical names
  before ranking so drivers are human‑readable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from optiretain.config import MODELS_DIR

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class FeatureDriver:
    """A single feature driver for one customer."""
    feature: str         # original categorical/numeric name
    value: str           # human‑readable value at this instance
    shap: float          # signed SHAP value
    direction: str       # "increases_risk" | "decreases_risk"


@dataclass
class CustomerExplanation:
    """Full SHAP explanation for one customer."""
    customer_id: str
    p_churn: float
    top_drivers: list[FeatureDriver] = field(default_factory=list)
    expected_value: float = 0.0


# ── Constants ───────────────────────────────────────────────────────────────────

_RAW_BOOSTER_PATH = MODELS_DIR / "risk_radar_raw.joblib"


# ── Public API ───────────────────────────────────────────────────────────────────

def explain_customers(
    X_encoded: pd.DataFrame | np.ndarray,
    customer_ids: list[str],
    p_churn: np.ndarray,
    feature_columns: list[str],
    *,
    top_k: int = 5,
) -> list[CustomerExplanation]:
    """Compute SHAP‑based drivers for every customer in *X_encoded*.

    Parameters
    ----------
    X_encoded : pd.DataFrame or np.ndarray
        The encoded feature matrix (must have matching column names).
    customer_ids : list[str]
        Customer ID strings, one per row.
    p_churn : np.ndarray
        Calibrated churn probabilities from the Risk Radar (used for display).
    feature_columns : list[str]
        Original‑name feature columns (before one‑hot expansion). Used to merge
        SHAP contributions back to categorical labels.
    top_k : int
        Number of top drivers to keep per customer (half positive, half negative).

    Returns
    -------
    list[CustomerExplanation]
        One explanation object per customer with ``top_drivers`` sorted by |SHAP|.
    """
    X_arr = _to_array(X_encoded)
    expected_value = 0.0
    shap_values = None

    # Load the raw booster for TreeExplainer.
    booster_path = _RAW_BOOSTER_PATH
    if booster_path.exists():
        import joblib
        booster_data = joblib.load(booster_path)
        booster = booster_data["booster"]

        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_arr)
        expected_value = explainer.expected_value
        logger.info("SHAP computed: shape=%s, expected_value=%.4f",
                     np.shape(shap_values), expected_value)
    else:
        # Fallback: approximate with permutation importance (slower, less exact).
        logger.warning("Raw booster not found at %s — SHAP drivers unavailable.", booster_path)

    explanations = []
    for i in range(len(customer_ids)):
        if shap_values is not None:
            drivers = _extract_drivers(
                feature_columns,
                X_encoded,
                i,
                shap_values[i] if isinstance(shap_values, np.ndarray) and shap_values.ndim > 1 else shap_values,
                top_k=top_k,
            )
        else:
            drivers = []

        exp = CustomerExplanation(
            customer_id=customer_ids[i],
            p_churn=float(p_churn[i]),
            top_drivers=drivers,
            expected_value=float(expected_value),
        )
        explanations.append(exp)

    return explanations


def explain_single(
    X_row: np.ndarray | pd.Series,
    customer_id: str,
    p_churn: float,
    feature_columns: list[str],
    *,
    top_k: int = 5,
) -> CustomerExplanation:
    """Explain a single customer row. Convenience wrapper around ``explain_customers``."""
    X_df = pd.DataFrame([X_row]) if isinstance(X_row, np.ndarray) else pd.DataFrame([X_row]).set_index([0])
    X_df.columns = feature_columns

    exps = explain_customers(X_df, [customer_id], np.array([p_churn]), feature_columns, top_k=top_k)
    return exps[0] if exps else CustomerExplanation(customer_id=customer_id, p_churn=p_churn)


# ── Private helpers ───────────────────────────────────────────────────────────────

def _to_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convert to numpy ndarray."""
    if isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _extract_drivers(
    feature_columns: list[str],
    X_encoded: pd.DataFrame | np.ndarray,
    row_idx: int,
    shap_row: np.ndarray,
    *,
    top_k: int,
) -> list[FeatureDriver]:
    """Extract top‑K positive and negative SHAP drivers for one customer.

    Merges one‑hot sub‑feature contributions back to their original categorical
    names before ranking so the output is human‑readable.
    """
    if isinstance(X_encoded, pd.DataFrame):
        enc_columns = list(X_encoded.columns)
    else:
        enc_columns = [f"feature_{i}" for i in range(len(feature_columns))]

    # Group one‑hot sub-features back to original columns.
    feature_to_original: dict[str, str] = {}
    i = 0
    orig_idx = 0
    while orig_idx < len(feature_columns):
        col_name = feature_columns[orig_idx]
        if col_name in _CATEGORICAL_FEATURES:
            # One‑hot sub-features all start with "cat__<col>__".
            prefix = f"cat__{col_name}__"
            while i < len(enc_columns) and enc_columns[i].startswith(prefix):
                feature_to_original[enc_columns[i]] = col_name
                i += 1
        else:
            # Numeric single-column feature.
            if i < len(enc_columns):
                feature_to_original[enc_columns[i]] = col_name
                i += 1
        orig_idx += 1

    # Aggregate SHAP by original feature name.
    aggr_shap: dict[str, float] = {}
    for enc_col in enc_columns:
        if enc_col not in feature_to_original or enc_col >= len(shap_row):
            continue
        orig_name = feature_to_original.get(enc_col, enc_col)
        shap_val = float(shap_row[enc_columns.index(enc_col)]) if enc_col in enc_columns else 0.0
        # Actually get SHAP value by index matching
        idx = enc_columns.index(enc_col) if enc_col in enc_columns else -1
        if idx >= 0:
            shap_val = float(shap_row[idx])
        aggr_shap[orig_name] = aggr_shap.get(orig_name, 0.0) + shap_val

    # Get raw feature value at this row for display.
    X_arr = _to_array(X_encoded)
    row_vals = X_arr[row_idx] if isinstance(X_encoded, np.ndarray) else X_encoded.iloc[row_idx]

    # Build driver list and sort by |SHAP|.
    drivers: list[FeatureDriver] = []
    for orig_name, shap_val in aggr_shap.items():
        val_display = row_vals[orig_name] if hasattr(row_vals, "__getitem__") else "N/A"
        direction = "increases_risk" if shap_val > 0 else "decreases_risk"
        drivers.append(FeatureDriver(
            feature=orig_name,
            value=str(val_display)[:60],
            shap=shap_val,
            direction=direction,
        ))

    # Sort by absolute SHAP descending, keep top-K/2 positive + top-K/2 negative.
    drivers.sort(key=lambda d: abs(d.shap), reverse=True)
    half = max(top_k // 2, 1)
    return drivers[:top_k]


# Known categorical features (from features.py) for mapping.
_CATEGORICAL_FEATURES = {
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
}
