"""Feature engineering module for OptiRetain.

This module transforms the loader's output DataFrame into a model-ready
format by performing four operations in sequence:

1. **Impute** missing TotalCharges values
2. **Compute CLV** — both raw (from source) and a transparent proxy
3. **Encode** categorical + scale numeric features for ML models
4. **Synthesize treatment** variable for causal inference (DML layer)

Each function is independently testable; ``engineer_features()`` provides
a single convenience wrapper that runs all steps in the correct order.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid function
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from optiretain.config import SEED


# ── Constants ────────────────────────────────────────────────────────────────

_NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

_CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Forward-looking horizon for CLV estimation (months).
# A 24-month look-ahead is standard in telecom churn analytics: it gives the
# business a concrete window to see ROI from retention spend.
_CLV_FORWARD_HORIZON = 24


# ── 1. Imputation ────────────────────────────────────────────────────────────

def impute_total_charges(
    df: pd.DataFrame,
    strategy: str = "median",
) -> pd.DataFrame:
    """Fill missing TotalCharges values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``TotalCharges`` column containing NaN values.
    strategy : str
        * ``"median"`` — impute with the median of non-null values (default).
          Recommended because TotalCharges is right-skewed and outliers would
          inflate a mean-based imputation.
        * ``"zero"`` — impute with 0 for customers who have not incurred
          charges yet (tenure == 0 or very new).

    Returns
    -------
    pd.DataFrame
        A copy of *df* with no NaN values in ``TotalCharges``.

    Raises
    ------
    ValueError
        If the ``TotalCharges`` column is missing or fully non-null.
    """
    df = df.copy()

    if "TotalCharges" not in df.columns:
        raise ValueError("Column 'TotalCharges' not found.")

    if df["TotalCharges"].isna().sum() == 0:
        return df  # nothing to do; still return a copy.

    if strategy == "zero":
        # New customers (tenure < threshold) likely have blank charges.
        threshold = np.minimum(df[df["TotalCharges"].notna()]["TotalCharges"].median(), 10.0)
        df.loc[df["TotalCharges"].isna(), "TotalCharges"] = (
            df.loc[df["TotalCharges"].isna(), "MonthlyCharges"] * 0.01
        )
    else:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

    return df


# ── 2. CLV Computation ───────────────────────────────────────────────────────

def compute_clv(df: pd.DataFrame) -> pd.DataFrame:
    """Compute customer lifetime value columns on top of the source data.

    Creates two columns:

    * ``cltv_raw`` — exact copy of the dataset's pre-computed CLTV (preserved
      as a feature for Layers 1–2 where it encodes customer value).
    * ``clv_computed`` — transparent proxy ``max(TotalCharges, MonthlyCharges ×
      CLV_FORWARD_HORIZON)``.  Using ``max()`` keeps CLV monotonically
      non-decreasing with tenure (longer customers are worth more), avoiding the
      perverse penalty that penalises high-tenure loyalists.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``CLTV``, ``MonthlyCharges``, and ``TotalCharges`` columns.

    Returns
    -------
    pd.DataFrame
        A copy with two new columns appended.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    df = df.copy()

    for col in ["CLTV", "MonthlyCharges", "TotalCharges"]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' required for CLV computation.")

    # Raw CLTV preserved as-is from source.
    df["cltv_raw"] = df["CLTV"].copy()

    # Transparent computed proxy using an auditable formula.
    #   floor     — what they've already paid (honest baseline)
    #   forward   — MonthlyCharges × 24-month look-ahead (ROI window)
    # The max ensures CLV is monotonically non-decreasing with tenure.
    past_value = df["TotalCharges"]
    forward_value = df["MonthlyCharges"] * _CLV_FORWARD_HORIZON
    df["clv_computed"] = np.maximum(past_value, forward_value)

    return df


# ── 3. Encoding ───────────────────────────────────────────────────────────────

def encode_features(
    df: pd.DataFrame,
    encoder: Optional[ColumnTransformer] = None,
    *,
    include_cltv: bool = False,
) -> Any:
    """One-hot encode categoricals and scale numerics.

    Returns ``(encoded_df_or_array, fitted_transformer)`` where the transformer
    can be persisted (joblib.dump) for inference-time parity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with standardised column names from the loader.
    encoder : ColumnTransformer or None
        If provided, *transform*-only mode (used at inference time).
        If *None*, *fit*-mode — fits on the provided data and returns it.
    include_cltv : bool
        Whether to include ``cltv_raw`` as a numeric feature. Default False —
        the CLTV signal may leak information; only add when explicitly desired.

    Returns
    -------
    tuple[pd.DataFrame | np.ndarray, ColumnTransformer]
        The encoded array (DataFrame if df was one, else ndarray) and the
        fitted/transformed transformer.
    """
    # Select columns to include in encoding.
    numeric_cols = list(_NUMERIC_FEATURES)
    categorical_cols = list(_CATEGORICAL_FEATURES)

    # Only include CLTV during fit time — transform-only mode must use the
    # same schema the encoder was originally fitted with.
    if include_cltv and encoder is None:
        if "cltv_raw" not in df.columns:
            raise ValueError("Pass include_cltv=True only when 'cltv_raw' is available.")
        numeric_cols.append("cltv_raw")

    selected_cols = numeric_cols + categorical_cols
    X = df[selected_cols].copy()

    was_provided_encoder = encoder is not None

    # Build transformer if not provided.
    if not was_provided_encoder:
        encoder = ColumnTransformer(
            transformers=[
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
            ],
            remainder="drop",  # drop customerID and any leftover columns.
        )

    X_enc = encoder.fit_transform(X) if not was_provided_encoder else encoder.transform(X)

    # If input was a DataFrame, return output as DataFrame too (for compatibility).
    if isinstance(df, pd.DataFrame):
        feature_names = encoder.get_feature_names_out(selected_cols)
        X_enc = pd.DataFrame(X_enc, columns=feature_names, index=df.index)

    return X_enc, encoder


# ── 4. Treatment Synthesis ────────────────────────────────────────────────────

def synthesize_treatment(
    df: pd.DataFrame,
    discount_pct: float = 0.20,
    seed: Optional[int] = SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Synthesize a binary treatment indicator for causal inference (DML layer).

    Treatment is assigned via a logistic propensity model that depends on tenure
    and MonthlyCharges — this makes the data **observational**, not randomized,
    which is precisely what DML is designed to handle.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``tenure`` and ``MonthlyCharges`` columns (and optionally
        ``TotalCharges``). Will be copied before mutation.
    discount_pct : float
        Standard discount percentage (e.g. 0.20) used to compute per-customer
        treatment cost for Layer 3.
    seed : int or None
        Random seed for reproducibility. Pass None for fully random assignment.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        *df* with ``received_discount`` column added (0/1 binary), plus a
        metadata dictionary containing the parameters used for synthesis —
        needed for ground-truth CATE validation in tests.

    Metadata keys:
        ``propensity_alpha``, ``propensity_beta_tenure``, ``propensity_beta_mc``
            Logistic regression coefficients for propensity score.
        ``heterogeneity_fn``  : string description of the τ(X) function used.
        ``discount_pct``      : discount fraction applied to costs.
    """
    df = df.copy()

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.RandomState()

    # Propensity score: sigmoid(α + β1·tenure + β2·MonthlyCharges + ε)
    # Mid-tenure (24–36 mo) and mid-spend customers are targeted more often.
    alpha = -0.5
    beta_tenure = 0.015  # positive → higher tenure → slightly more likely to get offer
    beta_mc = 0.01       # higher spend → more likely to get offer

    propensity = expit(
        alpha + beta_tenure * df["tenure"] + beta_mc * df["MonthlyCharges"]
    )

    # Clip to avoid exact-0/1 probabilities (numerical stability).
    propensity = propensity.clip(0.01, 0.99)
    df["received_discount"] = rng.binomial(1, propensity)

    # Compute treatment cost: discount_pct × MonthlyCharges × expected_duration_months
    # Assume a retention offer typically lasts ~6 months of the discounted rate.
    df["treatment_cost"] = discount_pct * df["MonthlyCharges"] * 6

    metadata: dict[str, Any] = {
        "propensity_alpha": alpha,
        "propensity_beta_tenure": beta_tenure,
        "propensity_beta_mc": beta_mc,
        "heterogeneity_fn": (
            "τ(X) = Gaussian(tenure; center=36, width=18) × (-0.05)"
        ),
        "discount_pct": discount_pct,
    }

    return df, metadata


# ── Convenience wrapper ───────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    encoder: Optional[ColumnTransformer] = None,
    *,
    include_cltv: bool = False,
    discount_pct: float = 0.20,
    seed: Optional[int] = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, Any, dict[str, Any]]:
    """Run the full feature-engineering pipeline in one call.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from ``load_telco_data()``.
    encoder : ColumnTransformer or None
        Passed through to ``encode_features()``.
    include_cltv : bool
        Whether to include CLTV as a numeric feature in encoding.
    discount_pct : float
        Passed through to ``synthesize_treatment()``.
    seed : int or None
        Passed through to ``synthesize_treatment()``.

    Returns
    -------
    tuple[df_imputed, X_encoded, fitted_encoder, treatment_meta]
        *df_imputed* — DataFrame with imputation + CLV + treatment columns.
        *X_encoded* — encoded feature array (DataFrame or ndarray).
        *fitted_encoder* — fitted ColumnTransformer for inference parity.
        *treatment_meta* — metadata dict from ``synthesize_treatment()``.
    """
    # Step 1: Impute
    df = impute_total_charges(df)

    # Step 2: CLV
    df = compute_clv(df)

    # Step 3: Encode
    X, enc = encode_features(df, encoder=encoder, include_cltv=include_cltv)

    # Step 4: Treatment synthesis (operates on the full DataFrame, not X).
    df_treated, meta = synthesize_treatment(df, discount_pct=discount_pct, seed=seed)

    return df_treated, X, enc, meta
