"""Layer 4a — Uplift Engine: heterogeneous CATE estimation via EconML DML.

Estimates the **Conditional Average Treatment Effect** (CATE) of a discount
(treatment = ``received_discount``) on churn (outcome = ``Churn``), while
controlling for confounders ``X`` (all features).

The module uses ``CausalForestDML`` which:

1. Fits nuisance models for outcome ``m(X) = E[Y|X]`` and treatment propensity
   ``e(X) = E[T|X]`` using GradientBoostingRegressor/Classifier.
2. Computes residuals ``Y~ = Y - m(X)``, ``T~ = T - e(X)``.
3. Regresses ``Y~`` on ``T~`` with effect modifiers → yields ``τ(X) = CATE``.
4. Uses cross-fitting (default ``cv=5``) to prevent own-observation bias.

Since ``Y = 1`` is churn, a **beneficial** discount has **negative** CATE.
The module exposes an ``uplift`` score (= ``-CATE``) so that larger positive
values always mean "more likely to respond positively to treatment".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

from optiretain.config import MODELS_DIR, SEED

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

DML_MODEL_PATH = MODELS_DIR / "dml_cate.joblib"


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class CATEResult:
    """Container for DML output."""
    cate: np.ndarray            # raw CATE (negative = beneficial)
    uplift: np.ndarray          # -CATE (positive = beneficial)
    cate_lb: np.ndarray         # lower bound of effect_interval
    cate_ub: np.ndarray         # upper bound of effect_interval
    estimator: Any              # fitted EconML CausalForestDML object
    metadata: dict[str, Any]    # training params, R² scores, etc.


# ── Public API ───────────────────────────────────────────────────────────────────

def fit_dml_cate(
    X_features: pd.DataFrame | np.ndarray,
    W_confounders: Optional[pd.DataFrame | np.ndarray] = None,
    treatment: np.ndarray | pd.Series | list[int] = None,
    outcome: np.ndarray | pd.Series | list[int] = None,
    df_with_treatment: Optional[pd.DataFrame] = None,
    *,
    discount_col: str = "received_discount",
    churn_col: str = "Churn",
    n_estimators: int = 300,
    cv: int = 5,
    random_state: int = SEED,
) -> CATEResult:
    """Fit a CausalForestDML estimator and return CATE for every customer.

    Parameters
    ----------
    X_features : pd.DataFrame or np.ndarray
        Effect modifiers — features that modify the treatment effect (the X matrix).
    W_confounders : pd.DataFrame or np.ndarray, optional
        Pure confounders — features that affect treatment assignment but not
        the heterogeneous treatment effect. Pass ``None`` to include all in X.
    treatment : array-like, optional
        Binary treatment indicator. If *df_with_treatment* is provided, extracted
        from *discount_col*.
    outcome : array-like, optional
        Churn target (0/1). If *df_with_treatment* is provided, extracted from
        *churn_col*.
    df_with_treatment : pd.DataFrame, optional
        DataFrame containing both treatment and outcome columns. Convenience parameter.
    discount_col : str
        Column name for the treatment indicator in *df_with_treatment*.
    churn_col : str
        Column name for the outcome in *df_with_treatment*.
    n_estimators : int
        Number of trees in the causal forest (final stage).
    cv : int
        Number of cross-fitting folds.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    CATEResult
        Named tuple with *cate*, *uplift*, confidence bounds, fitted estimator,
        and metadata (nuisance R² scores).
    """
    # ── 0. Extract treatment/outcome from DataFrame if provided ────────────
    if df_with_treatment is not None:
        if treatment is None:
            treatment = df_with_treatment[discount_col].values
        if outcome is None:
            outcome = df_with_treatment[churn_col].values

    T = np.asarray(treatment).flatten()
    Y = np.asarray(outcome).flatten()
    X = np.asarray(X_features)
    W = np.asarray(W_confounders) if W_confounders is not None else None

    logger.info("Fitting CausalForestDML: %d samples, %d features (X), cv=%d",
                len(Y), X.shape[1], cv)

    # ── 1. Import and configure EconML DML ─────────────────────────────────
    try:
        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    except ImportError as exc:
        raise ImportError(
            "EconML is required for the Uplift Engine. Install with: uv add econml"
        ) from exc

    # Use GBMs as nuisance models (robust to non-linear confounding).
    model_y = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=random_state,
    )
    model_t = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=random_state,
    )

    # Causal forest: heterogeneous treatment effects.
    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        cv=cv,
        n_estimators=n_estimators,
        min_samples_leaf=20,
        random_state=random_state,
        max_samples=0.45,  # must be ≤ 0.5 when inference=True (EconML GRF constraint)
    )

    est.fit(Y=Y, T=T, X=X, W=W)

    # ── 2. Extract CATE and confidence intervals ───────────────────────────
    cate_hat = est.effect(X)                     # shape (n,)
    cate_lb, cate_ub = est.effect_interval(X, alpha=0.05)  # 95% CI

    uplift = -cate_hat                           # positive = good

    logger.info("CATE — median: %.4f, IQR: [%.4f, %.4f]",
                float(np.median(cate_hat)),
                float(np.percentile(cate_hat, 25)),
                float(np.percentile(cate_hat, 75)))
    logger.info("Uplift — median: %.4f, proportion > 0: %.2f%%",
                float(np.median(uplift)),
                float(100 * (uplift > 0).mean()))

    # ── 3. Nuisance model R² scores (for metadata / validation) ────────────
    # Fit separate nuisance models to compute R² on full data.
    y_hat = est.model_y_.predict(X) if hasattr(est, "model_y_") else None
    r2_y = float(np.corrcoef(Y, y_hat)[0, 1] ** 2) if y_hat is not None else None

    metadata: dict[str, Any] = {
        "method": "CausalForestDML",
        "n_estimators": n_estimators,
        "cv": cv,
        "n_samples": len(Y),
        "n_features_x": X.shape[1],
        "n_features_w": W.shape[1] if W is not None else 0,
        "treatment_rate": float(T.mean()),
        "churn_rate": float(Y.mean()),
        "cate_median": float(np.median(cate_hat)),
        "cate_std": float(cate_hat.std()),
        "uplift_median": float(np.median(uplift)),
        "uplift_positive_pct": float((uplift > 0).mean()),
        "nuisance_r2_y": r2_y,
    }

    result = CATEResult(
        cate=cate_hat,
        uplift=uplift,
        cate_lb=cate_lb,
        cate_ub=cate_ub,
        estimator=est,
        metadata=metadata,
    )

    # ── 4. Persist ─────────────────────────────────────────────────────────
    persist_cate(result, X_features)

    return result


def predict_uplift(
    X_features: pd.DataFrame | np.ndarray,
    *,
    model_path: Path = DML_MODEL_PATH,
) -> np.ndarray:
    """Load a persisted DML model and return uplift scores for new customers.

    Parameters
    ----------
    X_features : pd.DataFrame or np.ndarray
        Effect modifier features (must match training schema).
    model_path : Path
        Path to the persisted ``dml_cate.joblib`` file.

    Returns
    -------
    np.ndarray
        Uplift scores (positive = discount reduces churn) in ``(n_samples,)`` shape.
    """
    loaded = load(model_path)
    est = loaded["estimator"]
    X_arr = np.asarray(X_features)
    return -est.effect(X_arr)


def persist_cate(result: CATEResult, X_features: pd.DataFrame | np.ndarray) -> Path:
    """Persist the DML result to disk.

    Returns
    -------
    Path
        The path where the model was saved.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    import joblib
    data = {
        "estimator": result.estimator,
        "cate": result.cate,
        "uplift": result.uplift,
        "cate_lb": result.cate_lb,
        "cate_ub": result.cate_ub,
        "metadata": result.metadata,
    }
    dump(data, DML_MODEL_PATH)

    # Also save metadata as JSON for quick inspection.
    import json
    meta_path = MODELS_DIR / "dml_cate_metadata.json"
    with open(meta_path, "w") as f:
        serializable_meta = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                              for k, v in result.metadata.items()}
        json.dump(serializable_meta, f, indent=2, default=str)

    logger.info("DML model persisted → %s", DML_MODEL_PATH)
    return DML_MODEL_PATH


# ── Convenience: ground-truth CATE validation on synthetic data ──────────────────

def compute_ground_truth_cate(df: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """Compute the *ground-truth* heterogeneous treatment effect τ(X) for the
    synthetic Telco dataset. Used in tests to validate DML recovery.

    The τ(X) function follows the plan's specification:
    Gaussian(tenure; center=36, width=18) × (-0.05)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``tenure`` column.
    seed : int
        Random seed (unused here — τ(X) is deterministic).

    Returns
    -------
    np.ndarray
        Ground-truth CATE values with the same sign convention as EconML
        (negative = beneficial, i.e. discount reduces churn).
    """
    from scipy.special import expit

    tenure = df["tenure"].values
    center, width = 36.0, 18.0
    gaussian = np.exp(-0.5 * ((tenure - center) / width) ** 2)
    tau = gaussian * (-0.05)  # negative = discount reduces churn

    return tau
