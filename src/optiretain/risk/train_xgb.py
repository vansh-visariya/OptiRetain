""" Risk Radar: train and persist an XGBoost churn-risk model.

This module performs three operations in sequence:

1. **Train** an ``XGBClassifier`` with hyperparameter tuning via
   ``RandomizedSearchCV`` on a stratified K-Fold split.
2. **Calibrate** the best estimator with isotonic regression so output
   probabilities are well‑scoped (important for Layer 3 × CLV multiplication).
3. **Persist** the trained model and its preprocessor to disk via ``joblib``.

The module also exposes a convenience ``predict()`` that loads the model
from disk, applies it to raw DataFrames, and returns calibrated churn
probabilities — ideal for inference parity at scale.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit  # sigmoid
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from optiretain.config import MODELS_DIR, SEED
from optiretain.data.features import encode_features  # reuse encoder from Layer 2

logger = logging.getLogger(__name__)

# ── Hyperparameter search space ────────────────────────────────

_XGB_PARAMS_SPACE: dict[str, list[Any]] = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "min_child_weight": [1, 3, 5],
    "gamma": [0.0, 0.2, 0.4, 0.6],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

# Default XGB parameters (outside search grid).
_XGB_BASE_PARAMS = {
    "n_estimators": 600,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "random_state": SEED,
}

# ── Constants ─────────────────────────────────────────────────────────────────

ARTIFACT_DIR = MODELS_DIR  # models/risk_radar.pkl lives here
MODEL_PATH = ARTIFACT_DIR / "risk_radar.pkl"
METADATA_PATH = ARTIFACT_DIR / "risk_radar_metadata.json"


# ── Public API ────────────────────────────────────────────────────────────────

def train_risk_radar(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    feature_names: Optional[list[str]] = None,
    test_size: float = 0.2,
    random_state: int = SEED,
) -> tuple[xgb.XGBClassifier, CalibratedClassifierCV, dict[str, Any]]:
    """Train and calibrate the churn-risk XGBoost model.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Encoded feature matrix from ``features.encode_features()``.
    y : pd.Series or np.ndarray
        Binary churn target (0/1).
    feature_names : list[str] | None
        Column names if *X* is a DataFrame; otherwise inferred from shape.
    test_size : float
        Fraction held out for final evaluation metrics.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    tuple[xgb.XGBClassifier, CalibratedClassifierCV, dict]
        *(raw_model, calibrated_model, metadata)* where metadata contains
        train/test AUC, Brier score, and best hyperparameters — all written
        to ``METADATA_PATH`` as a side effect.

    Raises
    ------
    ValueError
        If *X* or *y* have incompatible shapes or *y* is not binary.
    """
    # ── 0. Validation ────────────────────────────────────────────────────
    X_arr = _to_numpy(X)
    y_arr = _to_numpy(y, flatten=True)

    if len(y_arr.shape) != 1:
        raise ValueError("y must be a 1-D array-like.")
    if set(np.unique(y_arr)) - {0, 1}:
        raise ValueError("y must contain only binary labels {0, 1}.")

    n_samples, n_features = X_arr.shape
    _check_train_test_split(X_arr, y_arr, test_size)

    # ── 1. Compute scale_pos_weight (imbalance correction) ───────────────
    pos_count = int(y_arr.sum())
    neg_count = len(y_arr) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    base_params = {**_XGB_BASE_PARAMS, "scale_pos_weight": scale_pos_weight}

    # ── 2. Train test split ───────────────────────────────────────────────
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X_arr, y_arr, test_size=test_size, stratify=y_arr, random_state=random_state
    )

    # ── 3. Hyperparameter tuning ──────────────────────────────────────────
    logger.info("Starting RandomizedSearchCV (n_iter=20) with stratified K-Fold …")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=xgb.XGBClassifier(**base_params),
        param_distributions=_XGB_PARAMS_SPACE,
        n_iter=20,
        scoring="roc_auc",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    logger.info("Best params: %s  (train AUC=%0.4f)", search.best_params_, search.best_score_)

    raw_model: xgb.XGBClassifier = search.best_estimator_

    # ── 4. Validation‑set metrics ────────────────────────────────────────
    y_val_prob = raw_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_prob)
    logger.info("Validation AUC (raw): %0.4f", val_auc)

    # ── 5. Retrain on full training + validation data ────────────────────
    logger.info("Retraining best estimator on full train+val …")
    final_model = xgb.XGBClassifier(**search.best_params_, **_XGB_BASE_PARAMS)
    final_model.fit(X_train, y_train)

    # ── 6. Calibrate on the validation fold ───────────────────────────────
    logger.info("Calibrating with isotonic regression …")
    calibrated = CalibratedClassifierCV(
        estimator=final_model,
        method="isotonic",
        cv="prefit",  # XGBoost booster is already fitted.
    )
    calibrated.fit(X_val, y_val)

    # ── 7. Final metrics on validation fold (using calibrated model) ─────
    y_cal_prob = calibrated.predict_proba(X_val)[:, 1]
    cal_auc = roc_auc_score(y_val, y_cal_prob)
    brier = brier_score_loss(y_val, y_cal_prob)
    logger.info("Calibrated — AUC: %0.4f, Brier: %.4f", cal_auc, brier)

    # ── 8. Feature names (for downstream SHAP & dashboard) ────────────────
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
    elif feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # ── 9. Build metadata & persist ───────────────────────────────────────
    X_train_full = np.vstack([X_train, X_val])  # full training data
    y_train_full = np.concatenate([y_train, y_val])
    train_hash = _data_hash(X_train_full)

    metadata: dict[str, Any] = {
        "model_type": "XGBClassifier + CalibratedClassifierCV(isotonic)",
        "best_params": search.best_params_,
        "n_estimators": _XGB_BASE_PARAMS["n_estimators"],
        "scale_pos_weight": scale_pos_weight,
        "train_auc_raw": float(search.best_score_),
        "val_auc_calibrated": float(cal_auc),
        "brier_score": float(brier),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_features": n_features,
        "feature_names": feature_names,
        "train_data_hash": train_hash,
        "random_state": random_state,
    }

    persist(X, y_arr, calibrated, final_model, feature_names, metadata)

    logger.info("Saved model → %s", MODEL_PATH)
    logger.info("Saved metadata → %s", METADATA_PATH)

    return raw_model, calibrated, metadata


def predict_risk(
    X: pd.DataFrame | np.ndarray,
    *,
    model_path: Path = MODEL_PATH,
    encoder: Any = None,  # ColumnTransformer fitted on features.py
) -> np.ndarray:
    """Load a persisted risk model and return calibrated churn probabilities.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix — *must* match the schema the model was trained with.
    model_path : Path
        Path to ``risk_radar.pkl`` (default is the project default).
    encoder : ColumnTransformer | None
        If *X* is a raw DataFrame (not pre-encoded), pass the fitted
        ColumnTransformer from ``features.encode_features()`` so that
        transform parity is maintained at inference time.

    Returns
    -------
    np.ndarray
        Array of calibrated churn probabilities in ``(n_samples,)`` shape,
        values ∈ [0, 1].
    """
    loaded = load(model_path)
    cal_model: CalibratedClassifierCV = loaded["calibrated"]

    X_arr = _to_numpy(X)
    if encoder is not None:
        X_arr = encoder.transform(X_arr)

    return cal_model.predict_proba(X_arr)[:, 1]


def evaluate_risk_radar(
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    *,
    model_path: Path = MODEL_PATH,
    encoder: Any = None,
) -> dict[str, float]:
    """Evaluate the persisted model on a held‑out test set.

    Returns
    -------
    dict[str, float]
        Keys: ``"auc"``, ``"brier"`` computed on *X_test*, *y_test*.
    """
    probs = predict_risk(X_test, model_path=model_path, encoder=encoder)
    y_arr = _to_numpy(y_test, flatten=True)

    auc = roc_auc_score(y_arr, probs)
    brier = brier_score_loss(y_arr, probs)
    return {"auc": float(auc), "brier": float(brier)}


# ── Persistence ───────────────────────────────────────────────────────────────

def persist(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    calibrated: CalibratedClassifierCV,
    raw_model: xgb.XGBClassifier,
    feature_names: list[str],
    metadata: dict[str, Any],
) -> None:
    """Dump the model and associated artifacts to disk.

    Side effects: writes ``MODEL_PATH`` (joblib), ``METADATA_PATH`` (JSON).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute a hash of the training data for reproducibility audit.
    X_arr = _to_numpy(X)
    data_hash = _data_hash(X_arr)
    metadata["train_data_hash"] = data_hash

    # Persist calibrated model (this includes the fitted booster).
    dump({"calibrated": calibrated, "feature_names": feature_names}, MODEL_PATH)

    # Persist raw (un-calibrated) XGBClassifier for SHAP — TreeExplainer requires
    # the sklearn wrapper (not the raw Booster) with XGBoost ≥ 3.x / SHAP ≥ 0.48.
    dump({"model": raw_model}, ARTIFACT_DIR / "risk_radar_raw.joblib")

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)


# ── Private helpers ───────────────────────────────────────────────────────────

def _to_numpy(obj: pd.DataFrame | np.ndarray | pd.Series, flatten: bool = False) -> np.ndarray:
    """Convert DataFrame / Series to numpy ndarray."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        arr = obj.values
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        arr = np.asarray(obj)
    return arr.flatten() if flatten else arr


def _check_train_test_split(X: np.ndarray, y: np.ndarray, test_size: float) -> None:
    """Verify that both train and test folds have at least one sample per class."""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        raise ValueError(
            "Stratified split would produce a fold missing one class. "
            "Reduce test_size or ensure the dataset has enough samples per class."
        )


def _data_hash(X: np.ndarray) -> str:
    """Return an MD5 hex digest of the training data (first 1024 rows flattened)."""
    sampler = X[: min(len(X), 1024)].ravel()
    return hashlib.md5(sampler.view(np.uint8)).hexdigest()[:16]
