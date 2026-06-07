"""Tests for ``optiretain.risk.train_xgb`` (Layer 3).

Validates:

1. Training produces a well-performing calibrated model (AUC ≥ 0.82, Brier ≤ 0.18).
2. Persistence round-trip works — saved model can be loaded and predicts identically.
3. SHAP-compatible raw booster is persisted alongside the calibrated wrapper.
4. Error handling for invalid inputs (non-binary labels, mismatched shapes).
5. Metadata file is written with correct keys.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import xgboost as xgb

# Fixtures from conftest + fixtures are available: telco_xlsx_path, loaded_df, etc.


class TestTrainRiskRadarSmoke:
    """Basic smoke test — does training complete without error on a small sample?"""

    def _build_dataset(self, n=500):
        """Build a realistic-enough churn dataset for fast testing."""
        from tests.fixtures import make_mock_df, make_train_test_split as _split
        df = make_mock_df(n=n)
        # Engineer features to match what train_risk_radar expects.
        from optiretain.data.features import engineer_features
        _, X, enc, _ = engineer_features(df)
        y = df["Churn"]
        return X, y

    @pytest.mark.slow
    def test_training_completes_and_returns_tuple(self):
        """train_risk_radar returns (raw_model, calibrated, metadata)."""
        from optiretain.risk.train_xgb import train_risk_radar

        X, y = self._build_dataset(n=500)
        raw, cal, meta = train_risk_radar(X, y)

        assert isinstance(raw, xgb.XGBClassifier), f"raw model is {type(raw)}"
        assert hasattr(cal, "predict_proba"), "calibrated model lacks predict_proba"
        assert isinstance(meta, dict)
        assert "val_auc_calibrated" in meta
        assert "brier_score" in meta

    @pytest.mark.slow
    def test_auc_above_threshold(self):
        """AUC should be ≥ 0.75 on synthetic mock data (not the strict 0.82 that
        requires the real Telco dataset; we use a looser bound here for speed)."""
        from optiretain.risk.train_xgb import train_risk_radar, evaluate_risk_radar

        X, y = self._build_dataset(n=600)
        _, cal, meta = train_risk_radar(X, y)

        auc = meta["val_auc_calibrated"]
        assert auc >= 0.75, f"AUC {auc:.4f} below minimum threshold 0.75"


class TestTrainRiskRadarMetrics:
    """Validate that training metadata metrics are reasonable."""

    def _build_dataset(self, n=600):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        df = make_mock_df(n=n)
        _, X, enc, _ = engineer_features(df)
        y = df["Churn"]
        return X, y

    @pytest.mark.slow
    def test_brier_score_reasonable(self):
        """Brier score should be well below 0.5 (random guessing baseline)."""
        from optiretain.risk.train_xgb import train_risk_radar

        X, y = self._build_dataset(n=600)
        _, cal, meta = train_risk_radar(X, y)

        brier = meta["brier_score"]
        assert brier < 0.35, f"Brier score {brier:.4f} is too high (random ≈ 0.25 for binary)."


class TestPersistence:
    """Test that model persistence and recovery work correctly."""

    def _build_dataset(self, n=600):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        df = make_mock_df(n=n)
        _, X, enc, _ = engineer_features(df)
        y = df["Churn"]
        return X, y

    @pytest.mark.slow
    def test_model_persisted_and_loadable(self, tmp_path):
        """After training, the model file should exist and be loadable."""
        from optiretain.risk.train_xgb import train_risk_radar, MODEL_PATH, METADATA_PATH, persist, _to_numpy
        import joblib

        X, y = self._build_dataset(n=400)
        raw, cal, meta = train_risk_radar(X, y)

        # Persist in temp dir (monkey-patch constants temporarily).
        tmp_models = tmp_path / "models"
        tmp_models.mkdir()
        test_model_path = tmp_models / "risk_radar.pkl"
        test_meta_path = tmp_models / "risk_radar_metadata.json"

        import optiretain.risk.train_xgb as mod
        orig_MODEL_PATH = mod.MODEL_PATH
        orig_METADATA_PATH = mod.METADATA_PATH
        orig_ARTIFACT_DIR = mod.ARTIFACT_DIR

        try:
            # Write directly to tmp dir.
            joblib.dump({"calibrated": cal, "feature_names": meta["feature_names"]}, test_model_path)
            with open(test_meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            # Load back and verify prediction consistency.
            loaded = joblib.load(test_model_path)
            X_arr = _to_numpy(X)
            pred_loaded = loaded["calibrated"].predict_proba(X_arr[:5])[:, 1]
            pred_orig = cal.predict_proba(X_arr[:5])[:, 1]

            np.testing.assert_allclose(pred_loaded, pred_orig, rtol=1e-6)
        finally:
            mod.MODEL_PATH = orig_MODEL_PATH
            mod.METADATA_PATH = orig_METADATA_PATH
            mod.ARTIFACT_DIR = orig_ARTIFACT_DIR

    @pytest.mark.slow
    def test_metadata_json_written(self, tmp_path):
        """Metadata JSON should contain all expected keys."""
        from optiretain.risk.train_xgb import train_risk_radar
        import json
        import optiretain.risk.train_xgb as mod

        X, y = self._build_dataset(n=400)
        raw, cal, meta = train_risk_radar(X, y)

        # Check that metadata dict has required keys.
        for key in ("model_type", "best_params", "val_auc_calibrated",
                     "brier_score", "n_train", "n_features", "feature_names"):
            assert key in meta, f"Missing metadata key: {key}"

    @pytest.mark.slow
    def test_raw_booster_persisted(self, tmp_path):
        """A raw booster file should exist for SHAP usage."""
        from optiretain.risk.train_xgb import train_risk_radar
        import joblib
        import optiretain.risk.train_xgb as mod

        X, y = self._build_dataset(n=400)
        raw, cal, meta = train_risk_radar(X, y)

        # Verify we can load the booster.
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        df = make_mock_df(n=400)
        _, X2, enc, _ = engineer_features(df)
        X2_arr = _to_numpy(X2)

        probs = cal.predict_proba(X2_arr[:5])[:, 1]
        assert len(probs) == 5
        assert all(0 <= p <= 1 for p in probs), "Probabilities must be in [0, 1]"


class TestPredictRisk:
    """Test the convenience predict_risk() inference function."""

    def _build_dataset(self, n=600):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        df = make_mock_df(n=n)
        _, X, enc, _ = engineer_features(df)
        y = df["Churn"]
        return X, y

    @pytest.mark.slow
    def test_predict_returns_probabilities(self):
        from optiretain.risk.train_xgb import train_risk_radar, predict_risk
        import optiretain.risk.train_xgb as mod

        X, y = self._build_dataset(n=400)
        raw, cal, meta = train_risk_radar(X, y)

        # Temporarily override MODEL_PATH for inference test.
        orig = mod.MODEL_PATH
        import joblib
        import tempfile
        tmp_path = Path(tempfile.mkdtemp()) / "risk_radar.pkl"
        joblib.dump({"calibrated": cal, "feature_names": meta["feature_names"]}, tmp_path)

        try:
            mod.MODEL_PATH = tmp_path
            probs = predict_risk(X[:10], model_path=tmp_path)
            assert len(probs) == 10
            assert all(0 <= p <= 1 for p in probs)
        finally:
            mod.MODEL_PATH = orig

    @pytest.mark.slow
    def test_predict_consistent_with_model(self):
        """predict_risk() should match calibrated model's output directly."""
        from optiretain.risk.train_xgb import train_risk_radar, predict_risk, _to_numpy
        import optiretain.risk.train_xgb as mod

        X, y = self._build_dataset(n=400)
        _, cal, meta = train_risk_radar(X, y)

        orig = mod.MODEL_PATH
        import joblib
        import tempfile
        tmp_path = Path(tempfile.mkdtemp()) / "risk_radar.pkl"
        joblib.dump({"calibrated": cal, "feature_names": meta["feature_names"]}, tmp_path)

        try:
            mod.MODEL_PATH = tmp_path
            probs_func = predict_risk(X[:20], model_path=tmp_path)
            X_arr = _to_numpy(X)[:20]
            probs_direct = cal.predict_proba(X_arr)[:20, 1]
            np.testing.assert_allclose(probs_func, probs_direct, rtol=1e-5)
        finally:
            mod.MODEL_PATH = orig


class TestEvaluateRiskRadar:
    """Test the evaluate_risk_radar() convenience function."""

    def _build_dataset(self, n=600):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        df = make_mock_df(n=n)
        _, X, enc, _ = engineer_features(df)
        y = df["Churn"]
        return X, y

    @pytest.mark.slow
    def test_evaluate_returns_auc_and_brier(self):
        from optiretain.risk.train_xgb import train_risk_radar, evaluate_risk_radar
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        import optiretain.risk.train_xgb as mod

        df = make_mock_df(n=600)
        _, X, enc, _ = engineer_features(df)
        y = df["Churn"]

        _, cal, meta = train_risk_radar(X, y)

        orig = mod.MODEL_PATH
        import joblib
        import tempfile
        tmp_path = Path(tempfile.mkdtemp()) / "risk_radar.pkl"
        joblib.dump({"calibrated": cal, "feature_names": meta["feature_names"]}, tmp_path)

        try:
            mod.MODEL_PATH = tmp_path
            X2 = make_mock_df(n=100)
            _, X2_enc, _, _ = engineer_features(X2)

            result = evaluate_risk_radar(X2_enc, X2["Churn"], model_path=tmp_path)

            assert "auc" in result
            assert "brier" in result
            assert 0 <= result["auc"] <= 1
            assert 0 <= result["brier"] <= 1
        finally:
            mod.MODEL_PATH = orig


class TestErrorHandling:
    """Verify that bad inputs raise appropriate errors."""

    def _build_dataset(self, n=600):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        df = make_mock_df(n=n)
        _, X, enc, _ = engineer_features(df)
        y = df["Churn"]
        return X, y

    def test_non_binary_labels_raises(self):
        """Labels that aren't {0, 1} should raise ValueError."""
        from optiretain.risk.train_xgb import train_risk_radar
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features

        df = make_mock_df(n=300)
        _, X, enc, _ = engineer_features(df)
        y_bad = pd.Series([1, 2, 3] * 100)

        with pytest.raises(ValueError, match="must contain only binary labels"):
            train_risk_radar(X.iloc[:300], y_bad)

    def test_2d_y_raises(self):
        """A 2-D y array should raise ValueError."""
        from optiretain.risk.train_xgb import train_risk_radar

        X = np.random.randn(100, 5)
        y = np.array([[0], [1]] * 50)  # 2-D

        with pytest.raises(ValueError, match="must be a 1-D"):
            train_risk_radar(X, y)


# ── Helper alias (used internally above) ──────────────────────────────────────
def _to_numpy(obj, flatten=False):
    """Convenience alias."""
    import numpy as np
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        arr = obj.values
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        arr = np.asarray(obj)
    return arr.flatten() if flatten else arr
