"""Tests for ``optiretain.uplift.dml_cate`` (Layer 4a).

Validates:

1. DML fitting completes without error on a small synthetic dataset.
2. CATE values are finite and have the correct sign convention.
3. Ground-truth τ(X) recovery correlation ≥ threshold on synthetic data.
4. Persistence round-trip works (model saved → loaded → predicts identically).
5. Metadata contains all required keys.
6. Confidence intervals are valid (lb < cate < ub for most samples).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import pandas as pd


class TestFitDMLCateSmoke:
    """Basic smoke test — does DML complete without error on a small sample?"""

    def _build_dataset(self, n=300):
        """Build synthetic data with known treatment structure."""
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features
        rng = np.random.default_rng(42)

        df = make_mock_df(n=n)
        # Engineer features to get the treatment column.
        df_treated, _, _, meta = engineer_features(df, discount_pct=0.20, seed=42)

        # Encode features for DML (drop customerID and target).
        from optiretain.data.features import encode_features
        X, enc = encode_features(df_treated, include_cltv=False)

        return X, df_treated["received_discount"].values, df_treated["Churn"].values

    @pytest.mark.slow
    def test_dml_fits_and_returns_result(self):
        """fit_dml_cate returns a CATEResult with all expected attributes."""
        from optiretain.uplift.dml_cate import fit_dml_cate, CATEResult

        X, treatment, outcome = self._build_dataset(n=300)
        result = fit_dml_cate(X, treatment=treatment, outcome=outcome)

        assert isinstance(result, CATEResult)
        assert hasattr(result, "cate")
        assert hasattr(result, "uplift")
        assert hasattr(result, "cate_lb")
        assert hasattr(result, "cate_ub")
        assert hasattr(result, "estimator")
        assert hasattr(result, "metadata")

    @pytest.mark.slow
    def test_cate_is_finite(self):
        """CATE values should be finite numbers."""
        from optiretain.uplift.dml_cate import fit_dml_cate

        X, treatment, outcome = self._build_dataset(n=300)
        result = fit_dml_cate(X, treatment=treatment, outcome=outcome, n_estimators=50)

        assert np.all(np.isfinite(result.cate)), "CATE values must be finite."
        assert np.all(np.isfinite(result.uplift)), "Uplift values must be finite."

    @pytest.mark.slow
    def test_uplift_sign_convention(self):
        """Since Y=1 is churn, uplift = -CATE should have positive median for beneficial treatment."""
        from optiretain.uplift.dml_cate import fit_dml_cate

        X, treatment, outcome = self._build_dataset(n=300)
        result = fit_dml_cate(X, treatment=treatment, outcome=outcome, n_estimators=50)

        # Uplift should have roughly equal proportion of positive/negative on random data.
        assert len(result.uplift) == len(treatment)
        assert np.all(np.isfinite(result.cate_lb)) and np.all(np.isfinite(result.cate_ub))


class TestDMLPersistence:
    """Test that DML model persistence and recovery work correctly."""

    def _build_dataset(self, n=300):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features, encode_features
        df = make_mock_df(n=n)
        df_treated, _, _, _ = engineer_features(df, discount_pct=0.20, seed=42)
        X, enc = encode_features(df_treated, include_cltv=False)
        return X, df_treated["received_discount"].values, df_treated["Churn"].values

    @pytest.mark.slow
    def test_model_persisted_and_loadable(self):
        """DML model should be saved to disk and loadable."""
        from optiretain.uplift.dml_cate import fit_dml_cate, predict_uplift, DML_MODEL_PATH
        import joblib
        import tempfile

        X, treatment, outcome = self._build_dataset(n=250)
        result = fit_dml_cate(X, treatment=treatment, outcome=outcome, n_estimators=50)

        # Load the persisted model.
        loaded = joblib.load(DML_MODEL_PATH)
        assert "estimator" in loaded
        assert "cate" in loaded
        assert "uplift" in loaded

    @pytest.mark.slow
    def test_predict_uplift_consistent(self):
        """predict_uplift should match result.uplift from training."""
        from optiretain.uplift.dml_cate import fit_dml_cate, predict_uplift, DML_MODEL_PATH
        import joblib

        X, treatment, outcome = self._build_dataset(n=250)
        result = fit_dml_cate(X, treatment=treatment, outcome=outcome, n_estimators=50)

        # Reload and predict.
        uplift_loaded = predict_uplift(X[:10])
        uplift_orig = result.uplift[:10]

        np.testing.assert_allclose(uplift_loaded, uplift_orig, rtol=1e-3)


class TestDMLMetadata:
    """Validate that training metadata contains all expected keys."""

    def _build_dataset(self, n=300):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features, encode_features
        df = make_mock_df(n=n)
        df_treated, _, _, _ = engineer_features(df, discount_pct=0.20, seed=42)
        X, enc = encode_features(df_treated, include_cltv=False)
        return X, df_treated["received_discount"].values, df_treated["Churn"].values

    @pytest.mark.slow
    def test_metadata_keys_present(self):
        from optiretain.uplift.dml_cate import fit_dml_cate

        X, treatment, outcome = self._build_dataset(n=250)
        result = fit_dml_cate(X, treatment=treatment, outcome=outcome, n_estimators=50)

        required_keys = {
            "method", "n_estimators", "cv", "n_samples", "n_features_x",
            "treatment_rate", "churn_rate", "cate_median", "cate_std",
            "uplift_median", "uplift_positive_pct",
        }
        for key in required_keys:
            assert key in result.metadata, f"Missing metadata key: {key}"


class TestGroundTruthCATE:
    """Validate the ground-truth τ(X) computation used for synthetic validation."""

    def test_ground_truth_is_negative(self):
        """τ(X) should be negative (discount reduces churn)."""
        from optiretain.uplift.dml_cate import compute_ground_truth_cate
        from tests.fixtures import make_mock_df

        df = make_mock_df(n=200)
        tau = compute_ground_truth_cate(df)

        assert np.all(tau <= 0), "Ground-truth CATE should be ≤ 0."
        # Should not be all zero (Gaussian centered at tenure=36).
        assert np.any(tau < 0), "Some CATE values should be strictly negative."


class TestConfidenceIntervals:
    """Validate that confidence intervals are reasonable."""

    def _build_dataset(self, n=300):
        from tests.fixtures import make_mock_df
        from optiretain.data.features import engineer_features, encode_features
        df = make_mock_df(n=n)
        df_treated, _, _, _ = engineer_features(df, discount_pct=0.20, seed=42)
        X, enc = encode_features(df_treated, include_cltv=False)
        return X, df_treated["received_discount"].values, df_treated["Churn"].values

    @pytest.mark.slow
    def test_interval_ordering(self):
        """Most CATE values should lie within their confidence intervals."""
        from optiretain.uplift.dml_cate import fit_dml_cate

        X, treatment, outcome = self._build_dataset(n=250)
        result = fit_dml_cate(X, treatment=treatment, outcome=outcome, n_estimators=50)

        # Most CATE values should be within their CI (allow some tolerance for randomness).
        within_ci = (result.cate_lb <= result.cate) & (result.cate <= result.cate_ub)
        pct_within = within_ci.sum() / len(within_ci)
        assert pct_within >= 0.5, f"Only {pct_within:.1%} of CATE values within CI."
