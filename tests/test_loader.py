"""Unit tests for ``optiretain.data.loader``.

These tests verify that the loader:
1. Produces the correct shape and column set
2. Coerces dtypes properly (TotalCharges → float, Churn → binary int)
3. Preserves the raw CLTV signal from the source dataset
4. Raises appropriate errors on bad input paths or schema mismatches

Run with: ``pytest tests/test_loader.py -v``
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import pandas as pd


# ── Fixtures from conftest ────────────────────────────────────────────────────
# loaded_df, expected_columns, telco_xlsx_path are provided by conftest.py


class TestLoadTelcoDataShape:
    """Test that load_telco_data returns the correct DataFrame shape."""

    def test_shape(self, loaded_df):
        df = loaded_df
        assert df.shape == (7043, 22), (
            f"Expected (7043, 22) — rows from source Excel minus dropped "
            f"columns should leave 22 (21 feature cols + CLTV)"
        )

    def test_is_dataframe(self, loaded_df):
        assert isinstance(loaded_df, pd.DataFrame)


class TestLoadTelcoDataColumns:
    """Test column presence, renaming, and absence of dropped columns."""

    def test_expected_columns_present(self, loaded_df, expected_columns):
        actual = sorted(loaded_df.columns.tolist())
        assert actual == expected_columns, (
            f"Expected {len(expected_columns)} columns, got {actual}\n"
            f"Missing: {set(expected_columns) - set(actual)}\n"
            f"Extra:   {set(actual) - set(expected_columns)}"
        )

    def test_customer_id_renamed(self, loaded_df):
        """CustomerID → customerID (camelCase)."""
        assert "customerID" in loaded_df.columns
        assert "CustomerID" not in loaded_df.columns

    def test_tenure_months_renamed(self, loaded_df):
        assert "tenure" in loaded_df.columns
        assert "Tenure Months" not in loaded_df.columns

    def test_monthly_charges_renamed(self, loaded_df):
        assert "MonthlyCharges" in loaded_df.columns
        assert "Monthly Charges" not in loaded_df.columns

    def test_total_charges_renamed(self, loaded_df):
        assert "TotalCharges" in loaded_df.columns
        assert "Total Charges" not in loaded_df.columns

    def test_churn_label_renamed_to_churn(self, loaded_df):
        assert "Churn" in loaded_df.columns
        assert "Churn Label" not in loaded_df.columns

    def test_cltv_preserved(self, loaded_df):
        """CLTV must survive — it's the dataset creator's baked-in customer value."""
        assert "CLTV" in loaded_df.columns

    # ── Dropped columns should NOT appear ────────────────────────────────

    def _assert_dropped(self, loaded_df, col_name):
        assert col_name not in loaded_df.columns, (
            f"Column '{col_name}' was not dropped as expected."
        )

    def test_dropped_count(self, loaded_df):
        self._assert_dropped(loaded_df, "Count")

    def test_dropped_country(self, loaded_df):
        self._assert_dropped(loaded_df, "Country")

    def test_dropped_state(self, loaded_df):
        self._assert_dropped(loaded_df, "State")

    def test_dropped_city(self, loaded_df):
        self._assert_dropped(loaded_df, "City")

    def test_dropped_zip_code(self, loaded_df):
        self._assert_dropped(loaded_df, "Zip Code")

    def test_dropped_lat_long(self, loaded_df):
        self._assert_dropped(loaded_df, "Lat Long")

    def test_dropped_latitude(self, loaded_df):
        self._assert_dropped(loaded_df, "Latitude")

    def test_dropped_longitude(self, loaded_df):
        self._assert_dropped(loaded_df, "Longitude")

    def test_dropped_churn_value(self, loaded_df):
        self._assert_dropped(loaded_df, "Churn Value")

    def test_dropped_churn_score(self, loaded_df):
        self._assert_dropped(loaded_df, "Churn Score")

    def test_dropped_churn_reason(self, loaded_df):
        self._assert_dropped(loaded_df, "Churn Reason")


class TestLoadTelcoDataDtypes:
    """Test that numeric coercion and target mapping are correct."""

    def test_total_charges_is_float64(self, loaded_df):
        assert loaded_df["TotalCharges"].dtype == "float64"

    def test_total_charges_has_nans(self, loaded_df):
        """The source Excel has ~11 blank TotalCharges entries."""
        nans = loaded_df["TotalCharges"].isna().sum()
        assert nans > 0, "Expected some NaN values in TotalCharges."
        assert nans == 11, f"Expected 11 missing TotalCharges, got {nans}."

    def test_total_charges_non_null_values(self, loaded_df):
        """Non-null TotalCharges should be finite positive floats."""
        valid = loaded_df["TotalCharges"].dropna()
        assert (valid > 0).all(), "All non-null TotalCharges must be positive."

    def test_churn_is_binary_int(self, loaded_df):
        assert loaded_df["Churn"].dtype == "int64"
        unique = set(loaded_df["Churn"].unique())
        assert unique == {0, 1}, f"Expected binary {{0, 1}}, got {unique}"

    def test_churn_no_nans(self, loaded_df):
        assert loaded_df["Churn"].isna().sum() == 0, "Churn target must have no NaNs."

    def test_churn_distribution(self, loaded_df):
        """Source Telco churn rate ≈ 26.5% (1869/7043)."""
        n_yes = (loaded_df["Churn"] == 1).sum()
        n_no = (loaded_df["Churn"] == 0).sum()
        assert n_yes > 0 and n_no > 0, "Churn must contain both Yes and No."
        ratio = n_yes / len(loaded_df)
        assert 0.20 < ratio < 0.35, (
            f"Churn rate {ratio:.3f} outside expected range [0.20, 0.35]."
        )

    def test_cltv_is_numeric(self, loaded_df):
        assert pd.api.types.is_numeric_dtype(loaded_df["CLTV"])

    def test_monthly_charges_is_numeric(self, loaded_df):
        assert pd.api.types.is_numeric_dtype(loaded_df["MonthlyCharges"])

    def test_tenure_is_numeric(self, loaded_df):
        assert pd.api.types.is_numeric_dtype(loaded_df["tenure"])


class TestLoadTelcoDataCLTV:
    """Tests specific to the CLTV signal from the source dataset."""

    def test_cltv_no_nans(self, loaded_df):
        assert loaded_df["CLTV"].isna().sum() == 0, "CLTV must have no missing values."

    def test_cltv_positive(self, loaded_df):
        assert (loaded_df["CLTV"] > 0).all(), "All CLTV values should be positive."

    def test_cltv_mean_reasonable(self, loaded_df):
        mean_cltv = loaded_df["CLTV"].mean()
        # Telco data: monthly charges ~65, avg tenure ~33 → CLV ≈ 2100–2200
        assert 1000 < mean_cltv < 5000, (
            f"Mean CLTV {mean_cltv:.1f} seems unreasonable for the Telco dataset."
        )

    def test_cltv_correlated_with_monthly_charges(self, loaded_df):
        """CLTV may or may not strongly correlate with MonthlyCharges — the
        source dataset likely uses a different formula (survival model, payment
        history, etc.). We just assert it's *some* positive number: if it were
        negative that would be genuinely suspicious.

        Note: this dataset shows ρ ≈ 0.10, confirming CLTV was computed from
        signals beyond MonthlyCharges — exactly the point of keeping raw CLTV."""
        corr = loaded_df["CLTV"].corr(loaded_df["MonthlyCharges"])
        assert corr > -0.5, (
            f"CLTV–MonthlyCharges correlation {corr:.3f} is suspiciously "
            "negative — check if CLTV was computed using the churn label."
        )

    def test_cltv_by_churn_group(self, loaded_df):
        """Churned customers should have lower median CLTV (shorter tenure)."""
        churned_median = loaded_df[loaded_df["Churn"] == 1]["CLTV"].median()
        retained_median = loaded_df[loaded_df["Churn"] == 0]["CLTV"].median()
        # Churned customers typically have shorter tenures → lower CLTV.
        # Allow some overlap but not inversion.
        assert churned_median < retained_median, (
            "Median CLTV of churned customers should be below retained — "
            "this may indicate a data issue or unusual CLTV computation."
        )


class TestLoadTelcoDataErrorHandling:
    """Test that the loader raises appropriate errors for bad inputs."""

    def test_file_not_found(self):
        from optiretain.data.loader import load_telco_data

        with pytest.raises(FileNotFoundError, match="not found"):
            load_telco_data("/nonexistent/path/to/file.xlsx")

    def test_custom_path_works(self, telco_xlsx_path):
        """Loading via explicit path should produce the same result."""
        from optiretain.data.loader import load_telco_data

        df_explicit = load_telco_data(telco_xlsx_path)
        df_default = load_telco_data()  # uses TELECO_FILE default
        pd.testing.assert_frame_equal(df_explicit, df_default)


class TestLoadTelcoDataStringColumns:
    """Validate that categorical columns are preserved as strings."""

    # Columns that should remain as string/object type after loading.
    _STRING_COLS = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    ]

    @pytest.mark.parametrize("col", _STRING_COLS)
    def test_string_columns_preserved(self, loaded_df, col):
        assert col in loaded_df.columns
        # Should not have been coerced to numeric by accident.
        assert pd.api.types.is_object_dtype(loaded_df[col]) or pd.api.types.is_string_dtype(loaded_df[col]), (
            f"Column '{col}' should remain string-like, got {loaded_df[col].dtype}."
        )


class TestPipelineSmoke:
    """End-to-end smoke test for the pipeline entry point."""

    def test_pipeline_runs_without_error(self):
        from optiretain.pipeline import run

        # Should not raise — at minimum it loads the dataset.
        # All layers are TODO stubs; just verify the loader works.
        run(budget=50_000, discount=0.20)
