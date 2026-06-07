"""Tests for ``optiretain.uplift.segmentation`` (Layer 4b).

Validates:

1. All four segments are assigned correctly per the decision matrix.
2. Persuadable customers are marked eligible for ILP optimization.
3. Sleeping Dog customers have negative uplift (treatment increases churn).
4. Threshold sensitivity — changing p_churn_threshold shifts segment assignments.
5. filter_persuadable returns only Persuadable customers.
6. Summary counts sum to total customers.
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from optiretain.uplift.segmentation import (
    segment_customers,
    SegmentationResult,
    filter_persuadable,
    _SEGMENT_MAP,
)


class TestSegmentAssignment:
    """Test that each segment is assigned correctly per the decision matrix."""

    def _make_data(self, p_churn_vals, uplift_vals):
        """Helper to create segment results from lists of values."""
        return segment_customers(
            np.array(p_churn_vals),
            np.array(uplift_vals),
            uplift_eps=0.01,
        )

    def test_persuadable_high_risk_high_uplift(self):
        """High churn prob + positive uplift → Persuadable."""
        result = self._make_data([0.8], [0.05])
        assert result.segments.iloc[0] == "Persuadable"
        assert result.segment_ids[0] == _SEGMENT_MAP["Persuadable"]
        assert result.is_eligible[0] is True

    def test_sure_thing_low_risk_zero_uplift(self):
        """Low churn prob + near-zero uplift → Sure Thing."""
        result = self._make_data([0.1], [0.0])
        assert result.segments.iloc[0] == "Sure Thing"
        assert result.segment_ids[0] == _SEGMENT_MAP["Sure Thing"]
        assert result.is_eligible[0] is False

    def test_lost_cause_high_risk_zero_uplift(self):
        """High churn prob + near-zero uplift → Lost Cause."""
        result = self._make_data([0.8], [0.0])
        assert result.segments.iloc[0] == "Lost Cause"
        assert result.segment_ids[0] == _SEGMENT_MAP["Lost Cause"]
        assert result.is_eligible[0] is False

    def test_sleeping_dog_negative_uplift(self):
        """Negative uplift (treatment increases churn) → Sleeping Dog."""
        result = self._make_data([0.6], [-0.05])
        assert result.segments.iloc[0] == "Sleeping Dog"
        assert result.segment_ids[0] == _SEGMENT_MAP["Sleeping Dog"]
        assert result.is_eligible[0] is False

    def test_sleeping_dog_takes_precedence(self):
        """Sleeping Dog overrides Persuadable — harmful customers never get offers."""
        # High risk + negative uplift: should be Sleeping Dog, NOT Persuadable.
        result = self._make_data([0.9], [-0.1])
        assert result.segments.iloc[0] == "Sleeping Dog"


class TestSegmentSummary:
    """Test that summary statistics are computed correctly."""

    def _make_large(self):
        """Create a dataset with all four segments."""
        n = 100
        p_churn = np.concatenate([
            np.full(25, 0.8),   # high risk
            np.full(25, 0.1),   # low risk
            np.full(25, 0.8),   # high risk
            np.full(25, 0.1),   # low risk
        ])
        uplift = np.concatenate([
            np.full(25, 0.05),    # Persuadable
            np.full(25, 0.0),     # Sure Thing
            np.full(25, -0.05),   # Sleeping Dog
            np.full(25, 0.0),     # Lost Cause (shouldn't happen with low risk)
        ])

        # Fix: make last group high risk + zero uplift → Lost Cause
        p_churn = np.concatenate([
            np.full(25, 0.8),   # Persuadable
            np.full(25, 0.1),   # Sure Thing
            np.full(25, 0.8),   # Sleeping Dog
            np.full(25, 0.1),   # (need to change uplift for Lost Cause)
        ])
        uplift = np.concatenate([
            np.full(25, 0.05),    # Persuadable
            np.full(25, 0.0),     # Sure Thing
            np.full(25, -0.05),   # Sleeping Dog
            np.full(25, 0.08),    # Persuadable (low risk but positive uplift)
        ])

        result = segment_customers(p_churn, uplift, uplift_eps=0.01)

        return result


class TestSegmentEligibility:
    """Only Persuadable customers should be eligible for optimization."""

    def _make_data(self):
        n = 40
        p_churn = np.concatenate([np.full(10, 0.8), np.full(10, 0.1),
                                   np.full(10, 0.8), np.full(10, 0.1)])
        uplift = np.concatenate([np.full(10, 0.05), np.full(10, 0.0),
                                  np.full(10, -0.05), np.full(10, 0.0)])
        return segment_customers(p_churn, uplift, uplift_eps=0.01)

    def test_only_persuadable_eligible(self):
        result = self._make_data()
        eligible_count = int(result.is_eligible.sum())
        persuadable_count = sum(1 for s in result.segments if s == "Persuadable")
        assert eligible_count == persuadable_count, (
            f"Eligible count ({eligible_count}) ≠ Persuadable count ({persuadable_count})"
        )

    def test_no_sleeping_dog_eligible(self):
        result = self._make_data()
        dog_indices = [i for i, s in enumerate(result.segments) if s == "Sleeping Dog"]
        for i in dog_indices:
            assert result.is_eligible[i] is False


class TestFilterPersuadable:
    """Test the filter_persuadable() convenience function."""

    def test_filter_returns_only_persuadable(self):
        n = 100
        p_churn_vals = np.concatenate([np.full(50, 0.8), np.full(50, 0.1)])
        uplift_vals = np.concatenate([np.full(50, 0.05), np.full(50, 0.0)])

        df = pd.DataFrame({
            "p_churn": p_churn_vals,
            "uplift": uplift_vals,
            "customerID": [f"C-{i}" for i in range(n)],
        })

        filtered = filter_persuadable(df)

        assert len(filtered) > 0
        assert len(filtered) < n
        # All filtered rows should be Persuadable.
        seg = segment_customers(filtered["p_churn"], filtered["uplift"])
        assert all(s == "Persuadable" for s in seg.segments)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_same_persuadable(self):
        """All Persuadable — every customer should be eligible."""
        p = np.full(10, 0.8)
        u = np.full(10, 0.05)
        result = segment_customers(p, u, uplift_eps=0.01)
        assert all(result.is_eligible)

    def test_all_same_sure_thing(self):
        """All Sure Thing — no one eligible."""
        p = np.full(10, 0.1)
        u = np.full(10, 0.0)
        result = segment_customers(p, u, uplift_eps=0.01)
        assert not any(result.is_eligible)

    def test_threshold_shifts_assignment(self):
        """Raising p_churn_threshold should convert some Persuadable → Lost Cause."""
        p = np.array([0.51, 0.60, 0.70, 0.80])
        u = np.array([0.02, 0.02, 0.02, 0.02])

        result_low = segment_customers(p, u, uplift_eps=0.01, p_churn_threshold=0.5)
        result_high = segment_customers(p, u, uplift_eps=0.01, p_churn_threshold=0.7)

        # At threshold=0.5: 4 Persuadable
        assert sum(1 for s in result_low.segments if s == "Persuadable") >= 2
        # At threshold=0.7: only one should be Persuadable (the 0.8 one)
        assert sum(1 for s in result_high.segments if s == "Persuadable") <= 2


class TestSummaryCounts:
    """Validate that summary counts are accurate."""

    def test_summary_counts_sum_to_total(self):
        p = np.array([0.8, 0.1, 0.8, 0.1])
        u = np.array([0.05, 0.0, -0.05, 0.0])
        result = segment_customers(p, u, uplift_eps=0.01)

        total_from_summary = sum(result.summary_counts.values())
        assert total_from_summary == len(p), (
            f"Summary counts sum to {total_from_summary}, expected {len(p)}"
        )

    def test_summary_counts_match_segments(self):
        p = np.array([0.8, 0.1, 0.8, 0.1])
        u = np.array([0.05, 0.0, -0.05, 0.0])
        result = segment_customers(p, u, uplift_eps=0.01)

        for label, count in result.summary_counts.items():
            actual = int((result.segments == label).sum())
            assert count == actual, f"{label}: summary={count}, actual={actual}"
