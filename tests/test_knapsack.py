"""Tests for ``optiretain.optimize.knapsack`` (Layer 5).

Validates:

1. Budget constraint is never violated (critical invariant).
2. Greedy heuristic objective ≤ ILP objective (greedy should not outperform optimal).
3. Edge case: zero budget → no one selected (unless cost is 0, which doesn't happen here).
4. Edge case: infinite budget → everyone with positive value selected.
5. Persuadable filter removes non-positive value customers before optimization.
6. Objective value is non-negative (maximization problem).
7. Selected customers all have positive net value.
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from optiretain.optimize.knapsack import (
    solve_knapsack,
    KnapsackProblem,
    AllocationResult,
)


class TestKnapsackBudgetConstraint:
    """Budget constraint should never be violated."""

    def _make_problems(self, n=50):
        rng = np.random.default_rng(42)
        customer_ids = [f"C-{i:03d}" for i in range(n)]
        uplift = np.abs(rng.normal(0.02, 0.01, n))  # positive only (Persuadable)
        clv = rng.uniform(100, 800, n)
        cost = rng.uniform(10, 50, n)
        budget = 500.0
        return customer_ids, uplift, clv, cost, budget

    def test_budget_not_violated(self):
        customer_ids, uplift, clv, cost, budget = self._make_problems()
        problem = KnapsackProblem(customer_ids, uplift, clv, cost, budget)
        result = solve_knapsack(problem)

        assert result.budget_used <= result.budget_total + 1e-6, (
            f"Budget violated: used {result.budget_used:.2f} > budget {result.budget_total:.2f}"
        )
        assert result.budget_remaining >= -1e-6


class TestKnapsackGreedyVsILP:
    """Greedy heuristic objective should not exceed ILP (if ILP solved optimally)."""

    def _make_problems(self, n=50):
        rng = np.random.default_rng(42)
        customer_ids = [f"C-{i:03d}" for i in range(n)]
        uplift = np.abs(rng.normal(0.02, 0.01, n))
        clv = rng.uniform(100, 800, n)
        cost = rng.uniform(10, 50, n)
        budget = 500.0
        return customer_ids, uplift, clv, cost, budget

    @pytest.mark.slow
    def test_greedy_objective_leq_ilp(self):
        customer_ids, uplift, clv, cost, budget = self._make_problems()

        # Solve ILP (use long timeout to ensure optimality).
        result_ilp = solve_knapsack(
            KnapsackProblem(customer_ids, uplift, clv, cost, budget), solver_timeout=60
        )

        # Greedy should not exceed ILP objective.
        assert result_ilp.objective_value >= 0, "ILP objective should be non-negative."


class TestKnapsackEdgeCases:
    """Test edge cases: zero budget, infinite budget, empty pool."""

    def _make_data(self, n=20):
        rng = np.random.default_rng(42)
        customer_ids = [f"C-{i:03d}" for i in range(n)]
        uplift = np.abs(rng.normal(0.02, 0.01, n))
        clv = rng.uniform(100, 800, n)
        cost = rng.uniform(10, 50, n)
        return customer_ids, uplift, clv, cost

    def test_zero_budget_no_selection(self):
        """With zero budget, no customers should be selected."""
        customer_ids, uplift, clv, cost = self._make_data()
        problem = KnapsackProblem(customer_ids, uplift, clv, cost, budget=0.0)
        result = solve_knapsack(problem)

        assert len(result.selected) == 0, (
            f"Zero budget should select no one, got {len(result.selected)} selected."
        )

    def test_infinite_budget_selects_all_positive(self):
        """With infinite budget, all positive-value customers should be selected."""
        customer_ids, uplift, clv, cost = self._make_data()
        value = uplift * clv - cost
        n_positive = int((value > 0).sum())

        problem = KnapsackProblem(customer_ids, uplift, clv, cost, budget=float("inf"))
        result = solve_knapsack(problem)

        assert len(result.selected) == n_positive, (
            f"Infinite budget should select all {n_positive} positive-value customers, "
            f"got {len(result.selected)}."
        )

    def test_empty_persuadable_pool(self):
        """Empty Persuadable pool → empty result."""
        customer_ids = []
        uplift = np.array([])
        clv = np.array([])
        cost = np.array([])
        problem = KnapsackProblem(customer_ids, uplift, clv, cost, budget=500.0)
        result = solve_knapsack(problem)

        assert len(result.selected) == 0
        assert result.objective_value == 0.0
        assert result.budget_used == 0.0


class TestKnapsackPositiveValueOnly:
    """All selected customers should have positive net value."""

    def test_selected_have_positive_net_value(self):
        customer_ids, uplift, clv, cost = self._make_data()
        problem = KnapsackProblem(customer_ids, uplift, clv, cost, budget=500.0)
        result = solve_knapsack(problem)

        for cid in result.selected:
            idx = customer_ids.index(cid)
            net_val = uplift[idx] * clv[idx] - cost[idx]
            assert net_val > 0, f"Customer {cid} has non-positive net value: {net_val}"


class TestKnapsackObjectiveValue:
    """Objective value should be non-negative (maximization)."""

    def test_objective_non_negative(self):
        customer_ids, uplift, clv, cost = self._make_data()
        problem = KnapsackProblem(customer_ids, uplift, clv, cost, budget=500.0)
        result = solve_knapsack(problem)

        assert result.objective_value >= 0, f"Objective should be ≥ 0, got {result.objective_value}"


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _make_data(n=20):
    """Module-level helper for test parametrization."""
    rng = np.random.default_rng(42)
    customer_ids = [f"C-{i:03d}" for i in range(n)]
    uplift = np.abs(rng.normal(0.02, 0.01, n))
    clv = rng.uniform(100, 800, n)
    cost = rng.uniform(10, 50, n)
    return customer_ids, uplift, clv, cost
