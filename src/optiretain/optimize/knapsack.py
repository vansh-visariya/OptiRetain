"""Layer 5 — ROI Maximizer: budget-constrained customer allocation via PuLP ILP.

Solves a **0-1 Knapsack** problem: select the optimal subset of Persuadable
customers for retention offers to maximize expected net uplift within a budget.

Formulation::

    max  Σ x_i · (uplift_i × clv_i - cost_i)    where x_i ∈ {0, 1}
    s.t. Σ x_i · cost_i ≤ budget
         x_i = 0 for Sleeping Dogs (hard filter before calling)
         x_i = 0 for customers with value_i ≤ 0  (pre-filtered)

Solved with PuLP's CBC MILP solver. For large datasets (>10k), a greedy
heuristic by value/cost ratio provides near-optimal results as fallback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pulp

from optiretain.config import MODELS_DIR

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    """Container for ILP allocation output."""
    customer_ids: list[str]               # all Persuadable customer IDs (input order)
    selected: list[str]                   # recommended for intervention
    rejected: list[str]                   # not recommended (value ≤ 0 or budget constraints)
    x_values: dict[str, int]              # {customer_id: assigned_value} (1 or 0)
    objective_value: float                # total expected net uplift
    budget_total: float                   # input budget
    budget_used: float                    # total cost of selected offers
    budget_remaining: float               # budget_unused
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class KnapsackProblem:
    """Input parameters for the knapsack problem."""
    customer_ids: list[str]
    uplift: np.ndarray                    # positive = beneficial (already sign-flipped from CATE)
    clv: np.ndarray                       # CLV per customer
    cost: np.ndarray                      # treatment cost per customer
    budget: float                         # total retention marketing budget


# ── Public API ───────────────────────────────────────────────────────────────────

def solve_knapsack(
    problem: KnapsackProblem,
    *,
    solver_timeout: int = 120,
    use_greedy_fallback: bool = True,
) -> AllocationResult:
    """Solve the budget-constrained customer allocation ILP.

    Parameters
    ----------
    problem : KnapsackProblem
        Named tuple with *customer_ids*, *uplift*, *clv*, *cost*, *budget*.
    solver_timeout : int
        Maximum seconds for CBC solver (default 120).
    use_greedy_fallback : bool
        If the ILP times out, fall back to greedy value/cost ratio heuristic.

    Returns
    -------
    AllocationResult
        Named tuple with selected/rejected customer IDs, objective value, budget usage,
        and optimization metrics.
    """
    uplift = np.asarray(problem.uplift).flatten()
    clv = np.asarray(problem.clv).flatten()
    cost = np.asarray(problem.cost).flatten()

    # ── 0. Pre-filter: remove customers with non-positive value ───────────
    value_per_customer = problem.uplift * clv - problem.cost  # net coefficient
    positive_mask = value_per_customer > 0

    n_before_filter = len(problem.customer_ids)
    customer_ids = [cid for cid, pos in zip(problem.customer_ids, positive_mask) if pos]
    uplift_filt = uplift[positive_mask]
    clv_filt = clv[positive_mask]
    cost_filt = cost[positive_mask]
    value_filt = value_per_customer[positive_mask]

    logger.info("Pre-filtered: %d → %d Persuadable customers with positive net value",
                n_before_filter, len(customer_ids))

    if not customer_ids:
        logger.warning("No Persuadable customers with positive net value — nothing to allocate.")
        return AllocationResult(
            customer_ids=list(problem.customer_ids),
            selected=[], rejected=list(problem.customer_ids),
            x_values={cid: 0 for cid in problem.customer_ids},
            objective_value=0.0,
            budget_total=problem.budget,
            budget_used=0.0,
            budget_remaining=problem.budget,
        )

    # ── 1. Build ILP ───────────────────────────────────────────────────────
    prob = pulp.LpProblem("OptiRetain_Allocation", pulp.LpMaximize)

    x_vars = {cid: pulp.LpVariable(f"x_{cid}", cat="Binary") for cid in customer_ids}

    # Objective: maximize Σ x_i · (uplift_i × clv_i - cost_i)
    prob += pulp.lpSum(x_vars[cid] * value_filt[i] for i, cid in enumerate(customer_ids))

    # Budget constraint: Σ x_i · cost_i ≤ budget
    if problem.budget > 0 and cost_filt.sum() > 0:
        prob += (
            pulp.lpSum(x_vars[cid] * cost_filt[i] for i, cid in enumerate(customer_ids))
            <= problem.budget,
            "Budget",
        )

    # ── 2. Solve ───────────────────────────────────────────────────────────
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=solver_timeout))

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        logger.warning("ILP status: %s — falling back to greedy heuristic.", pulp.LpStatus[status])
        return _solve_greedy(customer_ids, uplift_filt, clv_filt, cost_filt, value_filt, problem.budget)

    # ── 3. Read solution ───────────────────────────────────────────────────
    selected = [cid for cid in customer_ids if pulp.value(x_vars[cid]) == 1]
    rejected = [cid for cid in customer_ids if pulp.value(x_vars[cid]) != 1]

    obj_value = float(pulp.value(prob.objective)) if prob.objective else 0.0
    budget_used = sum(cost_filt[i] for i, cid in enumerate(customer_ids) if pulp.value(x_vars[cid]) == 1)

    # Build full x_values dict (0 for customers not in Persuadable pool).
    full_x = {cid: 0 for cid in problem.customer_ids}
    for cid in customer_ids:
        full_x[cid] = int(pulp.value(x_vars[cid])) if pulp.value(x_vars[cid]) is not None else 0

    # ── 4. Compute output metrics ──────────────────────────────────────────
    n_selected = len(selected)
    total_uplift_lift = float(uplift_filt[:n_selected].sum() * clv_filt[:n_selected].mean()) if selected else 0.0

    metrics: dict[str, float] = {
        "n_persuadable_input": len(customer_ids),
        "n_selected": n_selected,
        "coverage_pct": float(100.0 * n_selected / max(len(customer_ids), 1)),
        "objective_value": obj_value,
        "budget_used": budget_used,
        "budget_remaining": problem.budget - budget_used,
        "avg_cost_per_offer": budget_used / max(n_selected, 1),
    }

    logger.info("Optimization complete — selected %d/%d customers, objective=%.2f, budget used=%0.0f",
                n_selected, len(customer_ids), obj_value, budget_used)

    return AllocationResult(
        customer_ids=list(problem.customer_ids),
        selected=selected,
        rejected=rejected + [cid for cid in problem.customer_ids if cid not in customer_ids],
        x_values=full_x,
        objective_value=obj_value,
        budget_total=problem.budget,
        budget_used=budget_used,
        budget_remaining=problem.budget - budget_used,
        metrics=metrics,
    )


def solve_greedy_fallback(
    customer_ids: list[str],
    uplift: np.ndarray,
    clv: np.ndarray,
    cost: np.ndarray,
    budget: float,
) -> AllocationResult:
    """Greedy value/cost ratio heuristic as a fast fallback for large datasets.

    Parameters are the same as ``solve_knapsack()`` but without the KnapsackProblem wrapper.
    """
    problem = KnapsackProblem(customer_ids, uplift, clv, cost, budget)
    return solve_knapsack(problem, use_greedy_fallback=True)


# ── Greedy heuristic (fallback) ─────────────────────────────────────────────────

def _solve_greedy(
    customer_ids: list[str],
    uplift: np.ndarray,
    clv: np.ndarray,
    cost: np.ndarray,
    value: np.ndarray,
    budget: float,
) -> AllocationResult:
    """Greedy knapsack: sort by value/cost ratio descending, fill until budget exhausted."""
    ratios = value / np.maximum(cost, 1e-10)  # avoid division by zero
    order = np.argsort(-ratios)  # descending

    selected = []
    used_budget = 0.0
    obj_value = 0.0

    for idx in order:
        cid = customer_ids[idx]
        if used_budget + cost[idx] <= budget:
            selected.append(cid)
            used_budget += cost[idx]
            obj_value += value[idx]

    rejected = [cid for cid in customer_ids if cid not in selected]

    full_x = {cid: 1 if cid in selected else 0 for cid in customer_ids}

    return AllocationResult(
        customer_ids=customer_ids,
        selected=selected,
        rejected=rejected,
        x_values=full_x,
        objective_value=obj_value,
        budget_total=budget,
        budget_used=used_budget,
        budget_remaining=budget - used_budget,
        metrics={
            "n_persuadable_input": len(customer_ids),
            "n_selected": len(selected),
            "coverage_pct": 100.0 * len(selected) / max(len(customer_ids), 1),
            "objective_value": obj_value,
            "budget_used": used_budget,
            "method": "greedy_ratio_heuristic",
        },
    )
