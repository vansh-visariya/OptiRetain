"""End-to-end pipeline entry point for OptiRetain.

Usage::

    python -m optiretain.pipeline --budget 50000 --discount 0.20

This is a placeholder orchestrator — each layer will be wired in as it
is implemented (currently only the data loader exists).
"""

from __future__ import annotations


def run(budget: float = 50_000, discount: float = 0.20) -> None:
    """Execute the full OptiRetain pipeline end-to-end.

    Parameters
    ----------
    budget : float
        Total retention marketing budget (used by Layer 3 — Knapsack).
    discount : float
        Standard discount percentage offered to customers (e.g. ``0.20``).
    """
    from optiretain.data.loader import load_telco_data

    # Layer 1 — Data loading
    df = load_telco_data()
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")
    print("TotalCharges missing values:", df["TotalCharges"].isna().sum())

    # TODO: Layer 2 — Feature engineering (features.py)
    # TODO: Layer 3 — Risk Radar (XGBoost churn model)
    # TODO: Layer 4 — Uplift Engine (EconML DML CATE)
    # TODO: Layer 5 — ROI Maximizer (PuLP knapsack)
    # TODO: Layer 6 — Dashboard export


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OptiRetain pipeline")
    parser.add_argument("--budget", type=float, default=50_000, help="Retention budget")
    parser.add_argument("--discount", type=float, default=0.20, help="Discount fraction")
    args = parser.parse_args()

    run(budget=args.budget, discount=args.discount)
