"""End-to-end pipeline entry point for OptiRetain.

Usage::

    python -m optiretain.pipeline --budget 50000 --discount 0.20

Each layer is implemented as it becomes available. Currently:

- Layer 1 (data loading) — complete
- Layer 2 (feature engineering) — complete
- Layers 3–6 — stubs (see comments below)
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
    # ── Layer 1 — Data loading ──────────────────────────────────────────────
    from optiretain.data.loader import load_telco_data

    df = load_telco_data()
    print(f"[L1] Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")
    print(f"     TotalCharges missing values: {df['TotalCharges'].isna().sum()}")

    # ── Layer 2 — Feature engineering ───────────────────────────────────────
    from optiretain.data.features import engineer_features

    df, X, encoder, treatment_meta = engineer_features(df, discount_pct=discount)
    print("[L2] Imputed TotalCharges -> 0 NaNs remaining")
    print(f"     CLV columns: cltv_raw={df['cltv_raw'].mean():.1f}, "
          f"clv_computed={df['clv_computed'].mean():.1f}")
    print(f"     Treatment rate: {df['received_discount'].mean():.2%} "
          f"({treatment_meta['discount_pct'] * 100:.0f}% discount)")

    # ── Layer 3 — Risk Radar (XGBoost) — TODO ───────────────────────────────
    # xgb_model, shap_values = train_risk_radar(X, df["Churn"])
    # Save with joblib.dump → models/risk_radar.pkl

    # ── Layer 4 — Uplift Engine (EconML DML) — TODO ────────────────────────
    # cate_hat = fit_dml_cate(X, df["received_discount"], df["Churn"])
    # segment_customers(cate_hat, p_churn) → Persuadable / Sure Thing ...

    # ── Layer 5 — ROI Maximizer (PuLP ILP) — TODO ──────────────────────────
    # prob = solve_knapsack(customer_ids, cate, clv_computed, treatment_cost, budget)
    # recommended = [i for i in customer_ids if pulp.value(x[i]) == 1]

    # ── Layer 6 — Dashboard export — TODO ───────────────────────────────────
    # export_dashboard_json(df, X, encoder, ..., "dashboard/customers.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OptiRetain pipeline")
    parser.add_argument("--budget", type=float, default=50_000, help="Retention budget")
    parser.add_argument("--discount", type=float, default=0.20, help="Discount fraction")
    args = parser.parse_args()

    run(budget=args.budget, discount=args.discount)
