"""End-to-end pipeline entry point for OptiRetain.

Usage::

    python -m optiretain.pipeline --budget 50000 --discount 0.20

Each layer is implemented as it becomes available:

- Layer 1 (data loading) — complete
- Layer 2 (feature engineering) — complete
- Layer 3 (risk radar / XGBoost + SHAP) — complete
- Layer 4a (uplift engine / EconML DML) — complete
- Layer 4b (customer segmentation) — complete
- Layer 5 (ROI maximizer / PuLP ILP) — complete
- Layer 6 (dashboard export + static frontend) — complete

Run order & artifacts:
    1. data/loader.py           → DataFrame (raw churn dataset)
    2. data/features.py         → df_treated + X_encoded + encoder + treatment_meta
    3. risk/train_xgb.py        → models/risk_radar.pkl (XGB + calibrated)
    4. risk/explain_shap.py     → per-customer SHAP driver explanations
    5. uplift/dml_cate.py       → CATE estimates + metadata
    6. uplift/segmentation.py   → segment labels + eligibility mask
    7. optimize/knapsack.py     → ILP allocation decisions
    8. export/dashboard_json.py → dashboard/customers.json + static HTML frontend
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("optiretain.pipeline")


def run(
    budget: float = 50_000,
    discount: float = 0.20,
    model_dir: Path | None = None,
    data_raw: Path | None = None,
) -> dict:
    """Execute the full OptiRetain pipeline end-to-end.

    Parameters
    ----------
    budget : float
        Total retention marketing budget (used by Layer 5 — Knapsack).
    discount : float
        Standard discount percentage offered to customers (e.g. ``0.20``).
    model_dir : Path | None
        Override default models directory.
    data_raw : Path | None
        Override default raw data path.

    Returns
    -------
    dict
        Summary dict with keys: *data*, *risk*, *dml*, *allocation*, *dashboard_path*.
    """
    t0 = time.time()
    results = {}

    # ═══════════════════ Layer 1 — Data Loading ════════════════════════════════
    logger.info("=" * 60)
    logger.info("Layer 1 — Data Loading")
    logger.info("=" * 60)
    from optiretain.data.loader import load_telco_data

    raw_path = data_raw or str(__import__("optiretain.config", fromlist=["TELECO_FILE"]).TELECO_FILE)
    df = load_telco_data(raw_path)
    logger.info("[L1] Loaded %d rows, %d columns.", df.shape[0], df.shape[1])
    logger.info("     TotalCharges missing values: %d", df["TotalCharges"].isna().sum())

    results["data"] = {"rows": df.shape[0], "columns": df.shape[1]}

    # ═══════════════════ Layer 2 — Feature Engineering ═════════════════════════
    logger.info("=" * 60)
    logger.info("Layer 2 — Feature Engineering")
    logger.info("=" * 60)
    from optiretain.data.features import engineer_features

    df_treated, X, encoder, treatment_meta = engineer_features(
        df, discount_pct=discount, seed=42
    )

    n_train = int(df_treated["Churn"].sum())
    logger.info("[L2] Imputed TotalCharges → 0 NaNs remaining")
    logger.info("     CLV columns: cltv_raw=%.1f, clv_computed=%.1f",
                df_treated["cltv_raw"].mean(), df_treated["clv_computed"].mean())
    logger.info("     Treatment rate: %.2f%% (%.0f%% discount)",
                df_treated["received_discount"].mean() * 100, discount * 100)

    results["data"].update({
        "cltv_raw_mean": float(df_treated["cltv_raw"].mean()),
        "clv_computed_mean": float(df_treated["clv_computed"].mean()),
        "treatment_rate": float(df_treated["received_discount"].mean()),
    })

    # ═══════════════════ Layer 3 — Risk Radar (XGBoost) ════════════════════════
    logger.info("=" * 60)
    logger.info("Layer 3 — Risk Radar (XGBoost + Calibration)")
    logger.info("=" * 60)
    from optiretain.risk.train_xgb import train_risk_radar

    raw_model, calibrated_model, risk_metadata = train_risk_radar(
        X, df_treated["Churn"], feature_names=list(X.columns) if isinstance(X, list) or hasattr(X, "__len__") else None
    )

    logger.info("[L3] Risk Radar — AUC=%.4f, Brier=%.4f",
                risk_metadata["val_auc_calibrated"], risk_metadata["brier_score"])

    results["risk"] = {
        "model_type": risk_metadata["model_type"],
        "val_auc_calibrated": risk_metadata["val_auc_calibrated"],
        "brier_score": risk_metadata["brier_score"],
    }

    # Get churn probabilities for all customers.
    import numpy as np
    X_arr = _to_array(X)
    p_churn_all = calibrated_model.predict_proba(X_arr)[:, 1]

    # ═══════════════════ SHAP Explanations ═════════════════════════════════════
    logger.info("=" * 60)
    logger.info("SHAP Explanations")
    logger.info("=" * 60)
    from optiretain.risk.explain_shap import explain_customers

    feature_cols_list = list(X.columns) if isinstance(X, pd.DataFrame) else [f"feature_{i}" for i in range(len(X_arr[0]))]
    customer_ids_all = df_treated["customerID"].astype(str).tolist()

    shap_exps = explain_customers(
        X_encoded=X,
        customer_ids=customer_ids_all,
        p_churn=p_churn_all,
        feature_columns=feature_cols_list,
        top_k=5,
    )
    logger.info("[SHAP] Explained %d customers (top 5 drivers each).", len(shap_exps))

    results["shap"] = {"n_explained": len(shap_exps)}

    # ═══════════════════ Layer 4a — Uplift Engine (EconML DML) ════════════════
    logger.info("=" * 60)
    logger.info("Layer 4a — Uplift Engine (EconML DML)")
    logger.info("=" * 60)
    from optiretain.uplift.dml_cate import fit_dml_cate

    cate_result = fit_dml_cate(
        X_features=X,
        treatment=df_treated["received_discount"].values,
        outcome=df_treated["Churn"].values,
        df_with_treatment=df_treated,
        n_estimators=300,
        cv=5,
    )

    logger.info("[L4a] CATE median=%.4f, uplift positive=%d%%",
                cate_result.cate.median(), int(cate_result.uplift_positive_pct))

    results["dml"] = {
        "cate_median": cate_result.cate_median,
        "uplift_positive_pct": cate_result.uplift_positive_pct,
        "method": cate_result.metadata.get("method", ""),
    }

    # ═══════════════════ Layer 4b — Customer Segmentation ══════════════════════
    logger.info("=" * 60)
    logger.info("Layer 4b — Customer Segmentation")
    logger.info("=" * 60)
    from optiretain.uplift.segmentation import segment_customers

    seg_result = segment_customers(
        p_churn=p_churn_all,
        uplift=cate_result.uplift,
        cate_lb=cate_result.cate_lb,
    )

    logger.info("[L4b] Segments: %s", seg_result.summary_counts)

    # Add segmentation to results DataFrame.
    df_treated["p_churn"] = p_churn_all
    df_treated["cate"] = cate_result.cate
    df_treated["uplift"] = cate_result.uplift
    df_treated["segment"] = seg_result.segments.values
    df_treated["is_eligible"] = seg_result.is_eligible

    results["segmentation"] = seg_result.summary_counts.copy()

    # ═══════════════════ Layer 5 — ROI Maximizer (PuLP ILP) ════════════════════
    logger.info("=" * 60)
    logger.info("Layer 5 — ROI Maximizer (PuLP ILP Knapsack)")
    logger.info("=" * 60)
    from optiretain.optimize.knapsack import solve_knapsack, KnapsackProblem

    # Only Persuadable customers enter the pool.
    persuadable_mask = seg_result.is_eligible
    if persuadable_mask.sum() == 0:
        logger.warning("[L5] No Persuadable customers — skipping optimization.")
        allocation_result = None
        results["allocation"] = {"status": "no_persuadable"}
    else:
        pers_ids = df_treated.loc[persuadable_mask, "customerID"].astype(str).tolist()

        # Build per-customer arrays for the Persuadable pool.
        uplift_filt = cate_result.uplift[persuadable_mask]
        clv_filt = df_treated.loc[persuadable_mask, "clv_computed"].values
        cost_filt = df_treated.loc[persuadable_mask, "treatment_cost"].values

        prob_input = KnapsackProblem(
            customer_ids=pers_ids,
            uplift=uplift_filt,
            clv=clv_filt,
            cost=cost_filt,
            budget=budget,
        )

        allocation_result = solve_knapsack(prob_input, solver_timeout=120)

        logger.info("[L5] Selected %d/%d Persuadable, objective=%.2f, budget used=%0.0f",
                    allocation_result.metrics.get("n_selected", 0),
                    len(pers_ids),
                    allocation_result.objective_value,
                    allocation_result.budget_used)

        results["allocation"] = {k: v for k, v in allocation_result.metrics.items()}

    # ═══════════════════ Layer 6 — Dashboard Export ═════════════════════════════
    logger.info("=" * 60)
    logger.info("Layer 6 — Dashboard Export")
    logger.info("=" * 60)
    from optiretain.export.dashboard_json import export_dashboard_json

    # Build full customer DataFrame for export.
    export_df = df_treated[["customerID", "p_churn", "cate", "uplift",
                             "clv_computed", "treatment_cost", "segment"]].copy()

    dashboard_path = Path(__import__("optiretain.config", fromlist=["DASHBOARD_DIR"]).DASHBOARD_DIR) / "customers.json"

    records = export_dashboard_json(
        customers=export_df,
        allocation_result=allocation_result,
        shap_explanations=shap_exps if shap_exps else None,
        risk_metadata=risk_metadata,
        dml_metadata=cate_result.metadata,
        output_path=dashboard_path,
    )

    logger.info("[L6] Dashboard JSON written → %s (%d customers)", dashboard_path, len(records))
    results["dashboard"] = {"path": str(dashboard_path), "n_customers": len(records)}

    # ═══════════════════ Summary ════════════════════════════════════════════════
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1fs", elapsed)
    logger.info("  AUC:        %s", f"{risk_metadata['val_auc_calibrated']:.4f}" if risk_metadata else "N/A")
    logger.info("  Segments:   %s", str(results.get("segmentation", {})))
    logger.info("  Allocation: %d selected from %s persuadable",
                results.get("allocation", {}).get("n_selected", 0),
                results.get("segmentation", {}).get("Persuadable", 0))
    logger.info("=" * 60)

    return results


# ── Convenience helpers ───────────────────────────────────────────────────────────

def _to_array(obj):
    """Convert DataFrame/Series to numpy."""
    import pandas as pd
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.values
    return __import__("numpy").asarray(obj)


# ── CLI entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OptiRetain: end-to-end customer retention optimization pipeline")
    parser.add_argument("--budget", type=float, default=50_000, help="Total retention marketing budget")
    parser.add_argument("--discount", type=float, default=0.20, help="Discount fraction (e.g. 0.20 = 20%%)")
    parser.add_argument("--model-dir", type=str, default=None, help="Override models directory path")
    parser.add_argument("--data-raw", type=str, default=None, help="Override raw data file path")
    args = parser.parse_args()

    results = run(
        budget=args.budget,
        discount=args.discount,
        model_dir=Path(args.model_dir) if args.model_dir else None,
        data_raw=Path(args.data_raw) if args.data_raw else None,
    )
