"""Dashboard export: merge all pipeline outputs into customers.json.

Produces a single JSON file containing one record per customer with:

- Risk Radar score (``p_churn``)
- Uplift/CATE estimate (``cate``, ``uplift``)
- CLV (both raw and computed)
- Segment label
- ILP allocation decision (``recommended``)
- Top SHAP drivers
- Metadata block with pipeline run summary

This JSON is consumed by the static HTML/JS dashboard frontend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from optiretain.config import DASHBOARD_DIR, SEED

logger = logging.getLogger(__name__)


# ── Public API ───────────────────────────────────────────────────────────────────

def export_dashboard_json(
    customers: pd.DataFrame,
    allocation_result=None,  # AllocationResult from Layer 5
    shap_explanations: Optional[list] = None,
    risk_metadata: Optional[dict] = None,
    dml_metadata: Optional[dict] = None,
    *,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """Merge all pipeline outputs into a single ``customers.json`` file.

    Parameters
    ----------
    customers : pd.DataFrame
        DataFrame with columns: *customerID*, *p_churn*, *cate*, *uplift*,
        *clv_computed*, *treatment_cost*, *segment* (at minimum).
    allocation_result : AllocationResult or None
        From ``optimize.knapsack.solve_knapsack()``. If provided, sets
        ``recommended`` flag per customer.
    shap_explanations : list[CustomerExplanation] or None
        From ``risk.explain_shap.explain_customers()``. Merges top drivers.
    risk_metadata : dict or None
        Training metadata from Layer 3 (AUC, Brier, best params).
    dml_metadata : dict or None
        Training metadata from Layer 4a (nuisance R², treatment rate, etc.).
    output_path : Path or None
        Custom output path. Defaults to ``dashboard/customers.json``.

    Returns
    -------
    list[dict]
        The serialized records (also written to disk).
    """
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = DASHBOARD_DIR / "customers.json"

    # ── Build customer records ─────────────────────────────────────────────
    customer_ids = customers["customerID"].astype(str).tolist() if "customerID" in customers else [f"C-{i}" for i in range(len(customers))]
    segments = customers.get("segment", pd.Series(["Sure Thing"] * len(customers)))
    p_churn = customers.get("p_churn", np.zeros(len(customers)))
    cate = customers.get("cate", np.zeros(len(customers)))
    uplift = customers.get("uplift", np.zeros(len(customers)))
    clv = customers.get("clv_computed", customers.get("CLTV", np.zeros(len(customers))))
    cost = customers.get("treatment_cost", np.zeros(len(customers)))

    # Build recommended flag from allocation result.
    recommended = np.array([False] * len(customers))
    if allocation_result is not None:
        x_values = allocation_result.x_values
        for i, cid in enumerate(customer_ids):
            recommended[i] = (x_values.get(str(cid), 0) == 1)

    # Build SHAP drivers lookup.
    shap_lookup = {}
    if shap_explanations:
        for exp in shap_explanations:
            shap_lookup[exp.customer_id] = [
                {
                    "feature": d.feature,
                    "value": d.value,
                    "shap": round(d.shap, 6),
                    "direction": d.direction,
                }
                for d in exp.top_drivers[:5]
            ]

    records = []
    for i, cid in enumerate(customer_ids):
        rec: dict = {
            "customer_id": str(cid),
            "p_churn": round(float(p_churn[i]), 4) if hasattr(p_churn, '__len__') else round(float(p_churn), 4),
            "cate": round(float(cate[i]), 6) if hasattr(cate, '__len__') else round(float(cate), 6),
            "uplift": round(float(uplift[i]), 6) if hasattr(uplift, '__len__') else round(float(uplift), 6),
            "clv": round(float(clv[i]), 2) if hasattr(clv, '__len__') else round(float(clv), 2),
            "cost": round(float(cost[i]), 2) if hasattr(cost, '__len__') else round(float(cost), 2),
            "expected_net_lift": round(
                float(p_churn[i] * clv[i] - cost[i]) if hasattr(p_churn, '__len__')
                else float(p_churn * clv - cost),
                2
            ),
            "recommended": bool(recommended[i]),
            "segment": str(segments.iloc[i]) if hasattr(segments, 'iloc') else str(segments),
        }

        # Add SHAP drivers.
        drivers = shap_lookup.get(str(cid), [])
        if drivers:
            rec["top_drivers"] = drivers

        records.append(rec)

    # ── Build metadata block ───────────────────────────────────────────────
    pipeline_meta: dict = {
        "run_timestamp": None,  # populated by caller if needed
        "random_state": SEED,
        "n_customers": len(records),
    }

    if risk_metadata:
        pipeline_meta["risk_radar"] = {
            "model_type": risk_metadata.get("model_type", ""),
            "val_auc_calibrated": risk_metadata.get("val_auc_calibrated"),
            "brier_score": risk_metadata.get("brier_score"),
            "best_params": risk_metadata.get("best_params"),
        }

    if dml_metadata:
        pipeline_meta["dml"] = {
            "method": dml_metadata.get("method", ""),
            "cate_median": dml_metadata.get("cate_median"),
            "uplift_positive_pct": dml_metadata.get("uplift_positive_pct"),
            "nuisance_r2_y": dml_metadata.get("nuisance_r2_y"),
        }

    if allocation_result is not None:
        pipeline_meta["allocation"] = {
            "n_selected": allocation_result.metrics.get("n_selected", 0),
            "objective_value": allocation_result.objective_value,
            "budget_used": allocation_result.budget_used,
            "budget_remaining": allocation_result.budget_remaining,
        }

    output = {"metadata": pipeline_meta, "customers": records}

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Also write a companion data.js so the dashboard works when opened
    # directly from the filesystem (file:// URLs block fetch() due to CORS).
    # app.js reads window.__OPTIRETAIN_DATA__ first and only falls back to
    # fetch() when served over HTTP.
    data_js_path = output_path.parent / "data.js"
    with open(data_js_path, "w") as f:
        f.write("// Auto-generated by OptiRetain pipeline — do not edit.\n")
        f.write("window.__OPTIRETAIN_DATA__ = ")
        json.dump(output, f)
        f.write(";\n")

    logger.info("Dashboard JSON written → %s (%d customers)", output_path, len(records))
    return records


def export_customers_json(
    df: pd.DataFrame,
    *,
    allocation_result=None,
    shap_explanations=None,
    risk_metadata=None,
    dml_metadata=None,
    output_path: Optional[Path] = None,
) -> list[dict]:
    """Convenience alias matching the original plan's naming."""
    return export_dashboard_json(
        customers=df,
        allocation_result=allocation_result,
        shap_explanations=shap_explanations,
        risk_metadata=risk_metadata,
        dml_metadata=dml_metadata,
        output_path=output_path,
    )
