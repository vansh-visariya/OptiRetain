"""Layer 4b — Customer segmentation using risk + uplift scores.

Classifies each customer into one of four retention segments:

| Segment       | Condition                                   | Action                         |
|---------------|---------------------------------------------|--------------------------------|
| **Persuadable**    | ``p_churn`` high AND ``uplift > ε``              | Eligible for ILP optimization      |
| **Sure Thing**     | ``p_churn`` low AND ``uplift ≈ 0``               | No intervention needed             |
| **Lost Cause**     | ``p_churn`` high AND ``uplift ≈ 0``              | Skip — unlikely to respond         |
| **Sleeping Dog**   | ``uplift < -ε`` (treatment *increases* churn)    | Explicitly exclude from offers     |

The thresholds are derived from the CATE confidence intervals so that only
customers with statistically meaningful uplift qualify as Persuadable or
Sleeping Dog.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────────

_SEGMENT_MAP = {
    "Persuadable": 1,   # treat — high value, high uplift
    "Sure Thing":   0,   # don't treat — already loyal
    "Lost Cause":  -1,   # don't treat — won't respond anyway
    "Sleeping Dog": -2,  # actively harmful to treat
}

# Default thresholds.
_P_CHURN_HIGH_DEFAULT: float = 0.5
_UPHLIFT_EPS_DEFAULT: float = 0.01  # minimum meaningful uplift


# ── Data classes ────────────────────────────────────────────────────────────────

@dataclass
class SegmentationResult:
    """Container for segmentation output."""
    segments: pd.Series              # segment label per customer
    segment_ids: np.ndarray          # numeric codes (Persuadable=1, Sure=0, Lost=-1, Dog=-2)
    is_eligible: np.ndarray          # bool — Persuadable only → eligible for ILP
    summary_counts: dict[str, int]   # count per segment


# ── Public API ───────────────────────────────────────────────────────────────────

def segment_customers(
    p_churn: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    cate_lb: Optional[np.ndarray] = None,
    *,
    p_churn_threshold: float = _P_CHURN_HIGH_DEFAULT,
    uplift_eps: float = _UHLIFT_EPS_DEFAULT,
) -> SegmentationResult:
    """Assign each customer to a retention segment.

    Parameters
    ----------
    p_churn : array-like
        Calibrated churn probabilities from the Risk Radar (Layer 3).
    uplift : array-like
        Uplift scores from the DML engine — positive means discount reduces churn.
    cate_lb : np.ndarray, optional
        Lower confidence bound on CATE. Used to derive ``uplift_eps`` dynamically.
        If provided and > 0, overrides *uplift_eps*.
    p_churn_threshold : float
        Threshold above which ``p_churn`` is considered "high" (default 0.5).
    uplift_eps : float
        Minimum absolute uplift to be considered meaningful (default 0.01).

    Returns
    -------
    SegmentationResult
        Named tuple with *segments* (labels), *segment_ids* (numeric codes),
        *is_eligible* (Persuadable mask), and *summary_counts*.
    """
    p_arr = np.asarray(p_churn).flatten()
    u_arr = np.asarray(uplift).flatten()

    # Derive eps from CATE confidence intervals if available.
    if cate_lb is not None:
        lb_arr = np.asarray(cate_lb).flatten()
        dynamic_eps = (-lb_arr).mean()  # upper bound of -CATE = lower bound of uplift
        if dynamic_eps > 0:
            uplift_eps = float(max(dynamic_eps, 1e-4))
            logger.info("Dynamic eps from CATE CI: %.6f", uplift_eps)

    high_risk = p_arr >= p_churn_threshold
    significant_uplift = u_arr >= uplift_eps
    harmful_uplift = u_arr < -uplift_eps

    segments: list[str] = []
    seg_ids: list[int] = []
    eligible: list[bool] = []

    for i in range(len(p_arr)):
        if harmful_uplift[i]:
            label = "Sleeping Dog"
            code = _SEGMENT_MAP["Sleeping Dog"]
            elig = False
        elif high_risk[i] and significant_uplift[i]:
            label = "Persuadable"
            code = _SEGMENT_MAP["Persuadable"]
            elig = True
        elif high_risk[i] and not significant_uplift[i]:
            label = "Lost Cause"
            code = _SEGMENT_MAP["Lost Cause"]
            elig = False
        else:
            label = "Sure Thing"
            code = _SEGMENT_MAP["Sure Thing"]
            elig = False

        segments.append(label)
        seg_ids.append(code)
        eligible.append(elig)

    seg_series = pd.Series(segments, name="segment")
    seg_arr = np.array(seg_ids, dtype=int)
    elig_arr = np.array(eligible, dtype=bool)

    summary_counts = {label: int((seg_series == label).sum()) for label in _SEGMENT_MAP}

    logger.info("Segmentation summary: %s", summary_counts)

    return SegmentationResult(
        segments=seg_series,
        segment_ids=seg_arr,
        is_eligible=elig_arr,
        summary_counts=summary_counts,
    )


def filter_persuadable(
    df: pd.DataFrame,
    *,
    p_churn_threshold: float = _P_CHURN_HIGH_DEFAULT,
    uplift_eps: float = _UHLIFT_EPS_DEFAULT,
) -> pd.DataFrame:
    """Filter DataFrame to only Persuadable customers (eligible for ILP).

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``p_churn``, ``uplift``, and optional ``cate_lb`` columns.
    p_churn_threshold, uplift_eps : float
        Same thresholds as ``segment_customers()``.

    Returns
    -------
    pd.DataFrame
        Only Persuadable customers.
    """
    p_arr = df["p_churn"].values if "p_churn" in df.columns else np.zeros(len(df))
    u_arr = df["uplift"].values if "uplift" in df.columns else np.zeros(len(df))

    lb_arr = df["cate_lb"].values if "cate_lb" in df.columns else None

    seg = segment_customers(p_arr, u_arr, cate_lb=lb_arr,
                            p_churn_threshold=p_churn_threshold, uplift_eps=uplift_eps)

    return df[seg.is_eligible].reset_index(drop=True)
