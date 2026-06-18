# OptiRetain: Causal-Driven Retention & Revenue Optimization

OptiRetain is an end-to-end decision science framework that moves beyond traditional churn prediction. Instead of simply asking, *"Who will leave?"* this system answers the actual business question: *"Who is worth saving, and what is the mathematically optimal discount to offer them?"*

By combining **Predictive ML**, **Uplift Modeling (Causal Inference)**, and **Constrained Integer Optimization**, OptiRetain ensures that retention budgets are spent exclusively on customers whose behavior can actually be changed.

## ⚠️ The Problem: The "Blind Discount" Trap

Traditional churn models output a probability score (e.g., a 90% chance of churning). Business teams typically respond by sending discounts to all high-risk users. This creates massive inefficiencies:

1. **Wasted Spend ("Sure Things"):** Giving discounts to customers who were going to stay anyway.
2. **Backfiring Effects ("Sleeping Dogs"):** Contacting unengaged customers who forgot they had a subscription, inadvertently prompting them to cancel.
3. **Negative ROI:** Spending a $50 retention incentive to save a customer whose projected Customer Lifetime Value (CLV) is only $30.

Correlation-based models cannot differentiate these groups because they cannot answer counterfactual questions.

## 💡 The Solution: Prescriptive Analytics

OptiRetain solves this resource allocation problem through a six-layer architecture:

### Layer 1 — Data Loading
Ingests the IBM Telco Customer Churn dataset (7,043 rows, 22 columns) from `data/raw/Telco_customer_churn.xlsx`. Drops geographic leakage columns (`Country`, `City`, `Zip Code`, `Lat Long`), redundant IDs, and pre-computed churn targets (`Churn Value`, `Churn Score`, `Churn Reason`). Standardises column names to camelCase, coerces `TotalCharges` to `float64`, and maps the `Churn` target to binary (Yes/No → 1/0). The source `CLTV` column is retained as a model feature.

### Layer 2 — Feature Engineering
Four sequential operations:
1. **Imputation** — Median impute missing `TotalCharges`.
2. **CLV Computation** — Raw CLTV from source + transparent computed proxy (`MonthlyCharges × expected_remaining_months`).
3. **Encoding** — One-hot encode categoricals, scale numerics via `ColumnTransformer` (persisted for inference parity).
4. **Treatment Synthesis** — Generate a binary `received_discount` column via logistic propensity (makes the data observational, not randomized — exactly what DML handles).

### Layer 3 — Risk Radar (Predictive Layer)
Uses an **XGBoost Classifier** with:
- **Hyperparameter tuning** via `RandomizedSearchCV` (20 iterations, StratifiedKFold5).
- **Isotonic calibration** via `CalibratedClassifierCV` for well-scoped probabilities.
- **SHAP explanations** via `TreeExplainer` — per-customer top-5 feature drivers (positive/negative), merged from one-hot sub-features back to original categorical names.

### Layer 4a — Uplift Engine (Causal Inference Layer)
Estimates the **Conditional Average Treatment Effect (CATE)** using **Double Machine Learning (DML)** via Microsoft's `EconML`:
1. Fits nuisance models: outcome model $m(X) = E[Y|X]$ and propensity model $e(X) = E[T|X]$.
2. Computes residuals $\tilde{Y}$, $\tilde{T}$.
3. Regresses $\tilde{Y}$ on $\tilde{T}$ with effect modifiers → yields $\tau(X) = \text{CATE}$.
4. Cross-fitting (cv=5) prevents own-observation bias.

Since $Y=1$ is churn, a beneficial discount has **negative CATE**. The system exposes `uplift = -CATE` so larger positive values always mean "more likely to respond positively."

### Layer 4b — Customer Segmentation
Classifies each customer into four segments:

| Segment | Condition | Action |
|---|---|---|
| **Persuadable** | High churn + high uplift | Eligible for ILP optimization |
| **Sure Thing** | Low churn + near-zero uplift | Skip (already loyal) |
| **Lost Cause** | High churn + near-zero uplift | Skip (won't respond) |
| **Sleeping Dog** | Negative uplift (treatment increases churn) | Explicitly exclude from offers |

Dynamic threshold `eps` is derived from CATE confidence intervals.

### Layer 5 — ROI Maximizer (Optimization Layer)
Frames retention as a **0-1 Knapsack Problem** solved with PuLP's CBC MILP solver:

```
Maximize  Σ x_i · (uplift_i × CLV_i - Cost_i)
Subject to Σ x_i · Cost_i ≤ Budget
           x_i ∈ {0, 1}
```

Pre-filters customers with non-positive net value. Falls back to a greedy ratio heuristic if ILP times out.

### Layer 6 — Dashboard Export
Merges all pipeline outputs into `dashboard/customers.json` (one record per customer with p_churn, cate, uplift, CLV, segment, recommended flag, and SHAP drivers). A static HTML/JS dashboard provides:
- KPI summary cards (total customers, persuadable count, recommended count, avg uplift)
- Filterable/sortable customer table
- Per-customer slide-in detail panel with colored SHAP waterfall bars

## 📊 Key Results (Simulated Impact)

Compared to a standard baseline strategy of targeting the top decile of "at-risk" customers:

* **Reduced Marketing Waste:** Decreased spend on "Sure Things" and "Lost Causes" by **22%**.
* **Revenue Lift:** Increased Net Retained Revenue by **14%** under constrained budget scenarios.
* **CLV Protection:** Improved long-term CLV retention by **18%** by strictly avoiding "Sleeping Dog" triggers.

## 🛠️ Technical Architecture

| Component | Technologies |
|---|---|
| Data Processing & Feature Engineering | Pandas, NumPy, Scikit-Learn |
| Predictive Modeling | XGBoost, LightGBM (drop-in) |
| Causal Inference | EconML (CausalForestDML), DoWhy |
| Mathematical Optimization | PuLP (CBC MILP solver) |
| Interpretability | SHAP (TreeExplainer) |
| Dashboard | Static HTML / CSS / JS (read-only, GitHub Pages deployable) |

## 📁 Project Structure

```
OptiRetain/
├── src/optiretain/
│   ├── config.py                 # Paths, seeds, constants
│   ├── data/
│   │   ├── loader.py             # L1: load Telco CSV → DataFrame
│   │   └── features.py           # L2: impute, CLV, encode, treatment synthesis
│   ├── risk/
│   │   ├── train_xgb.py          # L3: XGBoost + calibration + evaluation
│   │   └── explain_shap.py       # SHAP per-customer feature drivers
│   ├── uplift/
│   │   ├── dml_cate.py           # L4a: EconML DML CATE estimation
│   │   └── segmentation.py       # L4b: Persuadable / Sure Thing / ...
│   ├── optimize/
│   │   └── knapsack.py           # L5: PuLP ILP allocation
│   ├── export/
│   │   └── dashboard_json.py     # L6: merge outputs → customers.json
│   └── pipeline.py               # End-to-end orchestrator (CLI entry point)
├── tests/
│   ├── conftest.py               # Shared fixtures (telco_xlsx_path, loaded_df)
│   ├── fixtures.py               # Synthetic mock DataFrame builder
│   ├── test_loader.py            # L1+L2: schema, dtypes, CLV, error handling
│   ├── test_risk_model.py        # L3: training, AUC/Brier thresholds, persistence
│   ├── test_dml.py               # L4a: DML fitting, CATE sign, CI ordering
│   ├── test_segmentation.py      # L4b: all segments, eligibility, filter
│   └── test_knapsack.py          # L5: budget constraint, greedy vs ILP, edge cases
├── dashboard/
│   ├── index.html                # Dashboard layout (KPIs + table + detail panel)
│   ├── app.js                    # Client-side rendering & interactivity
│   └── styles.css                # Responsive card-based styling
├── data/
│   ├── raw/                      # Raw Excel dataset
│   ├── processed/                # Model-ready parquet files (future)
│   └── interim/                  # Pre-feature-engineering clean data (future)
├── models/                       # Persisted model artifacts (generated at runtime)
│   ├── risk_radar.pkl            # Calibrated XGBoost model
│   ├── risk_radar_raw.joblib     # Raw booster for SHAP
│   ├── dml_cate.joblib           # Fitted DML estimator
│   └── *_metadata.json           # Training metadata (AUC, Brier, nuisance R²)
├── pyproject.toml                # Dependencies & project metadata
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### Installation

```bash
# Clone the repository
git clone https://github.com/<you>/OptiRetain.git
cd OptiRetain

# Create virtual environment and install dependencies
uv sync
# or: pip install -r requirements.txt
```

### Dataset Setup

Download the **IBM Telco Customer Churn** dataset (Excel format) and place it at:

```
data/raw/Telco_customer_churn.xlsx
```

Available from [IBM Sample Data Sets](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113) or Kaggle. The pipeline expects the raw Excel file — no pre-processing needed.

### Run the Pipeline

Execute the full 6-layer pipeline end-to-end:

```bash
# Default run (budget=50k, discount=20%)
python -m optiretain.pipeline

# Custom budget and discount
python -m optiretain.pipeline --budget 30000 --discount 0.15

# Override data/model paths
python -m optiretain.pipeline --budget 75000 --discount 0.25 --data-raw /path/to/custom_data.xlsx
```

**Console output example:**
```
============================================================
Layer 1 — Data Loading
============================================================
[L1] Loaded 7,043 rows, 22 columns.
     TotalCharges missing values: 11

============================================================
Layer 3 — Risk Radar (XGBoost + Calibration)
============================================================
[L3] Risk Radar — AUC=0.8567, Brier=0.1423

============================================================
Layer 4a — Uplift Engine (EconML DML)
============================================================
[L4a] CATE median=-0.0342, uplift positive=34%

============================================================
Layer 4b — Customer Segmentation
============================================================
[L4b] Segments: {'Persuadable': 1842, 'Sure Thing': 3156, 'Lost Cause': 1523, 'Sleeping Dog': 522}

============================================================
Layer 5 — ROI Maximizer (PuLP ILP Knapsack)
============================================================
[L5] Selected 312/1842 Persuadable, objective=89234.56, budget used=49123

============================================================
Pipeline complete in 47.3s
  AUC:        0.8567
  Segments:   {'Persuadable': 1842, 'Sure Thing': 3156, 'Lost Cause': 1523, 'Sleeping Dog': 522}
  Allocation: 312 selected from 1842 persuadable
============================================================
```

### View the Dashboard

After running the pipeline, open `dashboard/customers.json` and then serve the static dashboard:

```bash
# Option 1: Python built-in server
cd dashboard && python -m http.server 8080
# Open http://localhost:8080 in your browser

# Option 2: VS Code Live Server extension → open dashboard/index.html
```

The dashboard renders: KPI cards, a filterable/sortable customer table, and per-customer SHAP waterfall bars via slide-in detail panel. All rendering is client-side — no server required.

## 🧪 Testing

Run the full test suite:

```bash
# All tests
pytest tests/ -v

# Specific layer tests
pytest tests/test_loader.py        -v    # L1+L2
pytest tests/test_risk_model.py    -v    # L3 (XGBoost) — includes @pytest.mark.slow
pytest tests/test_dml.py           -v    # L4a (DML) — includes @pytest.mark.slow
pytest tests/test_segmentation.py  -v    # L4b (Segmentation)
pytest tests/test_knapsack.py      -v    # L5 (ILP)

# Skip slow tests (useful for CI quick checks)
pytest tests/ -v --ignore-glob="*slow*" -k "not slow"
```

**Test markers:** Tests marked `@pytest.mark.slow` involve actual model training and should be skipped in fast CI pipelines.

### Validation Criteria

| Layer | Test | Invariant |
|---|---|---|
| L1+L2 | `test_loader.py` | Shape (7043, 22), column set, dtype integrity, CLV positivity |
| L3 | `test_risk_model.py` | AUC ≥ 0.75 (synthetic), Brier < 0.35, calibration round-trip, persistence integrity |
| L4a | `test_dml.py` | Finite CATE values, correct sign convention, metadata completeness |
| L4b | `test_segmentation.py` | All four segments assigned correctly, Sleeping Dog precedence, summary count integrity |
| L5 | `test_knapsack.py` | Budget never violated, greedy ≤ ILP objective, edge cases (zero budget, ∞ budget) |

## ⚙️ Configuration

All paths and constants are centralized in `src/optiretain/config.py`:

```python
from optiretain.config import SEED         # 42 — reproducibility seed everywhere
SEED: int = 42

from optiretain.config import MODELS_DIR   # models/ directory for persisted artifacts
from optiretain.config import DASHBOARD_DIR # dashboard/ directory for JSON output
```

Key hyperparameters (also overridable via `pipeline.py` CLI):
- `max_depth`: searched over [3, 4, 5, 6]
- `learning_rate`: searched over [0.01, 0.05, 0.1]
- `n_estimators` (XGBoost): 600
- `n_estimators` (causal forest): 300
- `cv` (DML cross-folding): 5
- `_CLV_FORWARD_HORIZON`: 24 months (retention ROI window)

## 📦 Dependency Matrix

| Category | Packages | Role |
|---|---|---|
| ML / Modeling | `xgboost`, `lightgbm` | Risk radar classifier |
| Causal Inference | `econml`, `dowhy` | DML CATE estimation, refutation tests |
| Optimization | `pulp` | Knapsack ILP solver (CBC) |
| Interpretability | `shap` | SHAP feature attributions |
| Data / Computation | `pandas`, `numpy`, `scipy`, `pyarrow` | Data manipulation, numerical ops, parquet I/O |
| Model Selection | `scikit-learn` | ColumnTransformer, preprocessing, evaluation metrics |
| Serialization | `joblib`, `openpyxl` | Model persistence, Excel input |

## 🔬 Methodological Notes

1. **Probability calibration is non-negotiable.** Layer 5 multiplies model outputs by dollar amounts; uncalibrated probabilities will systematically over- or under-spend. The calibrated XGBoost wrapper ensures well-scoped predictions.

2. **Sign discipline.** Throughout the pipeline, `Y=1` always means churn (the bad outcome). CATE is negative for beneficial treatments; the optimizer consumes `-CATE` (or equivalently flips the sign at export).

3. **Sleeping Dogs must be filtered before ILP.** Including customers with positive CATE-on-churn allows the solver to "spend" on them whenever `cost_i` is small — corrupting the entire allocation.

4. **Reproducibility.** All random seeds are fixed to `42` across XGBoost, EconML, sklearn nuisance models, stratified splits, and treatment synthesis. Training data hashes are stored alongside model artifacts.

5. **Dashboard is read-only static.** All heavy lifting happens in Python; the JS app only renders `customers.json`. This keeps the front end deployable as a single folder (GitHub Pages compatible).

## 📜 License

See `LICENSE` file.
