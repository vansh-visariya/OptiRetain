# OptiRetain — End-to-End Implementation Plan

Below is a developer-ready blueprint for building the OptiRetain pipeline as described in `doc.md` and `README.md`. It is organized by the natural data flow: **Data → Risk Radar → Uplift Engine → ROI Maximizer → Dashboard**, with the four pillars (XGBoost, SHAP, EconML/DML, PuLP) integrated at the correct stages.

---

## 0. Proposed Repository Layout

Since the repo currently only contains `LICENSE`, `README.md`, and `doc.md`, the following structure is recommended (do not create yet — for planning only):

```
OptiRetain/
├── data/
│   ├── raw/                 # original CSVs
│   ├── interim/             # cleaned, pre-feature-engineering
│   └── processed/           # model-ready parquet files
├── src/optiretain/
│   ├── config.py            # paths, seeds, hyperparams
│   ├── data/
│   │   ├── loader.py        # download + load Telco/synthetic
│   │   └── features.py      # encoding, CLV computation
│   ├── risk/
│   │   ├── train_xgb.py     # Layer 1 trainer
│   │   └── explain_shap.py  # SHAP value extraction
│   ├── uplift/
│   │   ├── dml_cate.py      # EconML DML estimator
│   │   └── segmentation.py  # Persuadable / Sure-Thing / etc.
│   ├── optimize/
│   │   └── knapsack.py      # PuLP ILP solver
│   ├── pipeline.py          # orchestrator
│   └── export/
│       └── dashboard_json.py # writes JSON for HTML/JS UI
├── dashboard/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── notebooks/               # exploratory analysis only
├── tests/
└── pyproject.toml
```

Dependency installation should be done via `uv pip install` / `poetry add` (per project rules), not by editing `pyproject.toml` by hand:

```
pandas numpy scikit-learn xgboost lightgbm shap econml dowhy pulp scipy pyarrow
```

---

## 1. Dataset Selection

### Primary recommendation: **IBM Telco Customer Churn** (with a synthetic treatment column overlaid)

**Why:** It is the canonical churn benchmark, has a clean `Churn` target, and contains features that map naturally to billing/contract attributes — but it **does not** include a real treatment indicator. We synthesize one to enable causal estimation, which is the standard approach in EconML tutorials.

| Property | Detail |
|---|---|
| Source | `IBM Telco Customer Churn` (7,043 rows, 21 cols) |
| Target $Y$ | `Churn` (binary: Yes/No → 1/0) |
| Features $X$ | tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, InternetService, OnlineSecurity, TechSupport, … |
| Treatment $T$ (synthesized) | `received_discount` (binary), assigned via a **logistic propensity** that depends on tenure & MonthlyCharges so the data is *observational, not randomized* (this is precisely what DML is designed for) |
| Cost per treatment | derived from `MonthlyCharges × discount_pct × expected_months` |
| CLV proxy | `MonthlyCharges × expected_remaining_tenure` (or fit a simple gamma-gamma model) |

### Treatment-generation recipe (so DML has something to estimate)

```
propensity_i = sigmoid(α + β1·tenure + β2·MonthlyCharges + ε)
T_i ~ Bernoulli(propensity_i)
Y_i = churn_baseline_i - τ(X_i)·T_i + noise
```
where `τ(X)` is a known heterogeneous effect (e.g., larger for mid-tenure / mid-spend customers). Persisting the **ground-truth `τ(X)`** lets you validate DML recovery in tests.

### Alternative (if a fully realistic causal signal is needed)
- **Criteo Uplift Modeling dataset** (~25M rows, real randomized exposure flag, anonymized features). Heavier compute but provides a real `treatment` and `conversion` pair without simulation.

---

## 2. Layer 1 — Risk Radar (XGBoost / LightGBM)

**Module:** `src/optiretain/risk/train_xgb.py`

### 2.1 Preprocessing
1. Coerce `TotalCharges` to numeric, impute the ~11 blank rows with `median`.
2. One-hot encode categorical columns via `sklearn.compose.ColumnTransformer` (keep the fitted transformer on disk for inference parity).
3. Stratified `train_test_split(test_size=0.2, stratify=Y, random_state=42)`.

### 2.2 Model
Use **XGBoost** as the default (LightGBM is a drop-in alternative for larger data).

```
XGBClassifier(
    n_estimators=600, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9,
    objective='binary:logistic', eval_metric='auc',
    scale_pos_weight = neg_count / pos_count,   # handle imbalance
    tree_method='hist', random_state=42)
```

### 2.3 Tuning & Validation
- `StratifiedKFold(n_splits=5)` with `RandomizedSearchCV` over `max_depth`, `learning_rate`, `min_child_weight`, `gamma`.
- Track **ROC-AUC, PR-AUC, Brier score** (calibration matters because we will combine `P(churn)` with monetary values downstream).
- Apply **`CalibratedClassifierCV(method='isotonic')`** on the best model — well-calibrated probabilities are essential when multiplying by CLV in Layer 3.

### 2.4 Persistence
Save with `joblib.dump`: `{preprocessor, calibrated_model, feature_names, train_metadata}` → `models/risk_radar.pkl`.

**Output of this layer per customer:** `p_churn_i ∈ [0,1]`.

---

## 3. SHAP Integration & Dashboard Surfacing

**Module:** `src/optiretain/risk/explain_shap.py`

### 3.1 Compute SHAP values
Because XGBoost is tree-based, use the exact and fast `TreeExplainer`:

```
explainer = shap.TreeExplainer(xgb_model)          # raw, uncalibrated model
shap_values = explainer.shap_values(X_processed)   # shape (n_customers, n_features)
expected_value = explainer.expected_value
```

> Note: SHAP must run on the raw XGBoost booster, not the `CalibratedClassifierCV` wrapper. Persist both — the calibrated one for probability scoring, the raw one for explanations.

### 3.2 Per-customer driver extraction
For each customer `i`:
1. Pair `(feature_name, shap_value_i)`.
2. Sort by `|shap_value|` desc.
3. Keep top-K (e.g., K=5) positive drivers (push toward churn) and top-K negative drivers (push toward retention).
4. Convert one-hot-encoded contributions back to their **original categorical name** (sum sub-feature contributions before ranking).

### 3.3 JSON contract for the dashboard
`src/optiretain/export/dashboard_json.py` writes one record per customer:

```
{
  "customer_id": "7590-VHVEG",
  "p_churn": 0.83,
  "cate": 0.21,
  "clv": 412.50,
  "cost": 25.00,
  "expected_net_lift": 61.62,
  "recommended": true,
  "segment": "Persuadable",
  "top_drivers": [
    {"feature": "Contract", "value": "Month-to-month", "shap": +0.41, "direction": "increases_risk"},
    {"feature": "tenure", "value": 2, "shap": +0.22, "direction": "increases_risk"},
    {"feature": "OnlineSecurity", "value": "No", "shap": +0.14, "direction": "increases_risk"},
    {"feature": "TechSupport", "value": "Yes", "shap": -0.09, "direction": "decreases_risk"}
  ]
}
```

### 3.4 Rendering in HTML/JS
- **Global view:** a SHAP **summary plot** (beeswarm) exported as a static PNG via `shap.summary_plot(..., show=False)` + `plt.savefig`, OR re-implemented in JS by feeding the JSON to a horizontal bar chart (D3 / Chart.js).
- **Per-customer drill-down:** a horizontal **waterfall** bar chart in `app.js` that reads `top_drivers[]` for the selected customer — red bars to the right for risk-increasing features, green to the left for risk-decreasing. The chart base = `expected_value` (also written into the JSON file's metadata block).
- **Search / filter bar:** dashboard loads `customers.json`, lets the analyst filter by `segment`, sort by `expected_net_lift`, and click a row to open the waterfall panel.

---

## 4. Layer 2 — Uplift Engine: CATE via EconML DML

**Module:** `src/optiretain/uplift/dml_cate.py`

### 4.1 Conceptual setup
DML estimates the causal effect of $T$ on $Y$ while non-parametrically partialling out confounders $X$:

1. **First stage (nuisance models):**
   - $\hat m(X) = E[Y|X]$ — outcome model
   - $\hat e(X) = E[T|X]$ — propensity model
2. **Compute residuals:** $\tilde Y = Y - \hat m(X)$,  $\tilde T = T - \hat e(X)$
3. **Second stage:** regress $\tilde Y$ on $\tilde T$ with $X$ as effect-modifiers → yields $\hat\tau(X) = \text{CATE}$.

Cross-fitting (default `cv=5`) prevents own-observation bias.

### 4.2 Code skeleton

```
from econml.dml import CausalForestDML       # heterogeneous CATE, tree-based final stage
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

est = CausalForestDML(
    model_y = GradientBoostingRegressor(n_estimators=300, max_depth=4),
    model_t = GradientBoostingClassifier(n_estimators=300, max_depth=4),
    discrete_treatment=True,
    cv=5,
    n_estimators=500,
    min_samples_leaf=20,
    random_state=42,
)

est.fit(Y=df['churn'].values,
        T=df['received_discount'].values,
        X=X_features,         # effect modifiers
        W=W_confounders)      # pure confounders (optional split)

cate_hat = est.effect(X_features)                 # shape (n,)
cate_lb, cate_ub = est.effect_interval(X_features, alpha=0.05)
```

> `CausalForestDML` is preferred over `LinearDML` because the problem explicitly requires *heterogeneous* effects. If interpretability of `τ(X)` itself is needed, follow up with `est.shap_values(X)` (EconML exposes SHAP natively for the final stage).

### 4.3 Sign convention & Persuadable isolation
Since $Y=1$ means churn, a *useful* discount should make $Y$ **smaller**, so the desired CATE is **negative**. Define an **uplift score**:

```
uplift_i = -cate_hat_i                  # positive = good (discount reduces churn)
```

Segment customers using both `p_churn` and `uplift`:

| Segment | Condition | Action |
|---|---|---|
| **Persuadable** | `p_churn high` **and** `uplift > +ε` | Eligible for ILP optimization |
| **Sure Thing** | `p_churn low` **and** `uplift ≈ 0` | Skip |
| **Lost Cause** | `p_churn high` **and** `uplift ≈ 0` | Skip |
| **Sleeping Dog** | `uplift < -ε` (treatment *increases* churn) | **Explicitly exclude** |

Thresholds `ε` can be set at e.g. the 95% confidence band from `effect_interval`.

### 4.4 Validation
- **Refutation tests via DoWhy:** `placebo_treatment_refuter`, `random_common_cause`, `data_subset_refuter`.
- **Qini / uplift curve** on a held-out fold — area under the Qini curve is the canonical uplift metric.
- If the synthetic dataset was used (Section 1), compare `cate_hat` to ground-truth `τ(X)` (MSE, Spearman ρ).

**Output of this layer per customer:** `cate_i` (signed) and a `segment` label.

---

## 5. Layer 3 — ROI Maximizer (PuLP ILP / 0-1 Knapsack)

**Module:** `src/optiretain/optimize/knapsack.py`

### 5.1 Prepare optimization inputs
For each customer `i` in the **Persuadable** pool (and only that pool):

| Input | Source |
|---|---|
| `cate_i` (uplift in churn-reduction prob.) | EconML output, sign-flipped so larger = better |
| `clv_i` | `MonthlyCharges_i × expected_remaining_lifetime_i` |
| `cost_i` | `discount_pct × MonthlyCharges_i × discount_duration_months` |
| `value_i` (net coefficient) | `cate_i × clv_i − cost_i` |

### 5.2 PuLP formulation

```
import pulp

prob = pulp.LpProblem("OptiRetain_Allocation", pulp.LpMaximize)

# Decision variables: x_i ∈ {0,1}
x = {i: pulp.LpVariable(f"x_{i}", cat='Binary') for i in customer_ids}

# Objective: maximize Σ x_i · (CATE_i · CLV_i − Cost_i)
prob += pulp.lpSum(x[i] * (cate[i] * clv[i] - cost[i]) for i in customer_ids)

# Budget constraint: Σ x_i · Cost_i ≤ Budget
prob += pulp.lpSum(x[i] * cost[i] for i in customer_ids) <= BUDGET, "Budget"

# (optional) Fairness / segment caps:
# prob += pulp.lpSum(x[i] for i in high_value_segment) >= MIN_TREATED

prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=120))
```

### 5.3 Reading the solution
```
selected = [i for i in customer_ids if pulp.value(x[i]) == 1]
total_lift = pulp.value(prob.objective)
budget_used = sum(cost[i] for i in selected)
```

- `x_i = 1` → customer enters the **"Recommended for Intervention"** list exported to `customers.json` with `recommended=true`.
- `x_i = 0` → customer kept in the dashboard but flagged `recommended=false` (so analysts can see *why* they were not chosen — drill into their CATE/CLV/cost trio).

### 5.4 Practical notes
- Pre-filter rows with `value_i ≤ 0` before constructing the LP — they will never be selected and just slow the solver.
- For >10⁵ customers, CBC may be slow; the LP relaxation gives a tight upper bound, and a greedy `value_i / cost_i` ratio heuristic is within a few % of optimal — useful as a fallback or warm-start.
- Expose `BUDGET` and `discount_pct` as dashboard sliders; re-solving CBC for ~10k Persuadables takes <1s, enabling interactive scenario analysis.

---

## 6. End-to-End Pipeline Architecture & Data Flow

`src/optiretain/pipeline.py` orchestrates a single CLI command (`python -m optiretain.pipeline --budget 50000 --discount 0.20`):

```
┌────────────────────┐
│ 1. data/loader.py  │  Telco CSV → DataFrame
└─────────┬──────────┘
          ▼
┌────────────────────┐
│ 2. data/features.py│  Clean, encode, compute CLV, synthesize T
└─────────┬──────────┘
          ▼
   ┌──────┴──────┐
   ▼             ▼
┌─────────┐  ┌───────────────────┐
│ Risk    │  │ Uplift Engine     │
│ Radar   │  │ (EconML DML)      │
│ (XGB)   │  │  → cate_i         │
│ → p_i   │  │  → segment_i      │
│ + SHAP  │  │                   │
└────┬────┘  └─────────┬─────────┘
     │                 │
     └────────┬────────┘
              ▼
   ┌─────────────────────┐
   │ 3. optimize/        │
   │    knapsack.py      │   inputs: cate, clv, cost, budget
   │    (PuLP ILP)       │   output: x_i ∈ {0,1}
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ 4. export/          │   merges p_i, cate_i, segment,
   │    dashboard_json.py│   shap_drivers, x_i → customers.json
   └──────────┬──────────┘
              ▼
   ┌─────────────────────┐
   │ 5. dashboard/       │   static HTML/JS reads customers.json
   │    index.html       │   – summary KPIs (waste ↓, revenue ↑)
   │    + app.js         │   – per-customer SHAP waterfall
   └─────────────────────┘
```

### Run order & artifacts
| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `data/loader.py` | raw CSV | `data/interim/telco.parquet` |
| 2 | `data/features.py` | interim | `data/processed/features.parquet` |
| 3 | `risk/train_xgb.py` | processed | `models/risk_radar.pkl` |
| 4 | `risk/explain_shap.py` | processed + model | `data/processed/shap_values.parquet` |
| 5 | `uplift/dml_cate.py` | processed | `models/dml_cate.pkl` + `cate.parquet` |
| 6 | `uplift/segmentation.py` | p_churn + cate | `segments.parquet` |
| 7 | `optimize/knapsack.py` | cate + clv + cost + budget | `allocation.parquet` |
| 8 | `export/dashboard_json.py` | all of above | `dashboard/customers.json` |

### Testing strategy (`tests/`)
- `test_features.py`: schema and CLV calculation invariants.
- `test_risk_model.py`: AUC ≥ 0.82 on held-out fold; calibration Brier ≤ 0.18.
- `test_dml.py`: on synthetic data, recovered CATE correlates ρ ≥ 0.7 with ground truth.
- `test_knapsack.py`: budget constraint never violated; greedy ratio heuristic ≤ ILP objective; LP behaves correctly when `BUDGET = 0` (no one selected) and `BUDGET = ∞` (everyone with `value_i > 0` selected).
- `test_pipeline_smoke.py`: end-to-end on a 500-row sample completes <60s.

---

## 7. Key Engineering Considerations

1. **Probability calibration is non-negotiable.** Layer 3 multiplies model outputs by dollar amounts; uncalibrated probabilities will systematically over- or under-spend.
2. **Train Risk Radar and DML on disjoint folds** if you want to feed `p_churn` *into* DML as a feature — otherwise leakage inflates CATE estimates.
3. **Sign discipline.** Document everywhere that `Y=1` is the *bad* outcome. CATE is negative for helpful treatments; the optimizer must consume `−cate` (or equivalently flip the sign at export time).
4. **Sleeping Dogs filter must precede the ILP.** Including customers with positive CATE-on-churn allows the optimizer to "spend" on them whenever `cost_i` is small — corrupting the entire allocation. Hard-filter at the segmentation step.
5. **Reproducibility:** fix `random_state=42` across XGBoost, EconML, sklearn nuisance models, and stratified splits. Store hashes of `features.parquet` alongside model artifacts.
6. **Dashboard is read-only static.** All heavy lifting happens in Python; the JS app only renders `customers.json`. This keeps the front end deployable as a single folder (GitHub Pages compatible) and avoids re-implementing model logic in the browser.

---

This plan is implementation-ready: each module has a defined input contract, output artifact, and validation criterion, and the four required technical pillars (XGBoost Risk Radar, SHAP explanations, EconML DML CATE, PuLP ILP) are wired into a single deterministic pipeline.

Want me to scaffold the directory structure and stub files next, or start by implementing a specific layer (e.g., `risk/train_xgb.py` against the Telco dataset)?
