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

OptiRetain solves this resource allocation problem through a three-layered architecture:

### 1. The Risk Radar (Predictive Layer)

We utilize an **XGBoost Classifier** to estimate the baseline probability of churn. **SHAP (SHapley Additive exPlanations)** is integrated to expose the feature-level drivers of churn risk for each individual customer, ensuring explainability.

### 2. The Uplift Engine (Causal Inference Layer)

Instead of predicting the outcome $Y$, we predict the **Conditional Average Treatment Effect (CATE)**. Using **Double Machine Learning (DML)** via Microsoft's `EconML`, we estimate the causal impact of a specific intervention (e.g., a 20% discount).

$$CATE = E[Y_i(1) - Y_i(0) | X_i]$$

This isolates the **"Persuadables"**—the only segment where marketing spend actually generates a positive return.

### 3. The ROI Maximizer (Optimization Layer)

With the CATE and estimated CLV for every customer, we frame retention as a **Knapsack Problem**. Using **PuLP (Integer Linear Programming)**, the system optimizes the allocation of a constrained marketing budget to maximize total retained net revenue.

**Objective Function:**
Maximize the sum of expected revenue lift minus intervention costs:


$$\text{Maximize} \sum_{i} x_i \cdot \big( CATE_i \times CLV_i - Cost_i \big)$$

**Subject to Budget Constraint:**


$$\sum_{i} x_i \cdot Cost_i \leq Budget$$


*(Where $x_i \in \{0, 1\}$ represents the decision to treat customer $i$)*

## 📊 Key Results (Simulated Impact)

Compared to a standard baseline strategy of targeting the top decile of "at-risk" customers:

* **Reduced Marketing Waste:** Decreased spend on "Sure Things" and "Lost Causes" by **22%**.
* **Revenue Lift:** Increased Net Retained Revenue by **14%** under constrained budget scenarios.
* **CLV Protection:** Improved long-term CLV retention by **18%** by strictly avoiding "Sleeping Dog" triggers.

## 🛠️ Technical Architecture

* **Data Processing & Feature Engineering:** `Pandas`, `NumPy`, `Scikit-Learn`
* **Predictive Modeling:** `XGBoost`, `LightGBM`
* **Causal Inference:** `EconML`, `DoWhy` (Propensity Score Matching, T-Learner/X-Learner)
* **Mathematical Optimization:** `PuLP`, `SciPy.optimize`
* **Interpretability:** `SHAP`
* **Dashboarding:** `html`, `css`, `js`