# Agriculture Crop Yield — Predictive Modeling

## Objective
Build a **predictive model** that estimates crop yield (tons/ha) from agronomic, environmental, and management features. The goal is to provide **actionable, explainable insights** that help stakeholders (farm managers, planners, and researchers) optimize inputs and anticipate production under varying conditions.

## Dataset
- **Location:** `capstone/dataset/crop_yield.csv` (as referenced in the notebook)
- **Records:** ~1,000,000 crop instances (large-scale tabular data).
- **Target:** `Yield_tons_per_hectare`
- **Key Features (examples):**
  - **Categorical:** `Region` (North/East/South/West), `Soil_Type` (Clay/Sandy/Loam/Silt/Peaty/Chalky), `Crop` (Wheat/Rice/Maize/Barley/Soybean/Cotton)
  - **Numeric:** `Rainfall_mm`, `Temperature_Celsius`, `Days_to_Harvest`, `Fertilizer_Used`, `Irrigation_Used` (and related indicators)
- **Notes:** The notebook performs schema inspection (`df.info()`, `df.describe()`), checks for missing values/outliers, and applies **one‑hot encoding** to categorical fields before modeling.


## Analysis Workflow
### 1) Exploratory Data Analysis (EDA)
- Visualize **distributions** for numeric features (histograms/boxplots) to spot skewness and outliers.
- Analyze **categorical** variables (count plots) and their relationship to yield.
- Examine **relationships** (correlation matrix, scatter plots) between inputs and the target.

### 2) Preprocessing
- **One‑hot encoding** for categorical variables (`pd.get_dummies(..., drop_first=True)`).
- **Feature scaling** (Standardization) for models sensitive to feature scales.
- Train/Validation/Test split to support fair model selection and final evaluation.

### 3) Modeling
The notebook compares three families of models to balance simplicity, accuracy, and inference cost:

- **Linear Regression (baseline)**
  - Fast, interpretable coefficients; good when relationships are mostly linear.
- **XGBoost (tree‑based gradient boosting)**
  - Captures nonlinear effects and interactions; robust to outliers; strong tabular baseline.
- **Neural Network (MLP)**
  - Flexible function approximator; can capture complex patterns with adequate tuning and regularization.

### 4) Evaluation & Visualization
- Metrics on validation/test sets: **RMSE**, **MAE**, **R²**.
- **Cross‑validation** used (e.g., for Linear Regression) to gauge generalization.
- Plots included in the notebook:
  - **Learning curves** (where applicable).
  - **Actual vs. Predicted** scatter plots (Validation and Test).
  - **Side‑by‑side model comparison** (errors + R²) for a compact overview.

## Results Summary (High‑Level)
Across runs in the notebook, **Linear Regression, XGBoost, and MLP achieve comparable performance** on the test data. In scenarios where model performance is statistically similar, choosing the **simplest viable model** is recommended:
- Start with **Linear Regression** for speed, interpretability, and ease of maintenance.
- Prefer **XGBoost** when you expect **nonlinearities** or **feature interactions** (e.g., weather × soil).
- Consider **MLP** only if additional engineered features or larger datasets reveal patterns that tree‑based models cannot capture.

> Practical takeaway: If accuracy gains from complex models are marginal, **deploy Linear Regression** first, and graduate to XGBoost/MLP only when justified by measurable improvements or new requirements.

## Conclusion
This workflow demonstrates a **robust, end‑to‑end pipeline** for predicting crop yield:
1. Careful EDA to understand data quality and key drivers.
2. Solid preprocessing (encoding + scaling) tailored to tabular ML.
3. A **model comparison** that balances accuracy with interpretability and cost.
4. Clear guidance for **model selection**: default to the simplest model that meets performance targets, and iterate toward complexity only as needed.

## Reproducibility & Environment Notes
- **Random seed:** `42` (used throughout for splits and training reproducibility).
- **Hardware:** Notebook supports CPU‑only workflows. For XGBoost GPU inference/training, set parameters appropriately (e.g., `tree_method="gpu_hist"` if available; fallback to `hist` on CPU).
- **Dependencies:** `pandas`, `numpy`, `scikit‑learn`, `matplotlib`, `xgboost`, and (optionally) `torch` for MLP.
- **Known tip:** If GPU‑accelerated XGBoost causes issues in cross‑validation, **switch to CPU (`tree_method="hist"`)** for CV and back to GPU for the final fit.

## How to Use
1. Place your data at `capstone/dataset/crop_yield.csv` (or update the path in the first cell).
2. Run the notebook cells in order:
   - EDA → Preprocessing → Modeling → Evaluation → Comparison.
3. Review the **comparison plot and metrics** to pick the model that best matches your constraints (accuracy / latency / interpretability).
4. (Optional) Log experiments (e.g., with MLflow) to track metrics and artifacts for different seeds or feature sets.

## Repository Structure (suggested)
```
.
├── Agriculture Crop Yield.ipynb
├── capstone/
│   └── dataset/
│       └── crop_yield.csv
├── README.md  ← (this file)
└── requirements.txt
```

---

**License:** MIT (or update per your needs)
