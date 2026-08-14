# Module 2 — Analytics Pipeline

## Design notes (Sprint 1)

### Notebook vs script split

| Artifact | Purpose |
|----------|---------|
| `run_analytics.py` | Primary deliverable: EDA, visualizations, interpretations, ML experiments (executable Python script) |
| `cache_titanic.py` | One-time loader: `sns.load_dataset('titanic')` → `titanic.csv` |
| `titanic.csv` | Cached cleaned dataset — all script reads from here, never reload raw |

### Data flow

```
sns.load_dataset('titanic')  ──once──►  titanic.csv  ──always──►  notebook / scripts
```

### Part A (Sprint 3)

- Profile dtypes, missing values, duplicates
- Clean: drop duplicates, impute/fill age & embarked, encode categoricals for analysis
- EDA: survival rates by class/sex/embarked, age distributions
- Correlation heatmap (numeric features)
- ≥4 interpreted charts with Markdown cell commentary
- Standardization check (mean≈0, std≈1 after StandardScaler)

### Part B (Sprint 4)

- **Split:** stratified train/test on `survived`
- **Pipeline:** `ColumnTransformer` (numeric: impute+scale; categorical: impute+one-hot) — fit on train only
- **Classifiers:** Decision Tree (with `plot_tree`), Random Forest, Logistic Regression
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, confusion matrix
- **Imbalance:** compare class weights vs SMOTE (or stratified baseline)
- **Tuning:** `GridSearchCV` on RF + OOB score
- **Regression side-task:** predict `fare` with linear regression in same pipeline pattern
- **Persist:** `joblib.dump` / reload and verify predictions match

### Leakage guardrails

- All preprocessing inside `Pipeline` / `ColumnTransformer` fit on **train only**
- Test set never seen during imputation, scaling, or feature selection
