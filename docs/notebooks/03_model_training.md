# Notebook 03 Implementation Guide: Model Training

**Target Notebook:** `notebooks/03_model_training.ipynb`  
**Input Directory:** `../dataset/processed/` (`X_train.csv`, `y_train.csv`, `X_val.csv`, `y_val.csv`)  
**Output Directory:** `../models/`  
**Reference Course:** `Machine learning.pdf` (Sections 4 & 5: Classical Regression Algorithms & Python Implementation)

---

## 1. What Needs to Be Implemented

Train, tune on the Validation split, compare Training vs. Validation errors, and persist all **four classical regression models** specified in `Machine learning.pdf` §4:
1. **Linear Regression** (OLS Baseline)
2. **Ridge Regression** ($L_2$ Regularization)
3. **Lasso Regression** ($L_1$ Regularization)
4. **Decision Tree Regressor** (Non-Linear Tree-Based Regression)

---

## 2. Why Each Model Is Included (Algorithm Theory & Formulas)

### 2.1 Linear Regression (Baseline)
Predicts used car price as a weighted linear combination of input features:
$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n$$
Learns weights $w$ by minimizing Ordinary Least Squares (MSE). Simple, interpretable baseline, but sensitive to multicollinearity (correlated features such as One-Hot `brand` / `fuel` dummies) and unable to capture non-linear price depreciation curves.

### 2.2 Ridge Regression ($L_2$ Regularization)
Adds an $L_2$ penalty proportional to the sum of squared weights:
$$J_{\text{Ridge}}(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2 + \alpha \sum_{j=1}^{n} w_j^2$$
- **Role of $\alpha$:** Controls regularization strength. Larger $\alpha$ shrinks weights closer to zero without eliminating them, stabilizing predictions when features are correlated and reducing overfitting.

### 2.3 Lasso Regression ($L_1$ Regularization)
Adds an $L_1$ penalty proportional to the sum of absolute weights:
$$J_{\text{Lasso}}(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2 + \alpha \sum_{j=1}^{n} |w_j|$$
- **Feature Selection Property (`Machine learning.pdf` §4.3):** Unlike Ridge, $L_1$ can drive unimportant feature weights strictly to **zero**, highlighting which brands, fuel types, or features contribute least to predicting `selling_price`.

### 2.4 Decision Tree Regressor (Non-Linear Model)
Recursively partitions the feature space using threshold rules (e.g., `car_age <= 4.5`, `brand_BMW == 1`, `km_driven <= 55000`) to minimize within-leaf variance (MSE).
- **Strength (`Machine learning.pdf` §4.4):** Captures **non-linear relationships** and **feature interactions** (e.g., luxury brands depreciating at different rates than economy brands).
- **Overfitting Risk:** An unconstrained tree (`max_depth=None`) memorizes training samples ($R^2_{\text{train}} \approx 1.0$) and generalizes poorly. Tuning `max_depth` and `min_samples_leaf` on `X_val, y_val` is mandatory.

---

## 3. Dependencies and Prerequisites

1. Preprocessed splits saved in `../dataset/processed/` by Notebook 02.
2. Directory `../models/` exists.
3. Python libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`.

---

## 4. Step-by-Step Implementation Instructions

### Step 1: Load Processed Train & Validation Sets
Keep `X_test.csv` and `y_test.csv` untouched until Notebook 04!

```python
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

X_train = pd.read_csv("../dataset/processed/X_train.csv")
y_train = pd.read_csv("../dataset/processed/y_train.csv").squeeze()
X_val = pd.read_csv("../dataset/processed/X_val.csv")
y_val = pd.read_csv("../dataset/processed/y_val.csv").squeeze()

print("Train:", X_train.shape, "| Val:", X_val.shape)
assert X_train.isnull().sum().sum() == 0 and X_val.isnull().sum().sum() == 0
```

*(Compatibility note: If using scikit-learn `< 1.4`, replace `root_mean_squared_error(y, pred)` with `np.sqrt(mean_squared_error(y, pred))`.)*

---

### Step 2: Define a Reusable Metric Helper
Avoid duplicate metric-calculation logic across the four models:

```python
def evaluate_split(model, X_tr, y_tr, X_v, y_v, model_name):
    tr_pred = model.predict(X_tr)
    val_pred = model.predict(X_v)
    return {
        "Model": model_name,
        "Train_MAE": mean_absolute_error(y_tr, tr_pred),
        "Val_MAE": mean_absolute_error(y_v, val_pred),
        "Train_RMSE": root_mean_squared_error(y_tr, tr_pred),
        "Val_RMSE": root_mean_squared_error(y_v, val_pred),
        "Train_R2": r2_score(y_tr, tr_pred),
        "Val_R2": r2_score(y_v, val_pred),
    }
```

---

### Step 3: Train Model 1 — Linear Regression (Baseline)

```python
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

results = [evaluate_split(lr_model, X_train, y_train, X_val, y_val, "Linear Regression")]
```

---

### Step 4: Train & Tune Model 2 — Ridge Regression ($\alpha$ Search)
Sweep $\alpha$ over several orders of magnitude using the validation set to select the best $L_2$ penalty:

```python
ridge_alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
best_ridge, best_ridge_rmse, best_ridge_alpha = None, float("inf"), None

for alpha in ridge_alphas:
    candidate = Ridge(alpha=alpha, random_state=42)
    candidate.fit(X_train, y_train)
    val_rmse = root_mean_squared_error(y_val, candidate.predict(X_val))
    print(f"Ridge alpha={alpha:<6} -> Val RMSE: {val_rmse:,.2f}")
    if val_rmse < best_ridge_rmse:
        best_ridge_rmse = val_rmse
        best_ridge = candidate
        best_ridge_alpha = alpha

print(f"Best Ridge alpha: {best_ridge_alpha}")
results.append(evaluate_split(best_ridge, X_train, y_train, X_val, y_val, f"Ridge (alpha={best_ridge_alpha})"))
```

---

### Step 5: Train & Tune Model 3 — Lasso Regression ($\alpha$ Search + Zero-Weight Inspection)
Because target `selling_price` is in INR ($10^5$–$10^6$ scale), Lasso needs larger `alpha` values (e.g., `10` to `5000`) and higher `max_iter=20000` to converge cleanly:

```python
lasso_alphas = [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2500.0]
best_lasso, best_lasso_rmse, best_lasso_alpha = None, float("inf"), None

for alpha in lasso_alphas:
    candidate = Lasso(alpha=alpha, max_iter=20000, random_state=42)
    candidate.fit(X_train, y_train)
    val_rmse = root_mean_squared_error(y_val, candidate.predict(X_val))
    n_zero = (candidate.coef_ == 0).sum()
    print(f"Lasso alpha={alpha:<7} -> Val RMSE: {val_rmse:,.2f} | Zeroed features: {n_zero}/{len(candidate.coef_)}")
    if val_rmse < best_lasso_rmse:
        best_lasso_rmse = val_rmse
        best_lasso = candidate
        best_lasso_alpha = alpha

print(f"Best Lasso alpha: {best_lasso_alpha}")
results.append(evaluate_split(best_lasso, X_train, y_train, X_val, y_val, f"Lasso (alpha={best_lasso_alpha})"))
```

---

### Step 6: Train & Tune Model 4 — Decision Tree Regressor (`max_depth` & `min_samples_leaf`)
Compare an unconstrained tree against depth/leaf-constrained candidates to prevent overfitting:

```python
best_tree, best_tree_rmse, best_tree_params = None, float("inf"), None

for depth in [3, 5, 7, 10, 12, 15, None]:
    for min_leaf in [1, 3, 5, 10]:
        candidate = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=min_leaf,
            random_state=42
        )
        candidate.fit(X_train, y_train)
        val_rmse = root_mean_squared_error(y_val, candidate.predict(X_val))
        if val_rmse < best_tree_rmse:
            best_tree_rmse = val_rmse
            best_tree = candidate
            best_tree_params = (depth, min_leaf)

print(f"Best Decision Tree -> max_depth={best_tree_params[0]}, min_samples_leaf={best_tree_params[1]}")
results.append(evaluate_split(
    best_tree, X_train, y_train, X_val, y_val,
    f"Decision Tree (depth={best_tree_params[0]}, leaf={best_tree_params[1]})"
))
```

*(Optional visualization: Plot `Train R2` vs `Val R2` across `max_depth` to illustrate the bias-variance tradeoff cleanly per `Week-6 Data Visualization.pdf`.)*

---

### Step 7: Review Validation Summary & Save Trained Models to `../models/`

```python
val_summary = pd.DataFrame(results).sort_values("Val_RMSE")
display(val_summary.round(3))

joblib.dump(lr_model, "../models/linear_regression.joblib")
joblib.dump(best_ridge, "../models/ridge_regression.joblib")
joblib.dump(best_lasso, "../models/lasso_regression.joblib")
joblib.dump(best_tree, "../models/decision_tree_regressor.joblib")

print("Saved all 4 trained models to ../models/")
```

---

## 5. Expected Outputs & Validation Criteria

- All 4 models fit without convergence warnings.
- Validation table displays `Train` and `Val` MAE, RMSE, and $R^2$ for every model (confirming Decision Tree does not severely overfit).
- Four `.joblib` model files exist in `models/`:
  - `linear_regression.joblib`
  - `ridge_regression.joblib`
  - `lasso_regression.joblib`
  - `decision_tree_regressor.joblib`
- `X_test.csv` and `y_test.csv` were **never used** during training or hyperparameter selection.
