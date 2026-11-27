# Notebook 04 Implementation Guide: Model Evaluation & Prediction Demo

**Target Notebook:** `notebooks/04_model_evaluation.ipynb`  
**Inputs:** `../models/*.joblib`, `../dataset/processed/X_test.csv`, `../dataset/processed/y_test.csv`, `../dataset/processed/raw_train_dev.csv`, `../dataset/processed/preprocessor.joblib`  
**Output:** Final evaluation report tables/charts, `../models/best_car_price_model.joblib`, and tested `predict_car_price()` function  
**Reference Courses:** `Machine learning.pdf` (Sections 5, 6, 7.5, 7.6), `Week-6 Data Visualization.pdf` (Section II.2 & V)

---

## 1. What Needs to Be Implemented

1. Evaluate all 4 trained models on the **unseen held-out Test Set** using **MAE**, **RMSE**, and **$R^2$**.
2. Present a clean side-by-side **Model Comparison Table** and visual metric charts.
3. Perform **Error & Residual Analysis** (Actual vs. Predicted scatter plots, Residual plots, Top largest errors).
4. Inspect **Feature Importance & Regularization Weights** (Linear/Ridge/Lasso coefficients vs. Decision Tree feature importances).
5. Document evidence-backed answers to all **5 required discussion questions** (`Machine learning.pdf` §7.6).
6. Retrain the winning model pipeline on the combined **`Train + Validation`** data (`raw_train_dev.csv`) per `Machine learning.pdf` §5, save `best_car_price_model.joblib`, and demonstrate new-car price prediction.

---

## 2. Evaluation Formulas (`Machine learning.pdf` §6)

Let $y^{(i)}$ be the actual selling price, $\hat{y}^{(i)}$ the predicted price, $\bar{y}$ the mean actual price of the test set, and $m$ the number of test samples:

1. **Mean Absolute Error (MAE):** Average absolute INR error:
   $$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} \left| y^{(i)} - \hat{y}^{(i)} \right|$$
2. **Root Mean Squared Error (RMSE):** Penalizes large pricing mistakes more heavily:
   $$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2}$$
3. **Coefficient of Determination ($R^2$):** Variance explained relative to predicting the average price ($\bar{y}$):
   $$R^2 = 1 - \frac{\sum_{i=1}^{m} \left( y^{(i)} - \hat{y}^{(i)} \right)^2}{\sum_{i=1}^{m} \left( y^{(i)} - \bar{y} \right)^2}$$
   - $R^2 \approx 1$: strong fit; $R^2 \approx 0$: no better than mean baseline; $R^2 < 0$: worse than mean baseline.

---

## 3. Dependencies and Prerequisites

1. Completed `03_model_training.ipynb` with all 4 models saved in `../models/`.
2. Untouched `X_test.csv`, `y_test.csv`, `raw_train_dev.csv`, and `preprocessor.joblib` in `../dataset/processed/`.
3. Python libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`.

---

## 4. Step-by-Step Implementation Instructions

### Step 1: Load Models and Held-Out Test Data

```python
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import clone

X_test = pd.read_csv("../dataset/processed/X_test.csv")
y_test = pd.read_csv("../dataset/processed/y_test.csv").squeeze()

models = {
    "Linear Regression": joblib.load("../models/linear_regression.joblib"),
    "Ridge Regression": joblib.load("../models/ridge_regression.joblib"),
    "Lasso Regression": joblib.load("../models/lasso_regression.joblib"),
    "Decision Tree Regressor": joblib.load("../models/decision_tree_regressor.joblib"),
}
print("Loaded test set:", X_test.shape, "and", len(models), "models.")
```

---

### Step 2: Compute Test-Set MAE, RMSE, and $R^2$ Comparison Table (`Machine learning.pdf` §7.5)

```python
eval_rows = []
predictions = {}

for name, model in models.items():
    preds = model.predict(X_test)
    predictions[name] = preds
    eval_rows.append({
        "Model": name,
        "MAE (INR)": mean_absolute_error(y_test, preds),
        "RMSE (INR)": root_mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    })

comparison_df = pd.DataFrame(eval_rows).sort_values("RMSE (INR)").reset_index(drop=True)
display(comparison_df.style.format({"MAE (INR)": "{:,.0f}", "RMSE (INR)": "{:,.0f}", "R2": "{:.4f}"}))
```

---

### Step 3: Visualize Model Comparison & Residual Diagnostics (`Week-6 Data Visualization.pdf`)
1. **Metric Comparison Bar Charts:** Side-by-side subplots comparing MAE, RMSE, and $R^2$ across the 4 models.
2. **Actual vs. Predicted Scatter Plot (with $y = x$ ideal diagonal):** Reveals whether linear models predict negative prices for very old/high-km cars or underpredict luxury cars.
3. **Residual Plot ($y - \hat{y}$ vs. $\hat{y}$):** Checks for heteroscedasticity (larger errors on expensive cars).

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
sns.barplot(data=comparison_df, x="MAE (INR)", y="Model", ax=axes[0])
axes[0].set_title("Test MAE (Lower is Better)")

sns.barplot(data=comparison_df, x="RMSE (INR)", y="Model", ax=axes[1])
axes[1].set_title("Test RMSE (Lower is Better)")

sns.barplot(data=comparison_df, x="R2", y="Model", ax=axes[2])
axes[2].set_title("Test R² (Higher is Better)")
plt.tight_layout()
plt.show()

best_name = comparison_df.loc[0, "Model"]
best_preds = predictions[best_name]
residuals = y_test - best_preds

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_test, best_preds, alpha=0.45)
lims = [0, max(y_test.max(), best_preds.max())]
axes[0].plot(lims, lims, "r--", label="Ideal Fit (y = x)")
axes[0].set_title(f"Actual vs. Predicted ({best_name})")
axes[0].set_xlabel("Actual Selling Price (INR)")
axes[0].set_ylabel("Predicted Selling Price (INR)")
axes[0].legend()

axes[1].scatter(best_preds, residuals, alpha=0.45)
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title(f"Residual Plot ({best_name})")
axes[1].set_xlabel("Predicted Selling Price (INR)")
axes[1].set_ylabel("Residual (Actual - Predicted, INR)")
plt.tight_layout()
plt.show()
```

---

### Step 4: Feature Importance & Error Inspection
Inspect which features drive predictions in both linear models and the Decision Tree, and print the Top 10 largest test errors to diagnose dataset limitations:

```python
feature_names = X_test.columns

# Linear / Ridge / Lasso coefficients
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Linear": models["Linear Regression"].coef_,
    "Ridge": models["Ridge Regression"].coef_,
    "Lasso": models["Lasso Regression"].coef_,
})
coef_df["Abs_Ridge"] = coef_df["Ridge"].abs()
display(coef_df.sort_values("Abs_Ridge", ascending=False).head(15))

# Decision Tree importances
dt_imp = pd.Series(
    models["Decision Tree Regressor"].feature_importances_,
    index=feature_names
).sort_values(ascending=False).head(12)

plt.figure(figsize=(8, 5))
sns.barplot(x=dt_imp.values, y=dt_imp.index)
plt.title("Top 12 Feature Importances — Decision Tree Regressor")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
```

---

### Step 5: Structured Discussion & Error Analysis (`Machine learning.pdf` §7.6)
Add a dedicated Markdown section answering all **5 required questions** using exact numbers observed in Steps 2–4:

1. **Which features appear most useful for predicting car prices?**  
   Analyze top Ridge/Lasso coefficients and Decision Tree importances (typically `car_age`, luxury `brand` indicators such as Mercedes-Benz/BMW/Audi/Toyota, `transmission_Automatic`, `fuel_Diesel`, and `km_driven`).
2. **How do Ridge and Lasso regularization affect model performance?**  
   Compare MAE/RMSE/$R^2$ vs. unregularized Linear Regression; report how Ridge stabilizes correlated dummy weights and how many features Lasso shrunk to zero.
3. **Does the Decision Tree capture relationships that the linear models miss?**  
   Explain non-linear depreciation with `car_age` and brand × transmission/fuel interactions that a single global hyperplane cannot represent (compare RMSE/$R^2$ directly).
4. **What are the main prediction errors and limitations?**  
   Report largest residuals (typically high-value luxury vehicles or specific trim variants), negative-price artifacts in linear models if any, and absence of trim/engine/present-price features after dropping raw `name`.
5. **How could the project be improved by adding location, car condition, brand, or service history?**  
   Note that `brand` was already engineered from `name` in this project, whereas engine displacement, original showroom price (`Present_Price`), accident/service history, city/location, and physical condition would directly reduce unexplained variance.

---

### Step 6: Retrain Best Model on Full Development Data (`Train + Val`) & Run Prediction Demo
Follow `Machine learning.pdf` §5 (*"After comparing the models, the chosen model can be retrained on the full training dataset before being used for predictions"*). See [05_model_artifacts_and_prediction_demo.md](../../models/05_model_artifacts_and_prediction_demo.md) for the exact retraining and inference wrapper code saved to `../models/best_car_price_model.joblib`.

---

## 5. Expected Outputs & Validation Criteria

- Comparison table reports **MAE (INR), RMSE (INR), and $R^2$** for all 4 models on the untouched test set.
- All charts follow `Week-6 Data Visualization.pdf` principles (clear titles, INR unit labels, no chartjunk).
- All 5 discussion questions from `Machine learning.pdf` §7.6 are answered with observed experimental evidence.
- Final pipeline `../models/best_car_price_model.joblib` is saved and tested on at least 2 realistic sample cars.
