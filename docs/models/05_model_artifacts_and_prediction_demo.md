# Model Artifacts & Prediction Demo Guide (`models/`)

**Target Directory:** `models/`  
**Producing Notebooks:** `notebooks/03_model_training.ipynb`, `notebooks/04_model_evaluation.ipynb`  
**Reference Course:** `Machine learning.pdf` (Sections 5 & 6)

---

## 1. What Needs to Be Implemented

1. Define the serialized model inventory stored inside `models/`.
2. Retrain the winning model architecture (selected on Test evaluation in Notebook 04) inside a unified scikit-learn `Pipeline(preprocessor + estimator)` on the full development dataset (`raw_train_dev.csv`, i.e., `Train + Validation`, keeping `Test` untouched).
3. Persist `models/best_car_price_model.joblib` and expose a validated `predict_car_price(...)` function accepting raw human-readable car inputs and returning predicted selling price in **INR** (with an explicit optional INR-to-USD conversion helper).

---

## 2. Why This Step Is Necessary

- **Reproducibility & Deployment Readiness:** Separating individual model artifacts (from Notebook 03) from the unified end-to-end inference pipeline (`preprocessor + best_estimator` in `best_car_price_model.joblib`) prevents preprocessing mismatches at prediction time.
- **Maximizes Training Signal (`Machine learning.pdf` §5):** Once model selection is complete, retraining on `Train + Validation` (85% of unique records instead of 70%) gives the final estimator more historical examples before predicting prices for unseen cars.
- **Currency Consistency (`Machine learning.pdf` §2 & §6):** Ensures predicted outputs clearly state **INR**, avoiding silent INR/USD confusion.

---

## 3. Model Inventory in `models/`

| File Path | Created In | Contents | Purpose |
| :--- | :--- | :--- | :--- |
| `models/linear_regression.joblib` | Notebook 03 | Fitted `LinearRegression` | Baseline evaluation in NB 04 |
| `models/ridge_regression.joblib` | Notebook 03 | Tuned `Ridge(alpha=...)` | Regularized $L_2$ evaluation in NB 04 |
| `models/lasso_regression.joblib` | Notebook 03 | Tuned `Lasso(alpha=...)` | Regularized $L_1$ evaluation in NB 04 |
| `models/decision_tree_regressor.joblib` | Notebook 03 | Tuned `DecisionTreeRegressor(...)` | Non-linear evaluation in NB 04 |
| `models/best_car_price_model.joblib` | Notebook 04 | Full `Pipeline([("preprocessor", ...), ("model", best_estimator)])` fit on `raw_train_dev.csv` | Production / Demo inference via `predict_car_price()` |

---

## 4. Step-by-Step Retraining & Prediction Function (Used in Notebook 04)

### Step 1: Retrain Winning Pipeline on `raw_train_dev.csv`

```python
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.base import clone

preprocessor = joblib.load("../dataset/processed/preprocessor.joblib")
raw_dev = pd.read_csv("../dataset/processed/raw_train_dev.csv")

X_dev = raw_dev.drop(columns=["selling_price"])
y_dev = raw_dev["selling_price"]

# Example: select winner dynamically from Notebook 04 comparison_df
winner_name = comparison_df.loc[0, "Model"]
winner_estimator = clone(models[winner_name])

final_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("regressor", winner_estimator),
])
final_pipeline.fit(X_dev, y_dev)

joblib.dump(final_pipeline, "../models/best_car_price_model.joblib")
print(f"Retrained and saved final pipeline ({winner_name}) on {len(X_dev)} records.")
```

---

### Step 2: Implement `predict_car_price()` Function
Maps raw car attributes directly into the exact schema expected by `preprocessor` (`car_age`, `km_driven`, `owner`, `fuel`, `seller_type`, `transmission`, `brand`):

```python
REFERENCE_YEAR = 2020

def predict_car_price(
    car_name: str,
    year: int,
    km_driven: int,
    fuel: str,
    seller_type: str,
    transmission: str,
    owner: str,
    model_path: str = "../models/best_car_price_model.joblib",
    inr_per_usd: float | None = None,
) -> dict:
    """Predict used car selling price in INR (and optionally USD) from raw car specs."""
    if year > REFERENCE_YEAR or year < 1980:
        raise ValueError(f"year must be between 1980 and {REFERENCE_YEAR}, got {year}")
    if km_driven < 0:
        raise ValueError(f"km_driven must be non-negative, got {km_driven}")

    brand = car_name.strip().split()[0]
    car_age = REFERENCE_YEAR - year

    input_df = pd.DataFrame([{
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
        "car_age": car_age,
    }])

    pipeline = joblib.load(model_path)
    pred_inr = float(max(0.0, pipeline.predict(input_df)[0]))

    result = {"predicted_price_inr": round(pred_inr, 2)}
    if inr_per_usd is not None and inr_per_usd > 0:
        result["predicted_price_usd"] = round(pred_inr / inr_per_usd, 2)
    return result
```

---

### Step 3: Smoke-Test Inference Demo

```python
sample_1 = predict_car_price(
    car_name="Maruti Swift VDI",
    year=2017,
    km_driven=40000,
    fuel="Diesel",
    seller_type="Individual",
    transmission="Manual",
    owner="First Owner",
)
sample_2 = predict_car_price(
    car_name="Hyundai Creta 1.6 SX",
    year=2018,
    km_driven=25000,
    fuel="Petrol",
    seller_type="Dealer",
    transmission="Automatic",
    owner="First Owner",
    inr_per_usd=84.0,
)
print("Sample 1:", sample_1)
print("Sample 2:", sample_2)
```

---

## 5. Validation Criteria

- All 5 `.joblib` artifacts exist in `models/` and load without errors.
- `predict_car_price()` handles unseen brands gracefully (via `OneHotEncoder(handle_unknown="ignore")`), rejects invalid negative `km_driven` or out-of-range `year`, and outputs non-negative INR estimates.
- Predicted values align reasonably with market ranges observed in Notebook 01 (e.g., newer Automatic cars > older high-km Manual cars).
