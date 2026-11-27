# Processed Dataset Splitting Strategy and Output Specification

**Target Directory:** `dataset/processed/`  
**Producing Notebook:** `notebooks/02_data_preprocessing.ipynb`  
**Consuming Notebooks:** `notebooks/03_model_training.ipynb`, `notebooks/04_model_evaluation.ipynb`  
**Reference Courses:** `Data Preprocessing.pdf` (Sections 4–5), `Machine learning.pdf` (Section 5)

---

## 1. What Needs to Be Implemented

Define the leakage-free **Train / Validation / Test** splitting protocol, transformation fit/transform rules, and exact file schema contracts stored inside `dataset/processed/` so Notebooks 03 and 04 consume consistent, verified feature matrices and target vectors.

---

## 2. Why This Specification Is Necessary

1. **Prevents Data Leakage:** If `StandardScaler` ($\mu, \sigma$), `OneHotEncoder` categories, or rare-brand groupings are computed on the full dataset *before* splitting, test-set statistics leak into training. Splitting **first** and fitting transformers **only on `X_train`** guarantees unbiased test evaluation.
2. **Supports Hyperparameter Tuning Safely:** `Machine learning.pdf` requires comparing **Linear Regression**, **Ridge Regression** ($\alpha$), **Lasso Regression** ($\alpha$), and **Decision Tree Regressor** (`max_depth`, `min_samples_leaf`). Having a dedicated **Validation** split (70% Train / 15% Validation / 15% Test) lets you tune $\alpha$ and tree depth without touching the held-out **Test** set.
3. **Supports Final Retraining (`Machine learning.pdf` §5):** Saving `X_train`, `X_val`, and `X_test` separately enables retraining the chosen best model on `Train + Validation` before final deployment in Notebook 04, while keeping `Test` strictly untouched.
4. **Decouples Notebooks 02, 03, and 04:** Saving clean CSVs and the fitted `preprocessor.joblib` allows Notebooks 03 and 04 to run independently and reproducibly.

---

## 3. Data Splitting Strategy (70% / 15% / 15%)

### Prerequisite Before Splitting
Exact duplicates **must already be removed** via `df.drop_duplicates()` in Notebook 02 before calling `train_test_split`. Otherwise, identical cars appear across Train, Validation, and Test sets.

### Two-Stage Split Protocol (`random_state=42`)
1. **Stage 1 (Hold out Test Set — 15%):**
   - Split cleaned unique dataset (`X, y`) into **85% Development (`Train + Val`)** and **15% Held-out Test (`Test`)**.
   - Optional stratification on binned `selling_price` quartiles (`pd.qcut(y, q=4, labels=False)`) ensures low-, mid-, and high-priced cars are evenly represented in all splits.
2. **Stage 2 (Split Development into Train & Validation):**
   - Split the 85% Development set with `test_size = 0.15 / 0.85` ($\approx 0.1765$) so overall proportions are:
     - **Train:** ~70%
     - **Validation:** ~15%
     - **Test:** ~15%

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Create price-quartile bins for stratified regression splitting
price_bins = pd.qcut(y, q=4, labels=False)

X_dev, X_test, y_dev, y_test, bins_dev, _ = train_test_split(
    X, y, price_bins,
    test_size=0.15,
    random_state=42,
    stratify=price_bins
)

X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.15 / 0.85,
    random_state=42,
    stratify=bins_dev
)
```

---

## 4. Fitted Preprocessing Pipeline Contract (`ColumnTransformer`)

Only `X_train` is passed to `.fit_transform()`. `X_val` and `X_test` receive `.transform()` only:

| Feature Group | Columns | Transformer | Rationale (`Data Preprocessing.pdf`) |
| :--- | :--- | :--- | :--- |
| **Numerical** | `car_age`, `km_driven` | `StandardScaler()` | Centers to mean `0`, std `1`; prevents `km_driven` ($10^5$) from dominating `car_age` ($10^1$) in Ridge/Lasso penalties. |
| **Ordinal Categorical** | `owner` | `OrdinalEncoder(categories=[["Test Drive Car", "First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]])` + `StandardScaler()` | Preserves natural ownership progression while matching scaled magnitude. |
| **Nominal Categorical** | `fuel`, `seller_type`, `transmission`, `brand` | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` | Converts unordered labels into binary dummy columns without implying numeric order. |

---

## 5. Required Output Files in `dataset/processed/`

| File Path | Contents | Purpose |
| :--- | :--- | :--- |
| `dataset/processed/car_cleaned.csv` | Deduplicated, anomaly-checked dataframe with `brand` & `car_age` (pre-encoding) | Human-readable audit & inspection |
| `dataset/processed/X_train.csv` | Transformed training features (float matrix with named columns) | Training in Notebook 03 |
| `dataset/processed/X_val.csv` | Transformed validation features | Hyperparameter tuning in Notebook 03 |
| `dataset/processed/X_test.csv` | Transformed test features | Final evaluation in Notebook 04 |
| `dataset/processed/y_train.csv` | Training target (`selling_price` in INR) | Training in Notebook 03 |
| `dataset/processed/y_val.csv` | Validation target (`selling_price` in INR) | Hyperparameter tuning in Notebook 03 |
| `dataset/processed/y_test.csv` | Test target (`selling_price` in INR) | Final evaluation in Notebook 04 |
| `dataset/processed/raw_train_dev.csv` | Unencoded `X_dev` + `y_dev` (85% `Train + Val`) | Retraining best full pipeline in Notebook 04 |
| `dataset/processed/preprocessor.joblib` | Fitted `ColumnTransformer` (fit on `X_train`) | Inference transformation for new cars |

---

## 6. Post-Split Validation Criteria

Run these checks at the end of Notebook 02 before proceeding to Notebook 03:

```python
assert len(X_train) + len(X_val) + len(X_test) == len(df_cleaned)
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
assert X_train.isnull().sum().sum() == 0
assert X_val.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0
assert len(y_train) == len(X_train)
assert len(y_val) == len(X_val)
assert len(y_test) == len(X_test)

# Verify StandardScaler centered training numericals (~0 mean, ~1 std)
assert abs(X_train["km_driven"].mean()) < 1e-6
assert abs(X_train["car_age"].mean()) < 1e-6
```
