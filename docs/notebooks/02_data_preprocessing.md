# Notebook 02 Implementation Guide: Data Cleaning & Preprocessing

**Target Notebook:** `notebooks/02_data_preprocessing.ipynb`  
**Input Dataset:** `../dataset/raw/car.csv` (**4,000 rows × 8 columns**)  
**Output Directory:** `../dataset/processed/`  
**Reference Courses:** `Data Preprocessing.pdf` (§1–6), `Week4-Data Pre-processing.pdf` (§2), `Machine learning.pdf` (§5, §7.4)

---

## 1. What Needs to Be Implemented

Transform the raw 4,000-record CarDekho dataset into clean, encoded, scaled, leakage-free **Training, Validation, and Test** splits ready for classical regression algorithms.

Tasks in strict execution order (matching **`Data Preprocessing.pdf`** Sections 1–6):
1. **Types of Data & Variable Roles (§1–2):** Classify all columns as Nominal, Ordinal, Interval, or Ratio, and separate $X$ vs. $y$.
2. **Missing Values & Placeholders (§3.2):** Scan for `NaN` plus disguised placeholders (`NA`, `n/a`, `""`, `?`, `-1`, `-99`, `-999`, `null`) and evaluate against Vishal Patel thresholds (`<50%` impute / `>95%` discard), without blind `.dropna()`.
3. **Text Cleaning & Duplicate Removal:** Normalize strings (`name`) and remove exact duplicates (`drop_duplicates()`) before splitting.
4. **Outlier Diagnosis (§3.3) & Scope Note (§3.4):** Boxplot + Z-score ($|Z|>3$) + IQR ($[Q_1 - 1.5\cdot\text{IQR},\; Q_3 + 1.5\cdot\text{IQR}]$) analysis, targeted `km_driven < 100` anomaly filtering, and explicit exclusion of classification-only SMOTE (§3.4).
5. **Feature Engineering, Low Variation (§3.1) & Correlation Selection (§6):** Engineer `brand` and `car_age`, check `var(x)` / `std(x)` for zero-variance columns, inspect Pearson $r_{xy}$ heatmap, and drop redundant `year` and raw `name`.
6. **Leakage-Free Split:** **Train / Validation / Test** split (70% / 15% / 15%).
7. **Feature Encoding (§5) & Scaling (§4):** **Ordinal Encoding** for `owner`, **One-Hot Encoding** for `fuel`, `seller_type`, `transmission`, `brand`, and **Standardization** (`StandardScaler`) vs. **Min-Max Normalization** (`MinMaxScaler`) comparison.
8. **Persistence:** Save processed splits and fitted `preprocessor.joblib` to `../dataset/processed/`.

---

## 2. Why Each Step Is Necessary

- **Deduplication Before Splitting:** Exact duplicate rows appearing in both Train and Test sets cause **data leakage**, artificially inflating $R^2$ and misleading model comparison.
- **String Cleaning & `brand` Extraction (`Week4-Data Pre-processing.pdf` §2):** Raw `name` has 1,491+ unique strings; one-hot encoding raw `name` causes severe dimensionality explosion and overfitting. Extracting `brand` reduces cardinality to ~29 brands (and grouping rare brands reduces noise further).
- **`car_age` Engineering (`Machine learning.pdf` §5):** Converting `year` into `car_age = 2020 - year` gives a **Ratio** feature with a meaningful zero (`0` = newest 2020 model year) and makes linear weights directly interpretable (price depreciation per year of age). Dropping raw `year` afterward avoids perfect negative collinearity ($r = -1.0$).
- **Outlier Awareness (`Data Preprocessing.pdf` §3.3):** Linear, Ridge, and Lasso Regression minimize squared error, making them sensitive to extreme outliers. Distinguishing *data-entry errors* (e.g., `km_driven = 1`) from *valid luxury vehicles* (`selling_price > 1.2M INR`) ensures domain-sound cleaning.
- **Scaling & Encoding (`Data Preprocessing.pdf` §4–5):** Regression models require numeric inputs and equal feature scales so Ridge ($\alpha \sum w_j^2$) and Lasso ($\alpha \sum |w_j|$) penalize `km_driven` ($10^5$) and `car_age` ($10^1$) fairly.

---

## 3. Dependencies and Prerequisites

1. Completed `dataset/raw/car.csv` (`4000 × 8`) and insights from Notebook 01.
2. Folder `dataset/processed/` exists.
3. Python libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`.

---

## 4. Step-by-Step Implementation Instructions

### Step 1: Load Raw Data & Check Missing / Placeholder Values
Check standard `NaN`s plus string placeholders (`""`, `"NA"`, `"null"`, `"?"`) and numeric sentinels (`-1`, `-999`) taught in `Data Preprocessing.pdf` §3.2 and `Week4-Data Pre-processing.pdf` §2:

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

df = pd.read_csv("../dataset/raw/car.csv")
print("Initial raw shape:", df.shape)

# Missing & sentinel check
print("Null counts:\n", df.isnull().sum())
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

# Note from Week4-Data Pre-processing.pdf: do NOT blindly call .dropna() without inspecting % missing
assert df.isnull().sum().sum() == 0, "Unexpected missing values found!"
```

---

### Step 2: Remove Exact Duplicates
Per `Week4-Data Pre-processing.pdf` p. 54 & `Machine learning.pdf` §7.4:

```python
n_before = len(df)
n_dups = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {n_dups} exact duplicates ({n_before} -> {len(df)} unique rows).")
```

---

### Step 3: Outlier Analysis (IQR & Z-Score) and Domain Filtering
Apply both mathematical methods from `Data Preprocessing.pdf` §3.3:
- **IQR Rule:** $[Q_1 - 1.5\cdot \text{IQR},\; Q_3 + 1.5\cdot \text{IQR}]$
- **Z-Score Rule (Gaussian reference):** $Z(x) = \frac{x - \mu}{\sigma}$, flagging $|Z| > 3$

**Senior ML Decision:**
- Do **not** blindly delete all high-price luxury cars (`selling_price > Q3 + 1.5*IQR`); those are real market novelties (`Data Preprocessing.pdf` p. 16).
- Inspect and optionally remove clear data-entry anomalies (e.g., `km_driven < 100` for non-test-drive used cars, or `km_driven > 500,000` if verified erroneous). Document exact row counts removed!

```python
def iqr_summary(series, name):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((series < low) | (series > high)).sum()
    print(f"[IQR] {name}: bounds=[{low:,.1f}, {high:,.1f}], outliers={outliers}")

def zscore_summary(series, name):
    z = (series - series.mean()) / series.std()
    outliers = (z.abs() > 3).sum()
    print(f"[Z-score |Z|>3] {name}: outliers={outliers}")

for col in ["selling_price", "km_driven", "year"]:
    iqr_summary(df[col], col)
    zscore_summary(df[col], col)

# Example targeted anomaly guard: remove impossible non-test-drive km_driven < 100
anomaly_mask = (df["km_driven"] < 100) & (df["owner"] != "Test Drive Car")
print("Anomalous near-zero km_driven rows (non-Test Drive):", anomaly_mask.sum())
df = df[~anomaly_mask].reset_index(drop=True)
```

---

### Step 4: Feature Engineering & Irrelevant Column Removal
1. **Extract `brand`** from `name` (first word). Group ultra-rare brands (`count < 15` in cleaned dataset) into `"Other"` to prevent sparse noisy dummy columns.
2. **Compute `car_age`** using dataset reference year `2020`:
   $$\text{car\_age} = 2020 - \text{year}$$
3. **Drop irrelevant/redundant columns** (`name`, `year`) per `Week4-Data Pre-processing.pdf` p. 64 & `Data Preprocessing.pdf` §6.

```python
REFERENCE_YEAR = 2020
df["brand"] = df["name"].str.split().str[0]

# Group rare brands (< 15 listings) into 'Other'
brand_counts = df["brand"].value_counts()
rare_brands = brand_counts[brand_counts < 15].index
df["brand"] = df["brand"].replace(rare_brands, "Other")

df["car_age"] = REFERENCE_YEAR - df["year"]

# Verify car_age vs year perfect inverse correlation before dropping year
print("Pearson r(year, car_age):", df["year"].corr(df["car_age"]))

df_cleaned = df.drop(columns=["name", "year"]).copy()
print("Cleaned schema:", df_cleaned.columns.tolist())
df_cleaned.to_csv("../dataset/processed/car_cleaned.csv", index=False)
```

---

### Step 5: Low-Variance & Feature Correlation Checks
Check for zero/near-zero variance (`Data Preprocessing.pdf` §3.1) and inspect the Pearson correlation matrix on numerical + ordinal-mapped features (`Data Preprocessing.pdf` §6):

```python
owner_order = [
    "Test Drive Car",
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
]
owner_map = {label: idx for idx, label in enumerate(owner_order)}

temp_corr = df_cleaned[["selling_price", "car_age", "km_driven"]].copy()
temp_corr["owner_ord"] = df_cleaned["owner"].map(owner_map)
print("Numerical variance:\n", temp_corr.var())
print("\nPearson correlation matrix:\n", temp_corr.corr().round(3))
```

---

### Step 6: Leakage-Free Train / Validation / Test Split (70% / 15% / 15%)
Split **before** encoding and scaling (see [02_processed_data_splitting_and_schema.md](../../dataset/processed/02_processed_data_splitting_and_schema.md)):

```python
X = df_cleaned.drop(columns=["selling_price"])
y = df_cleaned["selling_price"]

price_bins = pd.qcut(y, q=4, labels=False)

X_dev, X_test, y_dev, y_test, bins_dev, _ = train_test_split(
    X, y, price_bins, test_size=0.15, random_state=42, stratify=price_bins
)
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev, test_size=0.15 / 0.85, random_state=42, stratify=bins_dev
)

print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
```

---

### Step 7: Feature Encoding & Feature Scaling (`ColumnTransformer`)
Apply curriculum-aligned encoding and scaling formulas (`Data Preprocessing.pdf` §4–5):
- **Standardization (Z-score Scaling):** $x_{\text{new}} = \frac{x - \text{mean}(x)}{\text{std}(x)}$ — chosen as primary scaler because `km_driven` has upper-tail outliers that would compress most points near `0` under Min-Max normalization ($x_{\text{new}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$).
- **Ordinal Encoding:** `owner` mapped in true wear/ownership order (`0` to `4`).
- **One-Hot Encoding:** `fuel`, `seller_type`, `transmission`, `brand` (`handle_unknown="ignore"` handles unseen rare categories safely).

```python
from sklearn.pipeline import Pipeline

numeric_features = ["car_age", "km_driven"]
ordinal_features = ["owner"]
nominal_features = ["fuel", "seller_type", "transmission", "brand"]

ordinal_pipeline = Pipeline([
    ("ord", OrdinalEncoder(categories=[owner_order])),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("ord", ordinal_pipeline, ordinal_features),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
    ]
)

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_proc, columns=feature_names)
X_val_df = pd.DataFrame(X_val_proc, columns=feature_names)
X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)
```

*(Optional Teaching Check:)* You may also display a 5-row comparison of `StandardScaler` vs. `MinMaxScaler` on `km_driven` in a markdown/code cell to demonstrate why Standardization handles upper-tail outliers better than Min-Max (`[0, 1]`).

---

### Step 8: Persist Outputs to `../dataset/processed/`

```python
X_train_df.to_csv("../dataset/processed/X_train.csv", index=False)
X_val_df.to_csv("../dataset/processed/X_val.csv", index=False)
X_test_df.to_csv("../dataset/processed/X_test.csv", index=False)

y_train.to_csv("../dataset/processed/y_train.csv", index=False)
y_val.to_csv("../dataset/processed/y_val.csv", index=False)
y_test.to_csv("../dataset/processed/y_test.csv", index=False)

# Save raw combined Train+Val for final best-model retraining in Notebook 04
raw_dev = X_dev.copy()
raw_dev["selling_price"] = y_dev.values
raw_dev.to_csv("../dataset/processed/raw_train_dev.csv", index=False)

joblib.dump(preprocessor, "../dataset/processed/preprocessor.joblib")
print("All processed datasets and preprocessor saved to ../dataset/processed/")
```

---

## 5. Expected Outputs & Validation Criteria

- Exact duplicate count removed is explicitly printed (`0` duplicates remaining in `car_cleaned.csv`).
- `X_train`, `X_val`, and `X_test` have identical column count and zero `NaN` values.
- Training numerical features (`num__car_age`, `num__km_driven`) have mean $\approx 0.0$ and standard deviation $\approx 1.0$.
- `preprocessor.joblib` and all 8 CSVs exist in `dataset/processed/`.
