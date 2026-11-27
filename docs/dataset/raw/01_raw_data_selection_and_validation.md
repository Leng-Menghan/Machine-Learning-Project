# Raw Data Selection and Validation Guide (4,000 Records, 8 Columns)

**Target Output Artifact:** `dataset/raw/car.csv`  
**Source File:** `CAR DETAILS FROM CAR DEKHO.csv`  
**Reference Courses:** `Week4-Data Pre-processing.pdf` (Section 1: Data Collection & Governance), `Machine learning.pdf` (Section 2: Dataset Specification)

---

## 1. What Needs to Be Implemented

Extract a reproducible, schema-validated **4,000-record × 8-column** raw working dataset from `CAR DETAILS FROM CAR DEKHO.csv` (which contains **4,340 records × 8 columns**) and persist it to `dataset/raw/car.csv` alongside basic extraction metadata.

### Exact Schema of `CAR DETAILS FROM CAR DEKHO.csv` (8 Columns)

| # | Column Name | Data Type | Role | Observed Values / Range |
| :---: | :--- | :--- | :--- | :--- |
| 1 | `name` | `object` (string) | Input Feature | 1,491 unique car names (e.g., `"Maruti 800 AC"`) |
| 2 | `year` | `int64` | Input Feature | Manufacturing year (`1992` to `2020`) |
| 3 | `selling_price` | `int64` | **Target ($y$)** | Resale price in **INR** (`20,000` to `8,900,000`) |
| 4 | `km_driven` | `int64` | Input Feature | Odometer distance in km (`1` to `806,599`) |
| 5 | `fuel` | `object` (string) | Input Feature | `Diesel`, `Petrol`, `CNG`, `LPG`, `Electric` |
| 6 | `seller_type` | `object` (string) | Input Feature | `Individual`, `Dealer`, `Trustmark Dealer` |
| 7 | `transmission` | `object` (string) | Input Feature | `Manual`, `Automatic` |
| 8 | `owner` | `object` (string) | Input Feature | `First Owner`, `Second Owner`, `Third Owner`, `Fourth & Above Owner`, `Test Drive Car` |

> **Unit Clarification (`Machine learning.pdf` §2):**  
> `selling_price` in `CAR DETAILS FROM CAR DEKHO.csv` is recorded in **Indian Rupees (INR)** (e.g., `60000`, `450000`, `1650000`), not USD and not lakh INR. All downstream notebooks must label axes and predictions in **INR** unless an explicit exchange-rate conversion is applied.

---

## 2. Why This Step Is Necessary

1. **Matches Reference Directory Contract:** Notebook 01 expects to read from `../dataset/raw/car.csv`. Currently, neither `dataset/raw/` nor `car.csv` exists in the workspace.
2. **Fixes Schema Mismatch in Notebook 01:** The starter `01_exploratory_data_analysis.ipynb` previously referenced a different `8,128 × 10` CarDekho file containing `mileage` and `engine`. Standardizing `dataset/raw/car.csv` on the **4,000 × 8** dataset derived from `CAR DETAILS FROM CAR DEKHO.csv` aligns the project with the actual available data.
3. **Prevents Sampling Bias (`Week4-Data Pre-processing.pdf` §1):** Taking `df.head(4000)` can introduce ordering bias if listings are sorted by date, seller, or model. Reproducible random or stratified sampling (`random_state=42`) ensures the 4,000-record subset represents the full 4,340-record distribution.
4. **Data Governance & Traceability (`Week4-Data Pre-processing.pdf` §1):** Documenting source, record count, seed, and schema preserves reproducibility across team members.

### Important Note on Duplicates vs. 4,000 Records
Inspection of `CAR DETAILS FROM CAR DEKHO.csv` shows **4,340 total rows**, of which **763 are exact duplicate rows**, leaving **3,577 unique rows**.  
- Therefore, `dataset/raw/car.csv` satisfies the **4,000-record raw dataset size requirement**.
- Exact duplicates inside those 4,000 rows **must still be audited in Notebook 01 and removed in Notebook 02** before train/test splitting to prevent data leakage (identical cars appearing in both training and test sets).

---

## 3. Prerequisites and Dependencies

- **Workspace Folder:** Create `dataset/raw/` under the project root.
- **Python Libraries:** `pandas`, `os`, `json` (optional for metadata).
- **Source File:** `CAR DETAILS FROM CAR DEKHO.csv` present in project root.

---

## 4. Step-by-Step Implementation Instructions

### Step 1: Create the Target Directory Structure
Create the folders matching the reference image (`dataset/raw/`, `dataset/processed/`, `models/`):

```python
import os

os.makedirs("dataset/raw", exist_ok=True)
os.makedirs("dataset/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
```

### Step 2: Load and Validate the Full Source CSV
Verify shape (`4340, 8`), exact column names, and absence of missing/corrupt rows before sampling.

```python
import pandas as pd

SOURCE_PATH = "CAR DETAILS FROM CAR DEKHO.csv"
EXPECTED_COLUMNS = [
    "name", "year", "selling_price", "km_driven",
    "fuel", "seller_type", "transmission", "owner"
]

df_source = pd.read_csv(SOURCE_PATH)

assert df_source.shape == (4340, 8), f"Unexpected source shape: {df_source.shape}"
assert list(df_source.columns) == EXPECTED_COLUMNS, "Column mismatch!"
assert df_source.isnull().sum().sum() == 0, "Unexpected null values in source!"

print("Source validated:", df_source.shape)
```

### Step 3: Reproducibly Sample 4,000 Records
Because `fuel == "Electric"` contains only **1** row in `CAR DETAILS FROM CAR DEKHO.csv`, stratifying directly on `fuel` raises a `ValueError` (`train_test_split` requires at least 2 samples per stratum). Instead, stratify by **`transmission` × `selling_price` quartile** so every stratum is well populated (min count ≥ 14) while preserving both transmission balance and the full price range:

```python
from sklearn.model_selection import train_test_split

# Stratify by transmission + selling_price quartile (every stratum well populated)
price_q = pd.qcut(df_source["selling_price"], q=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
strat_key = df_source["transmission"] + "_" + price_q

df_raw, _ = train_test_split(
    df_source,
    train_size=4000,
    random_state=42,
    stratify=strat_key
)
df_raw = df_raw.reset_index(drop=True)
```

### Step 4: Save `dataset/raw/car.csv` and Write Governance Metadata
Save the 4,000-record file without the pandas index column:

```python
RAW_OUTPUT_PATH = "dataset/raw/car.csv"
df_raw.to_csv(RAW_OUTPUT_PATH, index=False)

print(f"Saved {df_raw.shape[0]} rows and {df_raw.shape[1]} columns to {RAW_OUTPUT_PATH}")
```

---

## 5. Expected Outputs & Validation Criteria

After running the extraction script or cell, verify all assertions pass:

```python
df_check = pd.read_csv("dataset/raw/car.csv")

assert df_check.shape == (4000, 8), f"Expected (4000, 8), got {df_check.shape}"
assert list(df_check.columns) == EXPECTED_COLUMNS
assert df_check.isnull().sum().sum() == 0
assert df_check["selling_price"].min() > 0
assert df_check["year"].between(1990, 2020).all()

print("Validation PASSED:")
print("Shape:", df_check.shape)
print("Fuel distribution (%):\n", (df_check["fuel"].value_counts(normalize=True) * 100).round(2))
print("Exact duplicates inside 4,000-row sample:", df_check.duplicated().sum())
```

### Checklist
- [ ] Directory `dataset/raw/` exists.
- [ ] `dataset/raw/car.csv` contains **4,000 rows** and **8 columns**.
- [ ] Column names match `CAR DETAILS FROM CAR DEKHO.csv` exactly.
- [ ] Sampling uses fixed `random_state=42` for reproducibility.
- [ ] Duplicate count inside the 4,000-row raw file is logged (for removal in Notebook 02).
