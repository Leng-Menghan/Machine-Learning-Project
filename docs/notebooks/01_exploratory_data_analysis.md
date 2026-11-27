# Notebook 01 Implementation Guide: Exploratory Data Analysis (EDA)

**Target Notebook:** `notebooks/01_exploratory_data_analysis.ipynb`  
**Input Dataset:** `../dataset/raw/car.csv` (**4,000 rows × 8 columns**)  
**Reference Courses:** `Week-6 Data Visualization.pdf`, `Data Preprocessing.pdf` (§1, §3, §6), `Week4-Data Pre-processing.pdf` (§2), `Machine learning.pdf` (§7.4)

---

## 1. Audit of Current Notebook 01 vs. Required Implementation

Reviewing the existing `01_exploratory_data_analysis.ipynb` (16 code cells, 0 markdown cells) revealed five critical issues that must be fixed:

| Existing Cell(s) | Current Behavior / Defect | Required Fix |
| :--- | :--- | :--- |
| Cell 1–4 | Outputs reflect an `8128 × 10` file (`mileage`, `engine`), not the 4,000 × 8 dataset from `CAR DETAILS FROM CAR DEKHO.csv`. | Re-run against `../dataset/raw/car.csv` (`4000 × 8`) and verify exact columns. |
| Cell 6 | Checks `df.isnull().sum()` only; omits duplicate check. | Add `df.duplicated().sum()`, empty-string checks, and duplicate sample inspection (`Machine learning.pdf` §7.4). |
| Cell 7–8, 13–14 | Plots `df["name"].value_counts()` and `groupby("name")["selling_price"].mean()` directly (1,400+ categories → unreadable x-axis). | Violates `Week-6 Data Visualization.pdf` (*Avoid Chartjunk / Information Overload*). Extract `brand` and plot **Top 10 / Top 15 brands**. |
| Cell 10, 12, 14 | Axis labels say `"Price (USD)"` and `"Average Price (USD)"`. | Change labels to **`Selling Price (INR)`** (`Machine learning.pdf` §2 unit rule). |
| Cell 15 | Runs `sns.countplot(data=df, x="engine")` — **crashes** because `engine` does not exist in the 8-column dataset. | Replace with categorical countplots for `fuel`, `seller_type`, `transmission`, and `owner`. |
| Missing Cells | Missing `km_driven` vs `selling_price` scatter plot, numerical correlation heatmap, and boxplots for outliers. | Add all three per `Machine learning.pdf` §7.4 & `Data Preprocessing.pdf` §3.3. |

---

## 2. Dependencies and Prerequisites

1. `dataset/raw/car.csv` (4,000 × 8) generated per [01_raw_data_selection_and_validation.md](../../dataset/raw/01_raw_data_selection_and_validation.md).
2. Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`.

---

## 3. Step-by-Step Implementation Instructions

### Step 1: Setup & Data Loading (Cells 1–2)
Add a Markdown header explaining the notebook objective, then import libraries and set clean visualization defaults (`Week-6 Data Visualization.pdf` §III.2: clarity over decoration).

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (10, 5)

df = pd.read_csv("../dataset/raw/car.csv")
print("Dataset shape:", df.shape)
df.head()
```

**Expected Output:** Shape `(4000, 8)` with columns `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`.

---

### Step 2: Structural & Data-Type Audit (Cells 3–4)
Map each column to its statistical scale from `Data Preprocessing.pdf` §1:
- **Nominal Categorical:** `name`, `fuel`, `seller_type`, `transmission`
- **Ordinal Categorical:** `owner`
- **Interval / Temporal Numerical:** `year`
- **Ratio Numerical:** `km_driven`, `selling_price` (Target $y$)

```python
df.info()
display(df.describe().T)
display(df.describe(include="object").T)
```

---

### Step 3: Data Quality Audit — Missing Values, Duplicates & Anomalies (Cells 5–6)
Required explicitly by `Machine learning.pdf` §7.4 and `Week4-Data Pre-processing.pdf` §2:

```python
# 1. Missing values & blank strings
print("Null counts:\n", df.isnull().sum())
blank_counts = {
    col: (df[col].astype(str).str.strip() == "").sum()
    for col in df.select_dtypes(include="object").columns
}
print("Blank string counts:", blank_counts)

# 2. Exact duplicates
dup_count = df.duplicated().sum()
print(f"Exact duplicate rows: {dup_count} ({dup_count / len(df) * 100:.2f}%)")
print(f"Unique rows remaining after deduplication: {len(df.drop_duplicates())}")

# 3. Suspicious minimum/extreme checks (e.g., near-zero km_driven)
print("Records with km_driven < 100:", (df["km_driven"] < 100).sum())
```

---

### Step 4: Univariate Analysis — Numerical Distributions & Outlier Boxplots (Cells 7–9)
Per `Week-6 Data Visualization.pdf` §III–IV & `Data Preprocessing.pdf` §3.3:
- **Histograms + KDE** reveal skewness (note: `selling_price` and `km_driven` are strongly right-skewed; `year` is left-skewed).
- **Boxplots + IQR Calculation** detect outliers mathematically using:
  $$\text{IQR} = Q_3 - Q_1, \quad \text{Lower Fence} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Fence} = Q_3 + 1.5 \times \text{IQR}$$

```python
num_cols = ["selling_price", "km_driven", "year"]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for idx, col in enumerate(num_cols):
    sns.histplot(df[col], bins=30, kde=True, ax=axes[0, idx])
    axes[0, idx].set_title(f"Distribution of {col}")

    sns.boxplot(x=df[col], ax=axes[1, idx])
    axes[1, idx].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# Programmatic IQR outlier summary (Week4-Data Pre-processing.pdf p. 52)
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: Q1={q1:,.0f}, Q3={q3:,.0f}, IQR={iqr:,.0f}, Fences=[{lower:,.1f}, {upper:,.1f}], Outliers={n_out}")
```

---

### Step 5: Univariate Analysis — Categorical Features & Engineered Brand (Cells 10–11)
Fix the unreadable `name` bar plot from the starter notebook by extracting `brand` (`df["name"].str.split().str[0]`) and plotting the **Top 12 Brands**, alongside clean countplots for `fuel`, `seller_type`, `transmission`, and `owner` (replacing the broken `engine` cell).

```python
df["brand"] = df["name"].str.strip().str.split().str[0]
print("Unique car names:", df["name"].nunique(), "| Unique brands:", df["brand"].nunique())

# Top 12 brands count & average price (INR)
top_brands = df["brand"].value_counts().head(12)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.barplot(x=top_brands.values, y=top_brands.index, ax=axes[0])
axes[0].set_title("Top 12 Car Brands by Listing Count")
axes[0].set_xlabel("Number of Cars")

brand_avg_price = (
    df[df["brand"].isin(top_brands.index)]
    .groupby("brand")["selling_price"].mean()
    .sort_values(ascending=False)
)
sns.barplot(x=brand_avg_price.values, y=brand_avg_price.index, ax=axes[1])
axes[1].set_title("Average Selling Price (INR) — Top 12 Brands")
axes[1].set_xlabel("Average Selling Price (INR)")
plt.tight_layout()
plt.show()

# Categorical distributions (replacing invalid 'engine' plot)
cat_cols = ["fuel", "seller_type", "transmission", "owner"]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, col in zip(axes.flatten(), cat_cols):
    order = df[col].value_counts().index
    sns.countplot(data=df, y=col, order=order, ax=ax)
    ax.set_title(f"Count of Cars by {col}")
    ax.set_xlabel("Number of Cars")
plt.tight_layout()
plt.show()
```

---

### Step 6: Bivariate Analysis — Key Predictors vs. `selling_price` (Cells 12–13)
Directly satisfies `Machine learning.pdf` §7.4 (*"A scatter plot of mileage versus selling price"* and *"checked using the actual data rather than assuming it is always negative"*):

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1. km_driven vs selling_price (required by Machine learning.pdf §7.4)
sns.scatterplot(data=df, x="km_driven", y="selling_price", alpha=0.4, ax=axes[0])
axes[0].set_title("Distance Driven (km) vs. Selling Price (INR)")
axes[0].set_xlabel("Kilometers Driven (km)")
axes[0].set_ylabel("Selling Price (INR)")

# 2. year vs selling_price
sns.scatterplot(data=df, x="year", y="selling_price", alpha=0.4, ax=axes[1])
axes[1].set_title("Manufacturing Year vs. Selling Price (INR)")
axes[1].set_xlabel("Manufacturing Year")
axes[1].set_ylabel("Selling Price (INR)")

plt.tight_layout()
plt.show()
```

Also inspect categorical relationships vs. `selling_price` using boxplots (`transmission` vs. `selling_price`, `fuel` vs. `selling_price`, `owner` vs. `selling_price`) to confirm whether Automatic, Diesel, and First Owner cars command higher median prices.

---

### Step 7: Multivariate Analysis — Pearson Correlation Heatmap (Cell 14)
Directly satisfies `Machine learning.pdf` §7.4 (*"A correlation heatmap for numerical features"*) and `Data Preprocessing.pdf` §6:
$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

```python
corr_matrix = df[["selling_price", "year", "km_driven"]].corr(method="pearson")

plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("Pearson Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()
```

---

### Step 8: EDA Findings Summary (Markdown Cell 15)
Conclude the notebook with a structured Markdown summary documenting:
1. Exact duplicate count found in the 4,000-record sample (must be removed in Notebook 02).
2. Skewness and outliers in `selling_price` (luxury cars) and `km_driven` (plus `km_driven == 1` anomaly).
3. Observed Pearson correlation signs (`year` positive with `selling_price`, `km_driven` negative with `selling_price`).
4. Rare categories (`Electric` in `fuel`, `Test Drive Car` in `owner`, rare brands in `name`) that Notebook 02 must handle safely.

---

## 4. Expected Outputs & Validation Criteria

- Notebook runs top-to-bottom (`Restart & Run All`) with **zero errors**.
- Zero references to missing `engine` or `mileage` columns.
- All price axes labeled in **INR** (never USD).
- Includes: shape `(4000, 8)`, null count (`0`), exact duplicate count, summary stats, histograms, boxplots + IQR fences, Top-12 brand charts, 4 categorical countplots, `km_driven` vs `selling_price` scatter plot, `year` vs `selling_price` scatter plot, and Pearson correlation heatmap.
