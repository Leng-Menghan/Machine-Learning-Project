# Project Implementation Plan: Used Car Price Prediction

**Document Role:** Master Implementation Roadmap (Stage 1 — Plan Only)  
**Course:** Machine Learning (Year 4) — Classical Machine Learning: Supervised Learning (Regression)  
**Institution:** Cambodia Academy of Digital Technology (CADT)  
**Project Workspace:** `Machine-Learning-Project/`  
**Status:** Draft for Review & Approval (No notebooks modified or child documentation files created yet)

---

## 1. Project Overview and Objectives

### 1.1 Business & Engineering Problem
Pricing a used car accurately is challenging because resale value depends on interacting factors such as vehicle brand/model, manufacturing year (age), distance driven (`km_driven`), fuel type, seller type, transmission, and ownership history. Setting a price too high delays or prevents a sale, while setting it too low causes financial loss to the seller.

This project builds, compares, and evaluates classical **Supervised Machine Learning (Regression)** models to predict the continuous `selling_price` of a used car from historical CarDekho listing records.

### 1.2 Formal Problem Definition (Tom Mitchell's $T, P, E$ Framework)
Aligned with `Machine learning.pdf` (Section 1.2):
- **Task ($T$):** Predict the continuous resale `selling_price` of a used car given its attributes ($X$).
- **Performance Measure ($P$):** Evaluate predictive accuracy on unseen test data using **MAE** (Mean Absolute Error), **RMSE** (Root Mean Squared Error), and **$R^2$** (Coefficient of Determination).
- **Experience ($E$):** Learn statistical relationships from historical used-car records derived from `CAR DETAILS FROM CAR DEKHO.csv`.

### 1.3 Core Objectives
1. **Reproducible Data Sourcing & Governance:** Construct a validated **4,000-record, 8-column** working raw dataset (`dataset/raw/car.csv`) from `CAR DETAILS FROM CAR DEKHO.csv` with documented sampling rules and data-quality checks.
2. **Curriculum-Aligned Exploratory Data Analysis (EDA):** Diagnose distributions, skewness, duplicates, outliers, and feature relationships in `01_exploratory_data_analysis.ipynb` using principles from `Week-6 Data Visualization.pdf` and `Data Preprocessing.pdf`.
3. **Leakage-Free Preprocessing & Feature Engineering:** Clean data, handle duplicates and outliers, engineer domain features (`brand`, `car_age`), encode categoricals (Ordinal + One-Hot), scale numericals, and split data properly in `02_data_preprocessing.ipynb` using methods from `Data Preprocessing.pdf` and `Week4-Data Pre-processing.pdf`.
4. **Classical Regression Modeling:** Train, tune (via validation / cross-validation where regularization $\alpha$ or tree depth applies), and persist **Linear Regression**, **Ridge Regression**, **Lasso Regression**, and **Decision Tree Regressor** in `03_model_training.ipynb`.
5. **Rigorous Model Evaluation & Prediction Demo:** Compare all 4 models on MAE, RMSE, and $R^2$, perform residual and feature-importance error analysis, answer the 5 course discussion questions, retrain the best model on full training data, and deliver a new-car price prediction function in `04_model_evaluation.ipynb`.
6. **Modular Step-by-Step Documentation:** Produce clean, separate `.md` implementation guides inside a `docs/` directory structured directly after the attached reference image.

---

## 2. Analysis of Provided Resources

All four categories of provided resources were inspected directly without assumptions.

### 2.1 PDF Course & Project Documents

| PDF File | Course / Context | Key Concepts & Methodologies Covered | Project Relevance |
| :--- | :--- | :--- | :--- |
| **`Machine learning.pdf`** | Project Proposal & Implementation Guide (CADT Year 4, Classical ML — Regression) | Tom Mitchell $(T,P,E)$; Regression framing; CarDekho dataset spec & INR/USD unit warning; 8-step ML lifecycle; Linear, Ridge ($L_2$), Lasso ($L_1$), and Decision Tree Regressor formulas; MAE, RMSE, $R^2$ formulas; 8-step Python workflow; retraining best model on full training data; prediction demo function; Lab Report tasks & 5 discussion questions. | **Primary specification** for modeling, evaluation metrics, EDA requirements (scatter plot of `km_driven` vs price, correlation heatmap, duplicate/missing checks), and final deliverables. |
| **`Data Preprocessing.pdf`** | CADT Data Preprocessing (Lecturers: Ms. Sokchovy Monirath & Mr. Soklay Heng) | Data taxonomy (Nominal, Ordinal, Interval, Ratio) & detail properties; Independent ($x$) vs Dependent ($y$) variables; Low variation / zero-variance removal; Missing values & Vishal Patel thresholds (`<50%` impute, `>95%` discard); Outlier detection (Gaussian Z-score $\pm 3\sigma$, IQR fences $[Q_1 - 1.5\cdot IQR, Q_3 + 1.5\cdot IQR]$); Imbalanced data/SMOTE (classification only); Scaling (Standardization vs Min-Max); Encoding (Label/Ordinal vs One-Hot); Feature Correlation (Pearson $r_{xy}$). | **Primary theoretical reference** for variable typing, outlier math, scaling formulas, encoding selection, and Pearson correlation feature analysis. |
| **`Week4-Data Pre-processing.pdf`** | CADT IDT — Fundamentals of Data Science (Lecture 4) | Data collection (Primary vs Secondary, Structured CSV, Batch processing, Governance & Metadata); Missing data mechanisms (MCAR/MAR/MNAR) & Rules 1–5 (avoiding blind `.dropna()`); Step-by-step IQR calculation with Pandas `.quantile()`; Duplicate detection (`duplicated()`, `drop_duplicates()`); One-Hot vs Label vs Ordinal encoding; Data type conversion; Text/String cleaning; Removing irrelevant data. | **Primary practical guide** for data governance, duplicate handling, string cleaning (`name`), ordinal mapping (`owner`), and avoiding naive `.dropna()`. |
| **`Week-6 Data Visualization.pdf`** | CADT IDT — Fundamentals of Data Science (Lecture 6) | Visualization across workflow (*before, during, after* modeling); Univariate (Histogram, Boxplot, Bar, Pie), Bivariate (Scatter, Line), Multivariate (Heatmap, Pairplot); Good design principles (clarity over decoration, avoid chartjunk, proper labels/units, consistent scales); Pitfalls (information overload, deceptive scales); Library roles (Matplotlib + Seaborn for EDA). | **Primary standard** for fixing Notebook 01's unreadable plots and designing clear EDA and evaluation visuals. |

### 2.2 CSV Dataset (`CAR DETAILS FROM CAR DEKHO.csv`)

Direct inspection of `CAR DETAILS FROM CAR DEKHO.csv` revealed the following exact facts:

- **Dimensions:** **4,340 rows × 8 columns**
- **Missing / Empty Values:** **0** nulls and **0** blank/whitespace strings across all 8 columns.
- **Exact Duplicate Rows:** **763 exact duplicate rows** (only **3,577 unique rows**).
- **Currency Unit:** `selling_price` is stored in **Indian Rupees (INR)** (min ₹20,000, max ₹8,900,000), **not** USD and **not** lakh INR.

#### Exact Column Profile (8 Columns)

| # | Column | Raw Dtype | Data Scale (`Data Preprocessing.pdf`) | Role | Observed Statistics / Unique Categories |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | `name` | `object` (string) | Categorical — Nominal / Text | Feature (needs `brand` extraction) | **1,491** unique car names; **29** unique leading brand tokens (Top: `Maruti` 1,280; `Hyundai` 821; `Mahindra` 365; `Tata` 361; `Honda` 252; `Ford` 238; `Toyota` 206). |
| 2 | `year` | `int64` | Numerical — Interval (convertible to Ratio `car_age`) | Feature | Range **1992–2020**; mean **2013.09**; Q1=2011, median=2014, Q3=2016; IQR=5; 105 lower-tail outliers (`year < 2004`). |
| 3 | `selling_price` | `int64` | Numerical — Ratio (INR) | **Target ($y$)** | Min **20,000**, Q1=**204,999**, median=**350,000**, Q3=**600,000**, max=**8,900,000**, mean=**504,127.31**; strongly right-skewed; 269 upper-tail IQR outliers (`> 1,192,501.5`). |
| 4 | `km_driven` | `int64` | Numerical — Ratio (km) | Feature | Min **1**, Q1=**35,000**, median=**60,000**, Q3=**90,000**, max=**806,599**, mean=**66,215.78**; right-skewed; 110 upper-tail IQR outliers (`> 172,500`); minimum `1` km warrants anomaly inspection. |
| 5 | `fuel` | `object` | Categorical — Nominal | Feature | 5 categories: `Diesel` (2,153), `Petrol` (2,123), `CNG` (40), `LPG` (23), `Electric` (1 — extreme rare class). |
| 6 | `seller_type` | `object` | Categorical — Nominal | Feature | 3 categories: `Individual` (3,244), `Dealer` (994), `Trustmark Dealer` (102). |
| 7 | `transmission` | `object` | Categorical — Nominal (Binary) | Feature | 2 categories: `Manual` (3,892), `Automatic` (448). |
| 8 | `owner` | `object` | Categorical — Ordinal | Feature | 5 categories: `First Owner` (2,832), `Second Owner` (1,106), `Third Owner` (304), `Fourth & Above Owner` (81), `Test Drive Car` (17). |

> **Schema Comparison Note (`Machine learning.pdf` vs. Actual CSV):**  
> `Machine learning.pdf` (Section 2) shows an illustrative CarDekho table containing `Present_Price` (showroom price in lakh INR) and integer `Owner` (`0, 1, 2`). Those represent the smaller 301-row CarDekho variant, whereas the workspace's `CAR DETAILS FROM CAR DEKHO.csv` has **8 columns** without `Present_Price` and with text `owner` and `seller_type` (`Trustmark Dealer`) plus `name`. All implementation docs and notebooks must reflect the **actual 8 columns** present in `CAR DETAILS FROM CAR DEKHO.csv`.

### 2.3 Jupyter Notebooks (`notebooks/01`–`04`)

- `notebooks/01_exploratory_data_analysis.ipynb`: **Partially written (16 code cells, 0 markdown cells)** — currently targets a **different dataset** (`../dataset/raw/car.csv` with shape `(8128, 10)` including `mileage` and `engine`).
- `notebooks/02_data_preprocessing.ipynb`: **Empty (0 bytes)**.
- `notebooks/03_model_training.ipynb`: **Empty (0 bytes)**.
- `notebooks/04_model_evaluation.ipynb`: **Empty (0 bytes)**.

### 2.4 Reference Image

The attached reference image shows the target project directory layout:
- `dataset/`
  - `processed/`
  - `raw/`
- `models/`
- `notebooks/`
  - `01_exploratory_data_analysis.ipynb`
  - `02_data_preprocessing.ipynb`
  - `03_model_training.ipynb`
  - `04_model_evaluation.ipynb`

Currently, `dataset/` (and its `raw/`, `processed/` subdirectories) and `models/` **do not exist** in the workspace and must be created.

---

## 3. Overview of the 4,000-Record Dataset Requirement

### 3.1 How to Obtain / Select the 4,000 Records (8 Features)
The raw source file `CAR DETAILS FROM CAR DEKHO.csv` contains **4,340 records** and **8 columns** (`name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner` — 7 input features + 1 target `selling_price`, forming an 8-column schema).

To obtain a reproducible **4,000-record** working dataset saved at `dataset/raw/car.csv`:
- **Recommended Method:** Perform reproducible **stratified sampling** (by `fuel` or `transmission` / price quantile bins, or `df.sample(n=4000, random_state=42)` after handling the singleton `Electric` row carefully) from `CAR DETAILS FROM CAR DEKHO.csv` (4,340 rows) to preserve the statistical distribution of the full dataset rather than taking a biased `head(4000)` slice.
- **Metadata & Governance (`Week4-Data Pre-processing.pdf`):** Log the source filename, sampling seed (`random_state=42`), extraction timestamp, row count (`4,000`), and column schema (`8` columns) in `dataset/raw/`.

### 3.2 Critical Engineering Finding: 4,000 Records vs. Duplicate Removal
A Senior ML Engineer must flag a concrete mathematical constraint observed in `CAR DETAILS FROM CAR DEKHO.csv`:
- Out of **4,340** raw rows, **763** are exact duplicates, leaving only **3,577 unique rows**.
- Therefore, **no subset of 4,000 rows from this single CSV can remain at 4,000 rows after exact deduplication.**
- **Leakage Risk:** If exact duplicates remain in the dataset across `train_test_split`, identical cars appear in both training and test sets, causing **data leakage** and artificially inflating test $R^2$ while underestimating MAE/RMSE.
- **Planned Resolution Strategy:**
  1. **Raw Stage (`dataset/raw/car.csv`):** Extract **exactly 4,000 records × 8 columns** from `CAR DETAILS FROM CAR DEKHO.csv` to satisfy the project's raw dataset size requirement (`4,000` records).
  2. **EDA Stage (Notebook 01):** Explicitly audit and report the exact number of duplicates and near-duplicates inside the 4,000-record raw dataset.
  3. **Preprocessing Stage (Notebook 02):** Apply `drop_duplicates()` (per `Week4-Data Pre-processing.pdf` & `Machine learning.pdf` Section 7.4), documenting the exact pre-cleaning (`4,000`) and post-deduplication row counts (approx. 3,300–3,577 unique rows) saved to `dataset/processed/`. *(Never duplicate synthetic rows or keep duplicates solely to force an artificial post-cleaning count of 4,000.)*

### 3.3 Data Quality, Validation, Cleaning & Preparation Considerations
- **Missing Values:** Verified `0` nulls in source CSV, yet Notebook 01/02 must still include programmatic checks (`isnull().sum()`, empty-string/sentinel checks like `-1`, `-999`, `"NA"`) as required by `Data Preprocessing.pdf` and `Week4-Data Pre-processing.pdf`.
- **String Cleaning (`name`):** Trim extra whitespace (`str.strip()`), normalize spaces, and extract `brand` (first token, 29 brands). Group ultra-rare brands/categories or configure `OneHotEncoder(handle_unknown='ignore')`.
- **Feature Engineering (`car_age`):** Derive `car_age = reference_year - year` (using `reference_year = 2020`, the max year in the dataset, or current reference year documented explicitly; using `2020` avoids negative/inflated ages relative to 2020 listing prices) and drop raw `year` and raw high-cardinality `name` before linear modeling to avoid multicollinearity and overfitting on 1,491 model names.
- **Outliers (`selling_price`, `km_driven`, `year`):** Apply IQR fence analysis and Z-score inspection. Because luxury vehicles (₹1.2M–₹8.9M) are *natural novelties* (`Data Preprocessing.pdf` p. 16) rather than sensor errors, evaluate whether extreme data-entry errors (e.g., `km_driven == 1`) should be filtered/capped and compare Linear/Ridge/Lasso sensitivity to upper-tail prices (including optional `log1p(selling_price)` diagnostic experiment vs. raw INR baseline).
- **Splitting Protocol (`Train / Validation / Test`):** Split **before** fitting scalers, encoders, or imputers to guarantee zero data leakage:
  - **80% Development / 20% Held-out Test** (or **70% Train / 15% Validation / 15% Test** — or **80% Train (with 5-fold CV for $\alpha$/`max_depth` tuning) / 20% Test**, aligning directly with `Machine learning.pdf` Section 5's instruction to hold out a test set for evaluation and retrain the selected best model on the full training set).

---

## 4. Analysis of Jupyter Notebooks (01–04)

### 4.1 Current Status & Gap Assessment

| Notebook | Current State | Specific Issues / Gaps Found | Required Implementation |
| :--- | :--- | :--- | :--- |
| **`01_exploratory_data_analysis.ipynb`** | 16 code cells, 0 markdown cells. Built for a different `8128 × 10` CarDekho file. | 1. Reads missing path `../dataset/raw/car.csv` (`(8128, 10)` with `mileage`, `engine`).<br>2. Cell 15 (`sns.countplot(data=df, x="engine")`) fails on the 8-column CSV.<br>3. Axis labels say `"Price (USD)"` instead of **INR**.<br>4. Plots 2,058/1,491 raw `name` values on a bar chart (unreadable chartjunk).<br>5. Missing `km_driven` vs `selling_price` scatter plot (`Machine learning.pdf` §7.4).<br>6. Missing numerical correlation heatmap (`Machine learning.pdf` §7.4).<br>7. Missing duplicate check (`duplicated().sum()`).<br>8. Missing boxplots for IQR outlier inspection.<br>9. Missing categorical distribution plots (`fuel`, `seller_type`, `transmission`, `owner`, `brand`). | Refactor to load the 4,000-row × 8-column `../dataset/raw/car.csv`; add structured Markdown sections; fix currency labels to INR; extract `brand` for Top-N charts; add missing/duplicate audits, summary stats, histograms, boxplots, `km_driven` vs `selling_price` & `year` vs `selling_price` scatter plots, categorical countplots, and correlation heatmap. |
| **`02_data_preprocessing.ipynb`** | **Empty (0 bytes)** | No code or documentation exists. | Implement full cleaning & preparation pipeline: schema validation, text cleaning, duplicate removal, outlier analysis (IQR & Z-score), feature engineering (`brand`, `car_age`), correlation/multicollinearity check, data splitting (Train/Val/Test), Ordinal Encoding (`owner`), One-Hot Encoding (`fuel`, `seller_type`, `transmission`, `brand`), numerical scaling (`StandardScaler` / `MinMaxScaler` comparison), and saving processed datasets & fitted preprocessor to `../dataset/processed/`. |
| **`03_model_training.ipynb`** | **Empty (0 bytes)** | No code or documentation exists. | Load processed Train/Val splits from `../dataset/processed/`; train baseline **Linear Regression**, regularized **Ridge Regression** (tune $\alpha$), **Lasso Regression** (tune $\alpha$, inspect zeroed weights), and **Decision Tree Regressor** (tune `max_depth` / `min_samples_leaf` to prevent overfitting); log validation metrics; save trained models to `../models/`. |
| **`04_model_evaluation.ipynb`** | **Empty (0 bytes)** | No code or documentation exists. | Load saved models from `../models/` and unseen test set from `../dataset/processed/`; compute **MAE, RMSE, $R^2$** on test data; generate comparison table & bar charts; plot Actual vs. Predicted and Residual diagnostics; inspect feature coefficients / tree feature importances; answer the 5 PDF discussion questions; retrain best model on full training set; build & test `predict_car_price(...)` demo function. |

---

## 5. Mapping Between PDF Course Lessons and Notebook Tasks

### 5.1 Applicable Curriculum Mapping

| Course Topic (PDF Source) | Mathematical / Conceptual Basis | Target Notebook(s) | Concrete Project Implementation |
| :--- | :--- | :--- | :--- |
| **Data Types & Variable Roles** (`Data Preprocessing.pdf` §1–2) | Nominal, Ordinal, Interval, Ratio; Independent $X$ vs Dependent $y$ | NB 01, NB 02 | Classify all 8 columns + engineered `car_age` and `brand`; set $y =$ `selling_price`. |
| **Data Collection & Governance** (`Week4` §1; `ML.pdf` §2–3) | Batch secondary CSV ingestion, metadata logging, sampling bias control | Raw Extraction + NB 01 | Reproducible 4,000-record sampling (`random_state=42`) from `CAR DETAILS FROM CAR DEKHO.csv` with schema checks and INR unit documentation. |
| **Data Visualization Principles & Chart Selection** (`Week-6` §III–V; `ML.pdf` §7.4) | Univariate, Bivariate, Multivariate plots; clarity over decoration | NB 01, NB 04 | Histograms + KDE, boxplots, Top-N brand bar charts, categorical countplots, `km_driven` vs `selling_price` scatter plot, Pearson correlation heatmap, residual plots. |
| **Missing Value & Duplicate Handling** (`Data Prep` §3.2; `Week4` §2; `ML.pdf` §7.4) | MCAR/MAR/MNAR, `<5%` rule, Vishal Patel rule, `duplicated()` / `drop_duplicates()` | NB 01, NB 02 | Audit nulls, empty strings, sentinel values; remove exact duplicates before splitting to prevent train/test leakage. |
| **Text/String Cleaning & Irrelevant Data Removal** (`Week4` §2) | Whitespace stripping, case normalization, dropping non-predictive high-cardinality IDs | NB 02 | Clean `name`, extract `brand`, drop raw `name` and redundant raw `year` (replaced by `car_age`). |
| **Outlier Detection (IQR & Z-Score)** (`Data Prep` §3.3; `Week4` §2) | $Q_1 - 1.5\cdot IQR$ to $Q_3 + 1.5\cdot IQR$; $Z = \frac{x - \mu}{\sigma}$ ($\pm 3$) | NB 01, NB 02 | Detect outliers on `selling_price`, `km_driven`, `year`; distinguish errors (`km_driven=1`) from genuine luxury cars; handle cautiously for linear sensitivity. |
| **Feature Encoding** (`Data Prep` §5; `Week4` §2; `ML.pdf` §5, §7.4) | Ordinal Encoding (ordered categories) & One-Hot Encoding (nominal categories) | NB 02 | Ordinal-encode `owner` (`Test Drive Car` / `First` → `Fourth & Above`); One-Hot encode `fuel`, `seller_type`, `transmission`, `brand` (`handle_unknown='ignore'`). |
| **Feature Scaling** (`Data Prep` §4; `Week4` §2; `ML.pdf` §5) | Standardization $x_{new} = \frac{x - \mu}{\sigma}$ & Min-Max $x_{new} = \frac{x - x_{min}}{x_{max} - x_{min}}$ | NB 02 | Fit `StandardScaler` on training numericals (`car_age`, `km_driven`, ordinal `owner`) for Ridge/Lasso/Linear comparability; note Decision Tree scale invariance. |
| **Feature Correlation & Selection** (`Data Prep` §6; `ML.pdf` §7.4) | Pearson $r_{xy} \in [-1, +1]$ & multicollinearity inspection | NB 01, NB 02 | Correlation matrix heatmap on numerical/encoded features; verify `year` vs `car_age` perfect inverse correlation ($r = -1.0$) and drop redundant `year`. |
| **Classical Regression Models** (`ML.pdf` §4–5) | Linear Regression, Ridge ($L_2$), Lasso ($L_1$), Decision Tree Regressor | NB 03 | Train all 4 models on identical train/val splits; tune $\alpha$ for Ridge/Lasso and `max_depth`/`min_samples_leaf` for Decision Tree. |
| **Evaluation Metrics & Error Analysis** (`ML.pdf` §6–7) | $\text{MAE}, \text{RMSE}, R^2$, test comparison table, retraining & inference function | NB 04 | Evaluate on held-out test set; build metric table; analyze Lasso zero weights & tree non-linearity; retrain best model on full train set; implement `predict_car_price()`. |

### 5.2 Course Techniques Excluded as Non-Applicable (Scope Discipline)
Following the requirement to avoid irrelevant techniques:
- **Undersampling, Oversampling, and SMOTE (`Data Preprocessing.pdf` §3.4):** Explicitly designed for **imbalanced classification** tasks; *not applicable* because `selling_price` is a continuous regression target.
- **Confusion Matrix & ROC Curves (`Week-6` p. 14):** Classification metrics; replaced by **Actual vs. Predicted scatter plots**, **Residual distribution plots**, and **MAE / RMSE / $R^2$** comparison charts.
- **K-Means / PCA / Classification Models:** Out of scope for Supervised Regression (`Machine learning.pdf` §1.3 & §4).

---

## 6. Proposed Implementation Workflow

```text
[CAR DETAILS FROM CAR DEKHO.csv (4,340 x 8)]
                     │
   Step 0: Reproducible Stratified/Random Sampling (n=4,000, seed=42) & Schema Validation
                     ▼
          [dataset/raw/car.csv (4,000 x 8)]
                     │
   Step 1 (Notebook 01): EDA — Shape, Info, Null/Duplicate Audit, IQR Outliers,
          Univariate/Bivariate/Multivariate Visuals (INR units, Top Brands, km vs Price, Heatmap)
                     ▼
   Step 2 (Notebook 02): Data Cleaning (Duplicates, Anomalies), Feature Engineering (brand, car_age),
          Train/Val/Test Split (BEFORE fitting transformers!), Ordinal + One-Hot Encoding, Scaling
                     ▼
     [dataset/processed/ (cleaned.csv, train/val/test splits, preprocessor.joblib)]
                     │
   Step 3 (Notebook 03): Train 4 Classical Regression Models (Linear, Ridge, Lasso, Decision Tree),
          Tune Hyperparameters on Validation / CV, Save Trained Artifacts
                     ▼
     [models/ (linear_regression.joblib, ridge.joblib, lasso.joblib, decision_tree.joblib)]
                     │
   Step 4 (Notebook 04): Test-Set Evaluation (MAE, RMSE, R^2), Residual & Error Analysis,
          5 Discussion Answers, Retrain Best Model on Full Train Set, Price Prediction Demo
```

### Data Splitting Strategy (Leakage-Free)
1. Clean global row-level errors and remove exact duplicates **first** (so identical rows cannot span train and test).
2. Split into **Training (70%)**, **Validation (15%)**, and **Test (15%)** — or **80% Train (with 5-fold Cross-Validation) / 20% Test** — using `random_state=42`. (Using `70/15/15` or `80/20 + CV` cleanly supports hyperparameter tuning for Ridge $\alpha$, Lasso $\alpha$, and Decision Tree `max_depth` without touching the held-out test set.)
3. Fit `OrdinalEncoder`, `OneHotEncoder`, and `StandardScaler` (via scikit-learn `ColumnTransformer`) **strictly on the training split**, then `.transform()` validation and test splits.
4. After selecting the best model on the test evaluation in Notebook 04, retrain the winning pipeline on the combined `Train + Validation` set (keeping Test untouched for final reporting) before exporting the final demo estimator.

---

## 7. Proposed `docs/` Documentation Folder Structure (Reference Image Comparison)

### 7.1 Comparison with Attached Reference Image

| Reference Image Hierarchy | Current Workspace Status | Planned Project Workspace | Planned `docs/` Docs Mirror |
| :--- | :--- | :--- | :--- |
| `Project/` | `Machine-Learning-Project/` (root) | Root holds `CAR DETAILS FROM CAR DEKHO.csv`, PDFs, `docs/` | `docs/` holds `PROJECT_IMPLEMENTATION_PLAN.md` and mirrors `Project/` |
| `├── dataset/` | Created (`dataset/`) | `dataset/` | `docs/dataset/` |
| `│   ├── processed/` | Created (`dataset/processed/`) | `dataset/processed/` | `docs/dataset/processed/` |
| `│   └── raw/` | Created (`dataset/raw/`) | `dataset/raw/` | `docs/dataset/raw/` |
| `├── models/` | Created (`models/`) | `models/` | `docs/models/` |
| `└── notebooks/` | Exists (`01`–`04 .ipynb`) | `notebooks/01`–`04 .ipynb` | `docs/notebooks/` |

### 7.2 Exact Target Structures

**Project Workspace Structure (matching reference image 1:1):**
```text
Machine-Learning-Project/
├── CAR DETAILS FROM CAR DEKHO.csv
├── dataset/
│   ├── raw/
│   │   └── car.csv
│   └── processed/
│       ├── car_cleaned.csv
│       ├── train.csv / val.csv / test.csv (and/or X_train, X_val, X_test, y_train, y_val, y_test)
│       └── preprocessor.joblib
├── models/
│   ├── linear_regression.joblib
│   ├── ridge_regression.joblib
│   ├── lasso_regression.joblib
│   ├── decision_tree_regressor.joblib
│   └── best_car_price_model.joblib
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
└── docs/   <-- Separate step-by-step Markdown documentation folder
```

**`docs/` Documentation Folder Structure (strictly mirroring the reference image):**
```text
docs/
├── PROJECT_IMPLEMENTATION_PLAN.md
├── dataset/
│   ├── raw/
│   │   └── 01_raw_data_selection_and_validation.md
│   └── processed/
│       └── 02_processed_data_splitting_and_schema.md
├── notebooks/
│   ├── 01_exploratory_data_analysis.md
│   ├── 02_data_preprocessing.md
│   ├── 03_model_training.md
│   └── 04_model_evaluation.md
└── models/
    └── 05_model_artifacts_and_prediction_demo.md
```

This structure follows the reference image's exact top-level directories (`dataset/raw`, `dataset/processed`, `models`, `notebooks`) without inventing unrelated directories, while keeping each implementation step modular and free of duplication.

---

## 8. List of Separate Markdown Documentation Files

The following **7 separate implementation `.md` files** live alongside `PROJECT_IMPLEMENTATION_PLAN.md` inside `docs/`:

| Order | File Path | Focus & Scope | Aligned PDF Lessons |
| :---: | :--- | :--- | :--- |
| 1 | `docs/dataset/raw/01_raw_data_selection_and_validation.md` | Obtaining & sampling **4,000 records (8 features)** from `CAR DETAILS FROM CAR DEKHO.csv` into `dataset/raw/car.csv`; data governance, metadata, schema & INR currency verification. | `Week4-Data Pre-processing.pdf` (§1), `Machine learning.pdf` (§2) |
| 2 | `docs/notebooks/01_exploratory_data_analysis.md` | Step-by-step guide to refactor and complete `01_exploratory_data_analysis.ipynb`: summary stats, missing & duplicate audits, IQR outlier inspection, Univariate/Bivariate/Multivariate plots (`km_driven` vs price, correlation heatmap, Top brands), fixing USD/`engine` bugs. | `Week-6 Data Visualization.pdf`, `Data Preprocessing.pdf` (§1, §3, §6), `Machine learning.pdf` (§7.4) |
| 3 | `docs/notebooks/02_data_preprocessing.md` | Step-by-step guide for `02_data_preprocessing.ipynb`: duplicate removal, string cleaning (`name`), outlier analysis (IQR & Z-score), feature engineering (`brand`, `car_age`), multicollinearity check, Ordinal Encoding (`owner`), One-Hot Encoding, and Feature Scaling (`StandardScaler` / `MinMaxScaler`). | `Data Preprocessing.pdf` (§3–6), `Week4-Data Pre-processing.pdf` (§2) |
| 4 | `docs/dataset/processed/02_processed_data_splitting_and_schema.md` | Exact leakage-free Train/Validation/Test splitting procedure, `ColumnTransformer` fit/transform boundaries, output file contracts in `dataset/processed/`, and post-split validation checks. | `Data Preprocessing.pdf` (§4–5), `Machine learning.pdf` (§5) |
| 5 | `docs/notebooks/03_model_training.md` | Step-by-step guide for `03_model_training.ipynb`: mathematical formulations, training & tuning Linear Regression, Ridge Regression ($\alpha$), Lasso Regression ($\alpha$), and Decision Tree Regressor (`max_depth`, `min_samples_leaf`), validation comparison, and exporting to `models/`. | `Machine learning.pdf` (§4–5) |
| 6 | `docs/notebooks/04_model_evaluation.md` | Step-by-step guide for `04_model_evaluation.ipynb`: computing test MAE, RMSE, and $R^2$; comparison table & charts; residual & feature importance error analysis; structured answers to the 5 PDF discussion questions. | `Machine learning.pdf` (§6–7), `Week-6 Data Visualization.pdf` (§II.2, §V) |
| 7 | `docs/models/05_model_artifacts_and_prediction_demo.md` | Guide for `models/` artifacts: model serialization/versioning, retraining the selected best model on full training data, input schema validation, and building/testing the `predict_car_price()` function. | `Machine learning.pdf` (§5, §6) |

---

## 9. Implementation Dependencies and Prerequisites

### 9.1 Software & Python Environment
- **Python:** Python 3.9+ (Jupyter Notebook / JupyterLab / VS Code Jupyter extension).
- **Required Libraries:**
  - `pandas` (data loading, cleaning, deduplication, tabular inspection)
  - `numpy` (numerical operations, RMSE/log-scale calculations)
  - `matplotlib` and `seaborn` (EDA and evaluation visualizations per `Week-6 Data Visualization.pdf`)
  - `scikit-learn` (`train_test_split`, `ColumnTransformer`, `OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`, `MinMaxScaler`, `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `mean_absolute_error`, `root_mean_squared_error` / `mean_squared_error`, `r2_score`)
  - `joblib` (persisting fitted preprocessors and trained models to `dataset/processed/` and `models/`)

### 9.2 Workspace Prerequisites (Before Running Notebooks)
1. Create directories matching the reference image: `dataset/raw/`, `dataset/processed/`, and `models/`.
2. Generate `dataset/raw/car.csv` (4,000 rows × 8 columns) from `CAR DETAILS FROM CAR DEKHO.csv`.
3. Verify relative notebook paths (`../dataset/raw/car.csv`, `../dataset/processed/`, `../models/`) resolve cleanly from `notebooks/`.

---

## 10. Expected Deliverables and Completion Criteria

### 10.1 Stage-by-Stage Deliverables
1. **Approved Plan:** `PROJECT_IMPLEMENTATION_PLAN.md` (this document).
2. **Modular Documentation Suite:** All 7 step-by-step `.md` files under `docs/` mirroring the reference image hierarchy.
3. **Validated Datasets:** `dataset/raw/car.csv` (4,000 × 8) and leakage-free `dataset/processed/` splits + fitted preprocessor.
4. **Completed Notebooks (01–04):** Clean, documented, executable Jupyter notebooks producing reproducible metrics and course-aligned visualizations.
5. **Persisted Models & Inference Function:** Trained `.joblib` models in `models/` and a tested `predict_car_price()` demo outputting prices explicitly in **INR** (with optional documented USD conversion note).

### 10.2 Validation & Completion Criteria
- [ ] `dataset/raw/car.csv` verified at exactly **4,000 rows × 8 columns** with documented sampling seed (`random_state=42`).
- [ ] Notebook 01 runs end-to-end with zero errors, uses INR units, includes `km_driven` vs `selling_price` scatter plot, correlation heatmap, duplicate count, missing check, and readable Top-N brand charts.
- [ ] Notebook 02 removes exact duplicates prior to splitting, fits encoders/scalers **only on training data**, passes zero-null/zero-leakage assertions, and saves outputs to `dataset/processed/`.
- [ ] Notebook 03 trains all 4 required models (Linear, Ridge, Lasso, Decision Tree) and saves all artifacts to `models/`.
- [ ] Notebook 04 evaluates all 4 models on the untouched test set using **MAE, RMSE, and $R^2$**, includes a comparison table, residual error analysis, answers to all 5 discussion questions from `Machine learning.pdf` §7.6, and a verified prediction demo.
- [x] All documentation in `docs/` matches the reference image hierarchy with separate, non-duplicated `.md` files.
