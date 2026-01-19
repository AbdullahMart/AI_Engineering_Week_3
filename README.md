# AI_Engineering_Week_3

# House Prices (End-to-End ML Pipeline)

### 📌 Project Title
**House Prices: Advanced Regression Techniques** [Kaggle Competition Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

### 🎯 Project Goal
The goal of this project is to develop a machine learning model that predicts the sales price of a house (**SalePrice**).

**With this project, you will learn:**

* **Exploratory Data Analysis (EDA):** Reading and exploring the data.
* **Data Cleaning:** Handling missing values and cleaning the dataset.
* **Feature Engineering:** Creating new features from existing data.
* **Preprocessing Pipeline:** Building a structure with `ColumnTransformer` and `Pipeline`.
* **Model Training:** Training and comparing multiple machine learning models.
* **Cross Validation:** Evaluating models reliably.
* **Overfitting Control:** Comparing training vs. validation results.
* **Model Explainability:** Using SHAP to understand how the model works.
* **Kaggle Submission:** Preparing and submitting results.
* **Error Analysis:** Analyzing the mistakes of the model.

---

### 📦 Dataset Link
[Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

**Dataset Files:**
* `train.csv`: Contains the target variable (**SalePrice**).
* `test.csv`: Does not contain the target variable.
* `data_description.txt`: Full description of each column.
* `sample_submission.csv`: An example file for submission.

---

### ✅ Deliverables (Mandatory)

**1. Notebook (MANDATORY)** File name format: `Week3_HousePrices_FirstNameLastName.ipynb`  
* The notebook must run without any errors from start to finish.  
* All outputs must be generated (graphs, tables, metrics).

**2. Kaggle Submission File (MANDATORY)** File name format: `submission_FirstNameLastName.csv`  
* Required columns: `Id`, `SalePrice`.

**3. Short Report (MANDATORY)** File name format: `report_FirstNameLastName.pdf` or `report_FirstNameLastName.md`  
The report should be 1 page and must include:  
* Which model is your best?  
* Your Cross-Validation (CV) score.  
* Your Kaggle score.  
* Top 5 most important features.  
* Top 3 EDA findings.  
* 2 problems you encountered and your solutions.

---

### 🧠 Project Rules

**✅ Minimum Requirements:** 
* **EDA:** Exploratory Data Analysis.  
* **Cleaning:** Data cleaning process.  
* **Preprocessing:** Scaling and encoding.  
* **Feature Engineering:** Creating new variables.  
* **Models:** Training at least 3 different models.  
* **Cross Validation:** Reliable performance measurement.  
* **Model Comparison Table:** Summary of all model results.  
* **SHAP:** Explaining model predictions.  
* **Kaggle Submission:** Uploading results to the platform.  
* **Error Analysis:** Analyzing 3 incorrect predictions.

---

### 🧩 PROJECT STEPS (Notebook Structure)

#### ✅ A) Data Loading and Inspection (MANDATORY)
**Tasks:**
* Read `train.csv` and `test.csv`.
* Print the dimensions of the datasets (`shape`).
* Show the first 5 rows (`head`).
* Inspect data types (`info`).
* Review the numerical summary (`describe`).
* Check for duplicate records.

#### ✅ B) EDA (Exploratory Data Analysis) (MANDATORY)

**1. Target Analysis (SalePrice)**
* **Required Charts:** Histogram of `SalePrice` and Histogram of `log1p(SalePrice)`.
* **Goal:** Are the sales prices right-skewed? Does the Log transformation provide a more normal distribution?

**2. Missing Value Analysis**
* List the top 20 columns with the most missing values.
* Calculate the percentage of missing values.
* Create a missing value chart (Bar plot is recommended).

**3. Correlation Analysis (Numerical)**
* Find the top 10 numerical features with the highest correlation to `SalePrice`.
* Create a small heatmap using these features.

**4. Feature vs. SalePrice Visualizations (MANDATORY)**
* Create at least 4 charts (Recommended: `OverallQual`vs `SalePrice`, `GrLivArea`vs `SalePrice`, `YearBuilt`vs `SalePrice`, `GarageCars` vs `SalePrice`).
* **Note:** Write 1-2 sentences of commentary after each chart.

#### ✅ C) Data Cleaning (MANDATORY)

**1. Filling Missing Values**
* **Numerical columns:** Use `median`.
* **Categorical columns:** Use `most_frequent` or "None".
* *Note: In some columns like `GarageType`, "None" might be a meaningful category.*

**2. Outlier Handling (MANDATORY)**
Apply at least one outlier method.
* **Example:** Identify and remove rows where `GrLivArea` is very high but `SalePrice` is low, or apply capping to extreme values.
* **Goal:** Reduce incorrect learning for the model.

---

### ✅ D) Feature Engineering (MANDATORY)
You must create at least **5 new features**.

**Recommended Features:**
* **TotalSF:** `TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
* **HouseAge:** `YrSold - YearBuilt`
* **RemodAge:** `YrSold - YearRemodAdd`
* **TotalBathrooms:** `FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath`
* **TotalPorchSF:** `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch`

*Note: For each new feature, write a 1-sentence explanation of "Why it might be useful."*

---

### ✅ E) Preprocessing Pipeline (MANDATORY)
**Requirements:**
* Separate numerical and categorical features.
* Use `SimpleImputer` for missing values.
* Use `OneHotEncoder` for categorical variables.
* Build a structure using `ColumnTransformer` + `Pipeline`.

*Note: Scaling is not mandatory but recommended for linear models.*

---

### ✅ F) Model Training (MANDATORY)
You must train at least **3 different regression models**.

**Minimum Requirements:**
* **Ridge or Lasso:** (Linear Model)
* **RandomForestRegressor**
* **GradientBoostingRegressor:** (Alternatively: LightGBM or XGBoost)

---

### ✅ G) Evaluation (MANDATORY)
Kaggle uses log-based error (similar to RMSLE) for this competition. Therefore, it is recommended to use **y = log1p(SalePrice)** for modeling.

**Requirements:**
* **K-Fold Cross Validation:** (K=5 is recommended)
* **Metric:** Calculate **RMSE** as your score.

**For each model, you must include:**
* CV mean score
* CV std (standard deviation) score

---

### ✅ H) Model Comparison Table (MANDATORY)
You must compare your 3 models in a single table.

**Your table should include these columns:**
* **Model:** Name of the model.
* **CV RMSE Mean:** Average cross-validation score.
* **CV RMSE Std:** Standard deviation of CV scores.
* **Train RMSE:** Error on the training set.
* **Valid RMSE:** Error on the validation set.
* **Note:** Short comments like *"could be overfitting"*, *"more stable"*, or *"good baseline"*.

*Note: You are expected to create this table using a `pandas.DataFrame`.*

---

### ✅ I) Overfitting Control (MANDATORY)
**Required Analysis:**
* Compare **Train RMSE** vs. **Validation RMSE**.
* If Train is very good but Validation is poor → **Overfitting**.
* **Short Commentary:** Is the model showing high variance or high bias?

---

### ✅ J) Explainability with SHAP (MANDATORY)
Apply SHAP to your best-performing model.

**Required Outputs:**
* **SHAP summary plot.**
* **Local explanation:** Explanation for a single house (one sample).
* **Minimum Commentary:** Which 5 features are the most effective? Do these features increase or decrease the price?

---

### ✅ K) Kaggle Submission (MANDATORY)
**Tasks:**
* Retrain your best model using the entire training set.
* Generate predictions on `test.csv`.
* Create the `submission.csv` file.
* Upload it to Kaggle and include your score in the report.

---

### ✅ L) Error Analysis: 3 Incorrect Predictions (MANDATORY)
The goal is to understand where the model makes mistakes.

**Required Steps:**
1. Make predictions on the validation set.
2. Compare the actual value with the prediction.
3. Calculate absolute error: `abs_error = abs(y_true - y_pred)`.
4. Find the **3 houses with the largest errors** and analyze them individually.

**For each example, answer these questions:**
* Which features are "extreme" in this house?
* Is the house very large but the price is low?
* Is the neighborhood or quality information unusual?
* Could the error be caused by missing data imputation?
* Could the outlier cleaning be insufficient?

*Note: This analysis will give you ideas for future model improvements.*
