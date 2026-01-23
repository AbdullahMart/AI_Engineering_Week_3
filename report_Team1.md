# House Prices Prediction - Project Report

**Student:** Team 1  
**Date:** January 23, 2026  
**Competition:** [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## 📊 Model Performance Summary

### Best Model
**Gradient Boosting Regressor**

### Performance Metrics
| Metric | Score |
|--------|-------|
| **Cross-Validation RMSE (5-Fold)** | 0.1101 ± 0.0121 |
| **Validation Set RMSE** | 0.1126 |
| **Mean Absolute Error (Validation)** | $9,818 |
| **Kaggle Public Leaderboard Score** | *To be submitted* |

*Note: RMSE is calculated on log-transformed prices (log1p). Lower is better.*

---

## 🎯 Top 5 Most Important Features

Based on SHAP (SHapley Additive exPlanations) analysis:

1. **OverallQual** (Overall Quality Rating)
   - Most influential feature with strong positive impact on price
   - Houses with quality 8-10 command premium prices

2. **GrLivArea** (Above Ground Living Area)
   - Second most important feature
   - Direct correlation: larger living area = higher price

3. **TotalBsmtSF** (Total Basement Area)
   - Significant positive contribution
   - Full finished basements add substantial value

4. **GarageCars** (Garage Capacity)
   - Strong predictor of home value
   - 2-3 car garages indicate higher-end properties

5. **YearBuilt** (Original Construction Year)
   - Newer homes (post-2000) command higher prices
   - Age depreciation is significant but not linear

---

## 🔍 Top 3 EDA Findings

### 1. **Right-Skewed Price Distribution**
- SalePrice shows strong right skewness (expensive outliers)
- Log transformation (log1p) normalizes distribution
- **Impact:** Justified use of log-scale modeling for better predictions

### 2. **Strong Quality-Price Correlation**
- OverallQual shows 0.79 correlation with SalePrice (highest)
- Quality ratings 8-10 have mean prices >$300,000
- Quality ratings 1-5 have mean prices <$150,000
- **Impact:** Quality is the single strongest predictor

### 3. **Living Area vs Price Outliers**
- Identified houses with GrLivArea > 4,000 sq ft but SalePrice < $200,000
- These extreme outliers (2 houses) were removed
- **Impact:** Outlier removal improved model generalization by 3-5%

---

## ⚠️ Problems Encountered & Solutions

### Problem 1: **High Missing Data (>80% in some columns)**
**Challenge:**
- 19 columns had missing values
- Some columns (PoolQC, MiscFeature, Alley) had >90% missing data
- Simple deletion would lose valuable information

**Solution:**
- Distinguished between "truly missing" vs. "None" as a category
  - Example: `GarageType=NaN` → "No Garage" (meaningful category)
- Applied SimpleImputer with strategy:
  - Numerical: `median` (robust to outliers)
  - Categorical: `most_frequent` or "None"
- Created separate list for meaningful "None" columns
- **Result:** Retained all samples, no information loss

### Problem 2: **Model Overfitting (Random Forest)**
**Challenge:**
- Random Forest showed perfect training RMSE (0.0245) but validation RMSE (0.1189)
- Gap of ~0.09 indicated severe overfitting
- Model memorizing training data instead of learning patterns

**Solution:**
- Switched to Gradient Boosting with controlled complexity:
  - `learning_rate=0.05` (slower learning)
  - `max_depth=4` (shallower trees)
  - `n_estimators=500` (more iterations with small steps)
  - `subsample=0.8` (80% sample per tree)
- Applied 5-Fold Cross-Validation for reliable evaluation
- **Result:** Gradient Boosting achieved Train RMSE (0.0536) vs. Validation RMSE (0.1126)
  - Reduced overfitting gap from 0.09 to 0.06
  - More stable and generalizable model

---

## 📈 Model Comparison

| Model | CV RMSE (Mean ± Std) | Train RMSE | Valid RMSE | Note |
|-------|---------------------|------------|------------|------|
| Ridge Regression | 0.1251 ± 0.0114 | 0.1161 | 0.1331 | Good baseline, no overfitting |
| Random Forest | 0.1181 ± 0.0134 | 0.0245 | 0.1189 | Severe overfitting |
| **Gradient Boosting** | **0.1101 ± 0.0121** | **0.0536** | **0.1126** | **Best balance** ✓ |

**Final Selection:** Gradient Boosting Regressor (best CV score with controlled overfitting)

---

## 🔧 Feature Engineering Highlights

Created **6 new features** to capture domain knowledge:

1. **TotalSF**: Combined total square footage (basement + 1st floor + 2nd floor)
2. **HouseAge**: Age of house at sale (YrSold - YearBuilt)
3. **RemodAge**: Years since remodeling (YrSold - YearRemodAdd)
4. **TotalBathrooms**: Weighted sum of all bathrooms (full + 0.5×half)
5. **TotalPorchSF**: Combined porch area (open + enclosed + screen)
6. **GarageAreaPerCar**: Garage efficiency metric (GarageArea / GarageCars)

---

## 🎯 Key Insights for Model Improvement

### Error Analysis (Top 3 Prediction Errors):

**Pattern Identified:**
- Largest errors (±$40k-60k) occur on:
  1. High-quality homes in lower-priced neighborhoods
  2. Older homes with premium renovations (not captured fully)
  3. Houses with unusual size-to-price ratios

**Future Improvements:**
1. Add neighborhood-specific pricing models
2. Create interaction features (Quality × Area, Age × Remodel)
3. Ensemble multiple model types (stacking)
4. Include external data (school districts, crime rates, walkability)
5. Apply separate models for different price segments (<$150k, $150k-$300k, >$300k)

---

## ✅ Deliverables

- ✅ **Notebook:** `Week3_HousePrices_Team1.ipynb` (71 cells, all executed)
- ✅ **Submission:** `submission_Team1.csv` (1,459 predictions)
- ✅ **Report:** This document
- ⏳ **Kaggle Score:** Pending submission upload

---

**Conclusion:** Successfully built an end-to-end ML pipeline achieving competitive performance (RMSE ~0.11 on log-scale). Gradient Boosting proved most effective, balancing accuracy and generalization. Key success factors: robust preprocessing, meaningful feature engineering, and controlled overfitting through cross-validation.
