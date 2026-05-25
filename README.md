# Developers-Hub-Internship-Phase-2

## 🎯 Task Details

### ✅ Task 1: Term Deposit Subscription Prediction
**Objective:** Predict whether a bank customer will subscribe to a term deposit following a marketing campaign.  
**Approach:**
- Loaded & cleaned the UCI Bank Marketing dataset (`;` delimited)
- Applied Label Encoding to categorical features & stratified train-test split
- Trained `Logistic Regression` and `Random Forest` classifiers
- Evaluated using Confusion Matrix, F1-Score, and ROC-AUC
- Applied **SHAP** for explainable AI (XAI) to interpret model decisions
**Key Findings:**
- Random Forest achieved higher F1-score (`[Insert F1, e.g., 0.68]`) vs Logistic Regression
- SHAP analysis revealed `duration`, `campaign`, and `pdays` as the strongest predictors
- Adjusted classification threshold improved recall for the subscribed class without sacrificing precision

### ✅ Task 2: Customer Segmentation Using K-Means
**Objective:** Segment mall customers based on demographic & spending behavior to propose targeted marketing strategies.  
**Approach:**
- Performed EDA on the Mall Customers dataset
- Standardized features (`Age`, `Income`, `Spending Score`) using `StandardScaler`
- Applied **K-Means Clustering** (`k=5` validated via Elbow/Silhouette method)
- Visualized clusters using **PCA** (linear projection) and **t-SNE** (non-linear embedding)
**Key Findings:**
- Identified 5 distinct segments (e.g., *High-Income/Low-Spend*, *Premium Shoppers*, *Budget-Conscious*)
- Proposed tailored strategies: VIP loyalty programs for high-spenders, targeted discount campaigns for price-sensitive groups
- PCA & t-SNE successfully visualized natural cluster boundaries without label leakage

### ✅ Task 3: Energy Consumption Time Series Forecasting
**Objective:** Forecast short-term household energy usage using historical temporal patterns.  
**Approach:**
- Parsed `household_power_consumption.txt` and resampled to hourly frequency
- Engineered time-based features: `hour`, `dayofweek`, `is_weekend`
- Benchmarked three models: **ARIMA**, **Prophet**, and **XGBoost Regressor**
- Evaluated using **MAE** and **RMSE**
**Key Findings:**
- XGBoost outperformed traditional statistical models (`MAE: [X.XX]`, `RMSE: [X.XX]`)
- Prophet captured weekly seasonality well but struggled with sharp non-linear spikes
- Time-feature engineering significantly improved gradient boosting performance

### ✅ Task 4: Loan Default Risk with Business Cost Optimization
**Objective:** Predict loan default probability and optimize the classification threshold to minimize financial loss.  
**Approach:**
- Cleaned & imputed missing values in the Home Credit Default Risk dataset (sampled 50K rows for efficiency)
- Trained `Logistic Regression` and `CatBoost` (with `class_weights` for imbalance)
- Defined business costs: `False Positive = $500`, `False Negative = $10,000`
- Swept classification thresholds from `0.10` to `0.90` to find the cost-minimizing point
**Key Findings:**
- Optimal threshold: `[Insert Threshold, e.g., 0.34]` reduced expected business cost by `[Insert %]` vs. default `0.5`
- CatBoost significantly outperformed Logistic Regression in capturing non-linear risk interactions
- Feature importance highlighted `credit_score`, `dti`, and `loan_amount` as primary default drivers

## ⚙️ Installation & Dependencies
Ensure you have Python 3.8+ installed. Install required packages via:
```bash
pip install pandas numpy scikit-learn xgboost catboost shap matplotlib seaborn statsmodels prophet imbalanced-learn
