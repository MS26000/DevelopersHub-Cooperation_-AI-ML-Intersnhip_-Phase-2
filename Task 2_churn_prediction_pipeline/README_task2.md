# 🔄 Task 2: End-to-End ML Pipeline — Customer Churn Prediction

**DevelopersHub Corporation – AI/ML Engineering Advanced Internship**

---

## 🎯 Objective

Build a **production-ready, end-to-end machine learning pipeline** to predict customer churn using the **IBM Telco Customer Churn Dataset**. The goal is to identify customers likely to cancel their subscription so the business can intervene proactively and reduce revenue loss.

The pipeline handles everything from raw data ingestion through preprocessing, model training, hyperparameter tuning, evaluation, and serialised deployment — in a single reproducible workflow.

**Classes:**
- `0` — No Churn (customer stays)
- `1` — Churn (customer leaves)

---

## 🧠 Methodology / Approach

### 1. Dataset
- **Source:** IBM Telco Customer Churn (public GitHub mirror)
- **Size:** 7,043 customers × 21 features
- **Target:** `Churn` — binary (Yes / No)
- **Class balance:** 73.5% No Churn (5,174) / 26.5% Churn (1,869) — imbalanced
- `customerID` dropped (no predictive value); `TotalCharges` coerced to numeric (11 blanks → NaN)

### 2. Feature Engineering & Preprocessing

Features were split into three types and processed via `ColumnTransformer`:

| Type | Count | Features | Preprocessing |
|------|-------|----------|---------------|
| Numeric | 3 | tenure, MonthlyCharges, TotalCharges | Impute (median) → StandardScaler |
| Binary | 6 | gender, SeniorCitizen, Partner, Dependents, PhoneService, PaperlessBilling | Impute (mode) → OrdinalEncoder |
| Nominal | 10 | Contract, InternetService, PaymentMethod, MultipleLines, etc. | Impute (mode) → OneHotEncoder |

After encoding: **19 raw features → 40 features**. All preprocessing was fitted only on training data — no data leakage.

### 3. Train / Test Split
- **80/20 stratified split** — preserves the 73.5/26.5% class ratio in both sets
- Train: **5,634 samples** | Test: **1,409 samples**

### 4. Models Compared

Three classifiers were built as full scikit-learn `Pipeline` objects (preprocessor + classifier) with `class_weight="balanced"` to handle class imbalance:

- Logistic Regression (`max_iter=1000`)
- Random Forest (`n_estimators=100`)
- Gradient Boosting (`n_estimators=200, learning_rate=0.08, max_depth=4`)

### 5. Hyperparameter Tuning
- **Method:** `GridSearchCV` with 5-fold `StratifiedKFold` cross-validation
- **Scoring:** ROC-AUC
- **Search space:** 54 combinations × 5 folds = **270 fits**
- **Parameters tuned:** `n_estimators` [100, 200, 300], `learning_rate` [0.05, 0.08, 0.1], `max_depth` [3, 4, 5], `subsample` [0.8, 1.0]
- **Best params found:** `learning_rate=0.05, max_depth=3, n_estimators=100, subsample=0.8`
- **Tuning time:** ~10.5 minutes

### 6. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix (raw + normalised)
- Per-class classification report
- 5-fold cross-validation AUC with 95% confidence interval

### 7. Persistence & Inference
- Full pipeline serialised with `joblib` → `churn_pipeline.joblib` (0.2 MB)
- Metadata saved to `pipeline_metadata.json`
- Pipeline reloaded and verified — predictions match exactly
- Tested on 5 synthetic new customers covering high-risk and low-risk profiles

---

## 📊 Key Results & Observations

### Model Comparison (Test Set — Before Tuning)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time |
|-------|----------|-----------|--------|-----|---------|------------|
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8414 | 0.17s |
| Random Forest | 0.7771 | 0.6000 | 0.4813 | 0.5341 | 0.8160 | 1.31s |
| Gradient Boosting | 0.7899 | 0.6283 | 0.5107 | 0.5634 | 0.8350 | 3.89s |

### Tuned Gradient Boosting vs Baseline

| Metric | Baseline | Tuned | Δ |
|--------|----------|-------|---|
| Accuracy | 0.7899 | **0.8041** | +0.0142 |
| Precision | 0.6283 | **0.6725** | +0.0442 |
| Recall | 0.5107 | **0.5107** | +0.0000 |
| F1 | 0.5634 | **0.5805** | +0.0171 |
| ROC-AUC | 0.8350 | **0.8467** | +0.0117 |

### Detailed Classification Report — Tuned GB (Best Model)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| No Churn | 0.84 | 0.91 | 0.87 | 1,035 |
| Churn | 0.67 | 0.51 | 0.58 | 374 |
| **Weighted Avg** | **0.79** | **0.80** | **0.79** | **1,409** |

### 5-Fold Cross-Validation (Tuned GB)

| Fold | AUC |
|------|-----|
| 1 | 0.8604 |
| 2 | 0.8447 |
| 3 | 0.8580 |
| 4 | 0.8341 |
| 5 | 0.8394 |
| **Mean** | **0.8473 ± 0.0103** |
| **95% CI** | **(0.8267, 0.8679)** |

### Top 5 Churn Predictors (Feature Importance)

| Feature | Importance |
|---------|-----------|
| Contract_Month-to-month | 0.3792 |
| tenure | 0.1517 |
| InternetService_Fiber optic | 0.0884 |
| OnlineSecurity_No | 0.0780 |
| MonthlyCharges | 0.0717 |

### Inference on New Customers

| # | Contract | Tenure | Monthly$ | Churn% | Verdict |
|---|----------|--------|----------|--------|---------|
| 1 | Month-to-month | 1 | $85 | 82.1% | ⚠️ LIKELY CHURN |
| 2 | Two year | 60 | $45 | 2.7% | ✅ LIKELY STAY |
| 3 | Month-to-month | 12 | $70 | 68.0% | ⚠️ LIKELY CHURN |
| 4 | One year | 36 | $55 | 5.4% | ✅ LIKELY STAY |
| 5 | Two year | 72 | $20 | 2.3% | ✅ LIKELY STAY |

### Key Observations

1. **Gradient Boosting is the best model** — highest ROC-AUC (0.8467) after tuning, better precision than Logistic Regression, and more stable across folds.

2. **Logistic Regression has the best recall (0.78)** — it catches more churners but at the cost of many false positives (lower precision 0.50). Useful if missing a churner is very costly.

3. **Tuning improved precision significantly (+0.044)** — the best params (`max_depth=3, lr=0.05`) prevent overfitting and produce a more conservative, reliable model.

4. **Month-to-month contract is the strongest churn signal** — with an importance score of 0.38, it dwarfs all other features. Customers without long-term commitment churn at 3–4× the rate.

5. **Tenure and monthly charges work together** — high charges for short-tenure customers is a dangerous combination, representing customers who haven't yet seen enough value to justify the cost.

6. **5-fold CV confirms no overfitting** — AUC of 0.8473 ± 0.0103 across folds shows the model generalises well and the test set result (0.8467) is reliable.

7. **ROC-AUC is the right primary metric** — with 73/27 imbalance, raw accuracy is misleading (predicting "No Churn" for everyone scores 73.5%). AUC measures discrimination across all thresholds regardless of class balance.

---

## 💾 Saved Outputs

| File | Description |
|------|-------------|
| `churn_pipeline.joblib` | Full serialised pipeline — preprocessor + tuned GB (0.2 MB) |
| `pipeline_metadata.json` | Best params, CV AUC, test metrics, feature names |
| `eda_overview.png` | Churn distribution, tenure, charges, contract, internet, payment charts |
| `correlation_scatter.png` | Numeric correlation matrix + tenure vs charges scatter by churn |
| `evaluation_dashboard.png` | Confusion matrix, ROC curves, all-model metrics comparison |
| `feature_importance.png` | Top 20 feature importances from tuned Gradient Boosting |
| `cross_validation.png` | 5-fold CV ROC-AUC bar chart with mean line |

---

## 🏗️ Pipeline Architecture

```
Raw CSV (7,043 rows × 21 cols)
    │
    ▼
ColumnTransformer
    ├── Numeric  (3)  → Impute (median)        → StandardScaler
    ├── Binary   (6)  → Impute (most_frequent)  → OrdinalEncoder
    └── Nominal  (10) → Impute (most_frequent)  → OneHotEncoder
    │                                           → 40 features total
    ▼
GradientBoostingClassifier
    └── Tuned via GridSearchCV (54 combos × 5-fold CV = 270 fits)
    └── Best: lr=0.05, max_depth=3, n_estimators=100, subsample=0.8
    │
    ▼
joblib.dump → churn_pipeline.joblib (0.2 MB)
    │
    ▼
joblib.load → .predict() / .predict_proba() on new customers
```

---

## 🔭 Future Improvements

- Apply **SMOTE oversampling** inside the pipeline to generate synthetic minority class samples and improve recall on the churn class
- Try **XGBoost / LightGBM** for faster training and potentially higher AUC
- Add **SHAP values** for individual prediction explanation ("why is this customer likely to churn?")
- Build a **Streamlit dashboard** for business users to input customer data interactively
- Use **threshold tuning** (e.g. 0.35 instead of 0.5) to further improve recall on churners
- Implement **model monitoring** — retrain automatically when data drift is detected
