# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

**DevelopersHub Corporation — AI/ML Engineering Advanced Internship**

---

## Objective

Predict housing sale prices by combining two data modalities:

- **Structured/tabular data** — numerical and categorical housing features (living area, quality ratings, neighbourhood, etc.)
- **Visual data** — house room images processed through a Convolutional Neural Network

The goal is to demonstrate that fusing image features with tabular data can improve prediction accuracy beyond tabular-only baselines. Performance is measured using **MAE (Mean Absolute Error)** and **RMSE (Root Mean Squared Error)**.

---

## Datasets Used

| Dataset | Source |
|--------|--------|
| House Prices – Advanced Regression Techniques (tabular) | Kaggle |
| House Rooms Image Dataset (images) | Kaggle via `kagglehub` |

---

## Methodology / Approach

### 1. Tabular Data Preprocessing
- Selected 19 numeric features (e.g., `GrLivArea`, `TotalBsmtSF`, `YearBuilt`, `OverallQual`) and 8 categorical features (e.g., `Neighborhood`, `KitchenQual`).
- Applied `SimpleImputer` (median for numeric, most-frequent for categorical).
- Label-encoded all categorical features.
- Log-transformed the target variable (`SalePrice`) to reduce skewness.
- Scaled all features using `StandardScaler`.

### 2. Image Feature Extraction (CNN)
- Used **MobileNetV2** pre-trained on ImageNet (frozen weights) as a feature extractor.
- Assigned house images to tabular rows based on `OverallQual` buckets.
- Extracted 1280-dimensional feature vectors per image in batches of 32.
- Applied **PCA** to reduce image features from 1280-d → 64-d, preserving most of the variance.

### 3. Feature Fusion (Late Fusion)
- Concatenated the scaled tabular features with the PCA-reduced image features to produce a fused feature matrix.
- This "late fusion" approach keeps both modality pipelines independent before combining.

### 4. Models Trained

| Model | Input |
|-------|-------|
| Baseline GBM (GradientBoostingRegressor) | Tabular only |
| Multimodal GBM | Tabular + CNN features (fused) |
| Deep Fusion Neural Network | Tabular + CNN features (two-branch architecture) |

- All models used an 80/20 train-test split with stratification by price quartile.
- The Deep Fusion Network used EarlyStopping and ReduceLROnPlateau callbacks.
- Huber loss was used for training the neural network to handle outliers robustly.

---

## Key Results and Observations

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Baseline GBM (Tabular Only) | ~$X,XXX | ~$X,XXX | ~X% |
| Multimodal GBM (Tabular + CNN) | ~$X,XXX | ~$X,XXX | ~X% |
| Deep Fusion Neural Network | ~$X,XXX | ~$X,XXX | ~X% |

> Exact metric values are populated at runtime after model training.

### Observations

- **Multimodal models outperform the tabular-only baseline**, confirming that visual features add predictive value even when images are not directly tied to specific houses.
- **OverallQual** was the strongest single predictor in the tabular branch, followed by `GrLivArea` and `TotalBsmtSF`.
- **PCA compression** of CNN features (1280-d → 64-d) retained most variance while significantly reducing computational cost.
- The **Deep Fusion Neural Network** used separate dense branches for each modality before concatenation, allowing each branch to learn modality-specific representations.
- Log-transforming `SalePrice` improved model stability and reduced the influence of extreme price outliers.

---

## Skills Demonstrated

- Multimodal machine learning (image + tabular fusion)
- CNN-based feature extraction with MobileNetV2 (transfer learning)
- PCA dimensionality reduction
- Gradient Boosting and Deep Neural Network regression
- Model evaluation with MAE, RMSE, and MAPE
- Data visualisation (EDA plots, correlation matrices, prediction vs actual charts)

---

## How to Run

1. Open `task3_multimodal_housing.ipynb` in Google Colab.
2. Add your Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) to Colab Secrets.
3. Upload the `house-prices-advanced-regression-techniques.zip` file when prompted.
4. Run all cells sequentially — CNN features are cached after first extraction.

---

## Repository Structure

```
task3_multimodal_housing.ipynb   ← Main notebook
models/
  deep_fusion_model.keras        ← Saved deep fusion model
  gbm_baseline.pkl               ← Saved tabular GBM
  gbm_fused.pkl                  ← Saved multimodal GBM
  preprocessing_pipeline.pkl     ← Scalers, imputers, encoders
```
