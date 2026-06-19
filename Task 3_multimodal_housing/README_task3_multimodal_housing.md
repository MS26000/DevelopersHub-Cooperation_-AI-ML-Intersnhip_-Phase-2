# Task 3: Multimodal ML – Housing Price Prediction Using Images + Tabular Data

**DevelopersHub Corporation — AI/ML Engineering Advanced Internship**

---

## Objective

Predict housing sale prices by combining two data modalities:

- **Structured/tabular data** — numerical and categorical housing features (living area, quality ratings, neighbourhood, etc.)
- **Visual data** — house room images processed through a Convolutional Neural Network

The goal is to test whether fusing image features with tabular data improves prediction accuracy over a tabular-only baseline. Performance is measured using **MAE**, **RMSE**, and **MAPE**.

---

## Datasets Used

| Dataset | Source | Size |
|--------|--------|------|
| House Prices – Advanced Regression Techniques (tabular) | Kaggle | 1,460 houses × 81 raw columns (27 selected features used) |
| House Rooms Image Dataset (images) | Kaggle via `kagglehub` | 5,250 images across 5 room categories |

Image category breakdown: Living Room (1,273), Bedroom (1,248), Dining (1,158), Kitchen (965), Bathroom (606).

> **Note on image–house pairing:** This dataset does not provide a real photo for each specific house. Each house was assigned one room image at random from a pool bucketed by its `OverallQual` rating, so the image features carry only a coarse, quality-bucket-level signal rather than house-specific visual detail. This limitation is directly relevant to the results below.

---

## Methodology / Approach

### 1. Tabular Data Preprocessing
- Selected 19 numeric features (e.g., `GrLivArea`, `TotalBsmtSF`, `YearBuilt`, `OverallQual`) and 8 categorical features (e.g., `Neighborhood`, `KitchenQual`) → 27 features total.
- Applied `SimpleImputer` (median for numeric, most-frequent for categorical).
- Label-encoded all categorical features.
- Log-transformed the target variable (`SalePrice`, via `log1p`) to reduce skewness.
- Scaled all features using `StandardScaler`.

### 2. Image Feature Extraction (CNN)
- Used **MobileNetV2** pre-trained on ImageNet (frozen weights, `pooling='avg'`) as a feature extractor.
- Assigned one room image per house based on `OverallQual` buckets (see note above).
- Extracted 1,280-dimensional feature vectors per image in batches of 32.
- Applied **PCA** to reduce image features from 1,280-d → 64-d, retaining **63.9%** of variance.

### 3. Feature Fusion (Late Fusion)
- Concatenated the scaled tabular features (27-d) with the PCA-reduced image features (64-d) → **91-d fused feature matrix**.
- This "late fusion" approach keeps both modality pipelines independent before combining.

### 4. Models Trained

| Model | Input |
|-------|-------|
| Baseline GBM (GradientBoostingRegressor) | Tabular only (27-d) |
| Multimodal GBM | Tabular + CNN features, fused (91-d) |
| Deep Fusion Neural Network | Tabular + CNN features, two-branch architecture |

- Data split: 992 train / 176 validation / 292 test, stratified by `SalePrice` quartile.
- The Deep Fusion Network used separate dense branches per modality (128→64 units each with BatchNorm + Dropout), concatenated and passed through further dense layers, trained with **Huber loss**, `EarlyStopping`, and `ReduceLROnPlateau` (best weights restored from epoch 46 of 66).

---

## Key Results (actual, from notebook run)

All metrics are computed on the held-out test set (292 houses), on the original dollar scale (predictions exponentiated back via `expm1`):

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Baseline GBM (Tabular Only) | **$19,158** | **$39,812** | **9.71%** |
| Multimodal GBM (Tabular + CNN) | $21,000 | $43,516 | 10.75% |
| Deep Fusion Neural Network (Tabular + CNN) | $103,448 | $521,546 | 43.78% |

**The tabular-only baseline GBM was the best-performing model on every metric.** Adding image features made the GBM slightly worse (MAE +9.6%, RMSE +9.3%), and the deep fusion network performed dramatically worse than both gradient-boosted models.

---

## Observations

- **The tabular-only baseline outperformed both multimodal models.** This contradicts the common assumption that adding a second modality automatically helps — in this case it didn't, for a specific reason (see below).
- **Root cause: weak image–house correspondence.** Because images were assigned to houses via a random draw within a coarse `OverallQual` bucket rather than being actual photos of each house, the CNN/PCA image features mostly encode noise plus a faint quality-bucket signal that the tabular `OverallQual` feature already captures directly. The image branch therefore added redundant noise rather than new information.
- **The Deep Fusion NN's training curves looked reasonable** (validation MAE on the log-scale target dropped to ~0.33–0.5 by the best epoch), but the test-set dollar-scale error was very large and the RMSE/MAE ratio (~5×) suggests a small number of severely mispredicted houses — consistent with `expm1` amplifying small log-scale errors into large dollar errors for higher-priced homes. This is a known risk of training on a log target and evaluating after back-transforming.
- **PCA compression** of CNN features (1,280-d → 64-d) retained 63.9% of variance — a meaningful compression, but also a meaningful information loss, which may have further diluted whatever weak visual signal existed.
- **Log-transforming `SalePrice`** is standard practice for this dataset and improved training stability for the neural network, but it also means dollar-scale evaluation metrics are sensitive to how well the model performs on the log scale at the tails.
- **Takeaway:** the experiment is a useful negative result. Multimodal fusion is only as good as the alignment between modalities — synthetic or loosely matched image-to-record pairing can hurt rather than help, even when each pipeline (CNN extraction, PCA, fusion architecture) is implemented correctly.

---

## Skills Demonstrated

- Multimodal machine learning (image + tabular fusion), including honest evaluation of when fusion *doesn't* help
- CNN-based feature extraction with MobileNetV2 (transfer learning)
- PCA dimensionality reduction
- Gradient Boosting and Deep Neural Network regression (two-branch fusion architecture)
- Model evaluation with MAE, RMSE, and MAPE on a back-transformed target
- Data visualisation (EDA plots, correlation matrices, training curves, prediction-vs-actual and residual plots, feature importance)

---

## How to Run

1. Open `task3_multimodal_housing.ipynb` in Google Colab.
2. Add your Kaggle credentials (`KAGGLE_USERNAME`, `KAGGLE_KEY`) to Colab Secrets.
3. Upload the `house-prices-advanced-regression-techniques.zip` file when prompted.
4. Run all cells sequentially — CNN features are cached (`/content/data/cnn_features.npy`) after first extraction to avoid recomputation.

---

## Repository Structure

```
task3_multimodal_housing.ipynb   ← Main notebook
models/
  deep_fusion_model.keras        ← Saved deep fusion model (731.8 KB)
  gbm_baseline.pkl                ← Saved tabular-only GBM (563.7 KB)
  gbm_fused.pkl                   ← Saved multimodal GBM (599.9 KB)
  preprocessing_pipeline.pkl      ← Scalers, imputers, encoders (334.0 KB)
```
