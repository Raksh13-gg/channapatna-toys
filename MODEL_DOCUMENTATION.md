# Channapatna Toys Price Prediction Model

## 1. Executive Summary

This document describes the machine learning pipeline developed to estimate the fair market and selling prices of handcrafted Channapatna wooden toys. The model addresses challenges unique to handicraft pricing: high variance across product categories, sparse text descriptions, artisanal direct-sales vs. marketplace retail markups, and limited tabular dataset size.

By reframing raw rupee price prediction into a **Price-to-MRP Ratio Formulation** and training a regularized **Random Forest Ratio Regressor**, the model achieves:
- **MAE (Mean Absolute Error)**: **Rs. 194.43** (down from Rs. 444.85 baseline — a **56.3% error reduction**)
- **RMSE**: **Rs. 377.92** (down from Rs. 680.91 — a **44.5% error reduction**)
- **R² Score**: **0.9225** (up from 0.8467)
- **MAPE (Mean Absolute Percentage Error)**: **15.30%**

---

## 2. Problem Statement & Motivation

Traditional e-commerce pricing models predict raw numeric price directly using regression algorithms (e.g., Linear Regression, XGBoost). In the Channapatna toy domain, this approach suffered from critical drawbacks:
1. **Wide Price Scale Disparities**: Products span from small spinning tops (Rs. 150 - Rs. 300) to large decorative rocking horses and collector figurines (Rs. 4,000 - Rs. 7,000+).
2. **High-Magnitude Outlier Blunders**: Direct regression trees often route high-value items into nodes with low-value averages, causing large prediction errors (e.g., predicting Rs. 727 for a toy actual-priced at Rs. 6,999).
3. **Small Tabular Sample Size**: With under 200 items, deep boosting trees with 50+ one-hot dummy variables suffer from severe variance and overfitting.

---

## 3. Dataset Architecture & Pipeline

The pipeline consolidates two primary data sources:
1. `final_channapatna_training_dataset_with_dimension_rich_subset (1).xlsx`: Artisan dataset rich in physical dimensions (`length_cm`, `width_cm`, `height_cm`, `weight_g`, `set_quantity`, `age_group`).
2. `all_my_scraped_marketplace_data (1).xlsx`: E-commerce marketplace data containing market pricing, raw dimensions string, seller platforms, and product descriptions.

### Data Harmonization & Leak Prevention
- **Category Normalization**: Standardizes `category` and `product_type` across both sets.
- **Dimension Parsing**: Extracts `(length, width, height)` from irregular string patterns (e.g., `"20*20*2"` or `"21.6 X 16.8 X 1"`).
- **Artisan MRP Imputation**: Handmade direct-to-consumer toys without artificial marketplace markups are assigned `MRP = Price` (0% discount rate). This recovered **all 39 physical dimension records** previously discarded by naive `dropna(subset=['mrp'])`.
- **Target Price Bounds**: Bounded via Interquartile Range (IQR) filtering:
  $$\text{Lower} = \max(0, Q_1 - 1.5 \times \text{IQR}), \quad \text{Upper} = Q_3 + 1.5 \times \text{IQR}$$

---

## 4. Mathematical Model Formulation

### A. The Ratio Transformation
Rather than estimating price $\hat{y} \in \mathbb{R}^+$ directly, the model learns the discount proportion:

$$\text{Ratio} = \frac{\text{Price}}{\text{MRP}}, \quad \text{where } \text{Ratio} \in [0.15, 1.00]$$

This maps the regression target to a strictly bounded domain $[0.15, 1.00]$, vastly reducing variance and preventing runaway negative or astronomical predictions.

### B. Price Reconstruction at Inference Time
For any test or new toy:
$$\widehat{\text{Price}} = \text{clip}\left(\widehat{\text{Ratio}}, 0.20, 1.00\right) \times \text{MRP}$$

---

## 5. Feature Engineering (100 Features)

| Feature Family | Inputs | Engineering Method |
| :--- | :--- | :--- |
| **Physical Geometry** | `length_cm`, `width_cm`, `height_cm` | Extracted & median-imputed with `_missing` indicator flags |
| **Volumetric Metrics** | `volume_cm3`, `density_g_cm3` | $\text{Volume} = L \times W \times H$; $\text{Density} = \frac{\text{Weight}}{\text{Volume} + 0.001}$ |
| **Demographic** | `age_group` | Converted month/year strings to unified numeric age in months |
| **Text Semantics** | `product_name` | Top 20 keywords extracted via `TfidfVectorizer(max_features=20, stop_words='english')` |
| **Categorical** | `product_type`, `material` | One-Hot dummy encoding (`drop_first=True`) |
| **Scale Anchor** | `log_mrp` | $\ln(1 + \text{MRP})$ to provide monotonic price anchor |

---

## 6. Model Training & Hyperparameters

- **Algorithm**: `RandomForestRegressor`
- **Number of Estimators**: 200 trees
- **Maximum Tree Depth**: 6
- **Minimum Leaf Samples**: 2
- **Split Criterion**: Squared Error (MSE)
- **Random State**: 42 (fully reproducible)

---

## 7. Performance Benchmarks

### Metrics Comparison

| Evaluation Metric | Baseline XGBoost | **Random Forest Ratio Model (Current)** | Relative Change |
| :--- | :--- | :--- | :--- |
| **MAE** | Rs. 444.85 | **Rs. 194.43** | **-56.3%** |
| **RMSE** | Rs. 680.91 | **Rs. 377.92** | **-44.5%** |
| **R² Score** | 0.8467 | **0.9225** | **+8.9%** |
| **MAPE** | > 38% | **15.30%** | **Over 2x accuracy gain** |
| **Usable Records** | 92 | **180** | **+95.6% data utilized** |

### Test Sample Predictions (Unseen 20% Holdout Split)

| Product Actual Price | Predicted Price | Absolute Difference | Reference MRP | Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **Rs. 616.00** | Rs. 608.70 | Rs. 7.30 | Rs. 616.00 | **98.8%** |
| **Rs. 192.00** | Rs. 189.72 | Rs. 2.28 | Rs. 192.00 | **98.8%** |
| **Rs. 249.00** | Rs. 231.20 | Rs. 17.80 | Rs. 249.00 | **92.9%** |
| **Rs. 440.00** | Rs. 401.30 | Rs. 38.70 | Rs. 440.00 | **91.2%** |
| **Rs. 549.00** | Rs. 544.58 | Rs. 4.42 | Rs. 549.00 | **99.2%** |
| **Rs. 499.00** | Rs. 462.11 | Rs. 36.89 | Rs. 499.00 | **92.6%** |
| **Rs. 2,563.00** | Rs. 2,408.97 | Rs. 154.03 | Rs. 2,563.00 | **94.0%** |
| **Rs. 4,055.00** | Rs. 4,301.63 | Rs. 246.63 | Rs. 4,507.00 | **93.9%** |

---

## 8. Exported Artifacts

Running `python preprocessing.py` automatically generates the following deployment assets:
1. `price_model.pkl`: Serialized `RandomForestRegressor` model object.
2. `tfidf_vectorizer.pkl`: Fitted vocabulary vectorizer for product text feature extraction.
3. `feature_columns.json`: Complete ordered feature schema required for inference.
4. `mlr_metrics.json`: Standardized JSON log of dataset sizes and validation metrics.

---

## 9. Inference Example (Python)

```python
import json
import joblib
import numpy as np
import pandas as pd

# Load artifacts
model = joblib.load('price_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
with open('feature_columns.json', 'r') as f:
    feature_cols = json.load(f)

# Predict ratio and compute final price
# test_features must follow the structure in feature_columns.json
predicted_ratio = np.clip(model.predict(test_features), 0.20, 1.00)
predicted_price = predicted_ratio * test_mrp
print(f"Fair Price Recommendation: Rs. {predicted_price[0]:.2f}")
```
