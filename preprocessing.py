import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------
print("Loading datasets...")
df1 = pd.read_excel('final_channapatna_training_dataset_with_dimension_rich_subset (1).xlsx')
df2 = pd.read_excel('all_my_scraped_marketplace_data (1).xlsx')

# Standardize category / product_type naming
df2 = df2.rename(columns={'category': 'product_type'})

# Parse dimensions from 'size' in df2 (e.g., '20*20*2' or '21.6 X 16.8 X 1')
def parse_size(s):
    if pd.isna(s) or 'free' in str(s).lower():
        return np.nan, np.nan, np.nan
    s = str(s).lower().replace(' x ', '*').replace('x', '*')
    parts = s.split('*')
    if len(parts) >= 3:
        try:
            return float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())
        except Exception:
            pass
    return np.nan, np.nan, np.nan

df2[['length_cm', 'width_cm', 'height_cm']] = df2['size'].apply(lambda x: pd.Series(parse_size(x)))

# In df1, handmade artisan toys have direct selling prices without separate marketplace MRP
if 'mrp' not in df1.columns:
    df1['mrp'] = df1['price']
else:
    df1['mrp'] = df1['mrp'].fillna(df1['price'])

# Align missing columns across both datasets
for col in ['weight_g', 'set_quantity', 'age_group']:
    if col not in df2.columns:
        df2[col] = np.nan

# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)
df.drop_duplicates(subset=['product_name', 'price'], inplace=True)

# -------------------------------------------------------------
# 2. Cleaning & Feature Engineering
# -------------------------------------------------------------
# Clean prices and MRPs
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['mrp'] = pd.to_numeric(df['mrp'], errors='coerce').fillna(df['price'])
df.dropna(subset=['price'], inplace=True)
df = df[df['price'] > 0]
df['mrp'] = df['mrp'].clip(lower=df['price'])  # MRP cannot be lower than selling price

pd.set_option('future.no_silent_downcasting', True)

# Clean age_group with explicit numeric mapping
age_map = {
    '1 Year': 12, '1 Years': 12, '12 Months': 12,
    '2 Years': 24, '3 Years': 36,
    '6-24 Months': 15, '3 Months': 3
}
df['age_group'] = df['age_group'].replace(age_map)
df['age_group'] = pd.to_numeric(df['age_group'], errors='coerce')
age_median = df['age_group'].dropna().median() if not df['age_group'].dropna().empty else 12.0
df['age_group'] = df['age_group'].fillna(age_median)

# Clean categorical fields
df['product_type'] = df['product_type'].fillna('unknown').astype(str).str.lower().str.strip()
df['material'] = df['material'].fillna('unknown').astype(str).str.lower().str.strip()

# Numeric imputation with missing flags
numeric_cols = ['length_cm', 'width_cm', 'height_cm', 'weight_g', 'set_quantity']
for col in numeric_cols:
    if col not in df.columns:
        df[col] = np.nan
    df[col + '_missing'] = df[col].isna().astype(int)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    med_val = df[col].dropna().median()
    df[col] = df[col].fillna(med_val if pd.notna(med_val) else 0.0)

# Calculated physical features
df['volume_cm3'] = df['length_cm'] * df['width_cm'] * df['height_cm']
df['density_g_cm3'] = df['weight_g'] / (df['volume_cm3'] + 0.001)

# Target formulation: Price-to-MRP ratio (bounded between 0.2 and 1.0)
df['price_to_mrp_ratio'] = (df['price'] / df['mrp']).clip(lower=0.15, upper=1.0)

# Filter extreme price outliers via IQR for clean learning
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = max(0, Q1 - 1.5 * IQR)
upper_bound = Q3 + 1.5 * IQR
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()

# Text feature extraction using TF-IDF on product titles
df['product_name'] = df['product_name'].fillna('').astype(str)
tfidf = TfidfVectorizer(max_features=20, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['product_name'])
tfidf_cols = [f"word_{w}" for w in tfidf.get_feature_names_out()]
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
df = pd.concat([df, tfidf_df], axis=1)

# Categorical One-Hot Encoding
df = pd.get_dummies(df, columns=['product_type', 'material'], drop_first=True)

# -------------------------------------------------------------
# 3. Features & Train/Test Split
# -------------------------------------------------------------
# Exclude raw target, identifiers, and metadata from features
exclude_cols = [
    'record_id', 'product_name', 'size', 'price', 'price_to_mrp_ratio',
    'platform', 'seller', 'product_url', 'source_type', 'data_quality',
    'collection_date', 'verification_note', 'rating', 'review_count', 'discount'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]

# Log-transform MRP to give linear scale to the model
df['log_mrp'] = np.log1p(df['mrp'])
if 'log_mrp' not in feature_cols:
    feature_cols.append('log_mrp')

X = df[feature_cols].astype(float)
y_ratio = df['price_to_mrp_ratio']
y_actual_price = df['price']
mrp_series = df['mrp']

X_train, X_test, y_train_ratio, y_test_ratio, mrp_train, mrp_test, y_train_price, y_test_price = train_test_split(
    X, y_ratio, mrp_series, y_actual_price, test_size=0.2, random_state=42
)

print(f"\nData successfully prepared:")
print(f"Total usable records: {len(df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of engineered features: {X.shape[1]}")

# -------------------------------------------------------------
# 4. Model Training (Ratio Regressor)
# -------------------------------------------------------------
print("\nTraining Random Forest Regressor on Price/MRP Ratio...")
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train_ratio)

# -------------------------------------------------------------
# 5. Prediction & Evaluation
# -------------------------------------------------------------
predicted_ratios = np.clip(model.predict(X_test), 0.20, 1.0)
predicted_prices = np.round(predicted_ratios * mrp_test, 2)

mae = mean_absolute_error(y_test_price, predicted_prices)
rmse = np.sqrt(mean_squared_error(y_test_price, predicted_prices))
r2 = r2_score(y_test_price, predicted_prices)
mape = np.mean(np.abs((y_test_price - predicted_prices) / y_test_price)) * 100

print("\n" + "=" * 45)
print("       EVALUATION METRICS (NEW MODEL)")
print("=" * 45)
print(f"MAE  (Mean Absolute Error): Rs. {mae:.2f}")
print(f"RMSE (Root Mean Sq. Error): Rs. {rmse:.2f}")
print(f"R2 Score:                   {r2:.4f}")
print(f"MAPE (Mean % Error):        {mape:.2f}%")
print("=" * 45)

# Comparison table
comparison_df = pd.DataFrame({
    'Actual_Price': y_test_price.values,
    'Predicted_Price': predicted_prices.values,
    'Diff': np.abs(np.round(y_test_price.values - predicted_prices.values, 2)),
    'MRP': mrp_test.values
})
print("\nSample Predictions on Unseen Test Toys (First 10):")
print(comparison_df.head(10).to_string(index=False))

# -------------------------------------------------------------
# 6. Save Model Artifacts
# -------------------------------------------------------------
joblib.dump(model, 'price_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f, indent=2)

metrics_summary = {
    "model": "RandomForestRegressor_RatioModel",
    "target": "price_to_mrp_ratio -> price",
    "total_samples": len(df),
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "features_count": len(feature_cols),
    "mae": round(float(mae), 2),
    "rmse": round(float(rmse), 2),
    "r2_score": round(float(r2), 4),
    "mape_percent": round(float(mape), 2)
}

with open('mlr_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print("\nTrained model and metrics saved successfully to 'price_model.pkl' and 'mlr_metrics.json'!")
