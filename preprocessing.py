import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df1 = pd.read_excel('final_channapatna_training_dataset_with_dimension_rich_subset (1).xlsx')
df2 = pd.read_excel('all_my_scraped_marketplace_data (1).xlsx')

# Rename category to match the original product_type
df2 = df2.rename(columns={'category': 'product_type'})

# Parse 'size' (e.g. '20*20*2' or '21.6 X 16.8 X 1') into dimensions
def parse_size(s):
    if pd.isna(s) or 'free' in str(s).lower():
        return np.nan, np.nan, np.nan
    s = str(s).lower().replace(' x ', '*').replace('x', '*')
    parts = s.split('*')
    if len(parts) >= 3:
        try:
            return float(parts[0].strip()), float(parts[1].strip()), float(parts[2].strip())
        except:
            pass
    return np.nan, np.nan, np.nan

df2[['length_cm', 'width_cm', 'height_cm']] = df2['size'].apply(lambda x: pd.Series(parse_size(x)))

# Ensure missing numeric columns exist in df2 so they can be filled with median later
for col in ['weight_g', 'set_quantity', 'age_group']:
    if col not in df2.columns:
        df2[col] = np.nan

# Combine both datasets
df = pd.concat([df1, df2], ignore_index=True)

# 3. Handle Future Features
future_text_cols = ['design_complexity', 'painting_type', 'craftsmanship']
for col in future_text_cols:
    if col not in df.columns:
        df[col] = 'unknown'

if 'number_of_parts' not in df.columns:
    df['number_of_parts'] = np.nan
df.drop_duplicates(inplace=True)
df['product_type'] = df['product_type'].str.lower().str.strip().fillna('unknown')
df['material'] = df['material'].str.lower().str.strip().fillna('unknown')
df['age_group'] = df['age_group'].replace({
    '1 Year': 12, '1 Years': 12, '12 Months': 12,
    '2 Years': 24, '3 Years': 36,
    '6-24 Months': 15, '3 Months': 3
}).infer_objects(copy=False)
df['age_group'] = pd.to_numeric(df['age_group'], errors='coerce')
df['age_group'] = df['age_group'].fillna(df['age_group'].median())

# 1. Drop missing MRPs so we don't ruin the model with a fake median
df['mrp'] = pd.to_numeric(df['mrp'], errors='coerce')
df.dropna(subset=['mrp'], inplace=True)

numeric_cols = ['length_cm', 'width_cm', 'height_cm', 'weight_g', 'set_quantity', 'number_of_parts']
for col in numeric_cols:
    df[col+'_missing'] = df[col].isna().astype(int)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median()).fillna(0)
df['volume_cm3'] = df['length_cm'] * df['width_cm'] * df['height_cm']
df['density_g_cm3'] = df['weight_g'] / (df['volume_cm3'] + 0.001)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df.dropna(subset=['price'], inplace=True)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
# 7. Check your features before doing anything else
print("\nColumns:")
print(df.columns.tolist())

print("\nPrice statistics:")
print(df['price'].describe())

print("\nNumeric correlations:")
print(df.corr(numeric_only=True)['price'].sort_values())

print("\n--- Cleaned Data Preview ---")
print(df.head())

# 3. Use better features & Don't use record_id
# We select features that influence price and explicitly drop irrelevant identifiers/metadata
# (Alternatively, we could explicitly drop columns, but selecting the useful ones achieves the same)
useful_cols = ['mrp', 'product_type', 'material', 'length_cm', 'width_cm', 'height_cm', 'weight_g', 'set_quantity', 'age_group', 'design_complexity', 'painting_type', 'craftsmanship', 'volume_cm3', 'density_g_cm3']

# 2. Extract Top 20 Keywords from product_name using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
df['product_name'] = df['product_name'].fillna('').astype(str)
tfidf = TfidfVectorizer(max_features=20, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['product_name'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"name_{w}" for w in tfidf.get_feature_names_out()], index=df.index)

# Combine the new keyword columns with our dataframe
df = pd.concat([df, tfidf_df], axis=1)

missing_cols = [col+'_missing' for col in numeric_cols]
useful_cols = useful_cols + missing_cols + list(tfidf_df.columns) + ['price']
df = df[useful_cols]

# 4. Convert categorical data properly using pd.get_dummies
df = pd.get_dummies(
    df,
    columns=['product_type', 'material', 'design_complexity', 'painting_type', 'craftsmanship'],
    drop_first=True
)

X = df.drop(columns=['price'])
y = df['price']

# 6. Train/Test Split (as a fallback, but CV is better)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data cleaned and split successfully!")
print(f"Total Rows Saved: {len(df)}")
print(f"Training Data Size: {len(X_train)}")
print(f"Testing Data Size: {len(X_test)}")

from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

base_model = XGBRegressor(
    random_state=42,
    n_jobs=-1
)

model = TransformedTargetRegressor(
    regressor=base_model,
    func=np.log1p,
    inverse_func=np.expm1
)

# Set up Hyperparameter Tuning
param_distributions = {
    'regressor__n_estimators': [100, 300, 500, 1000],
    'regressor__max_depth': [3, 5, 7, 9],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0]
}

print("\nStarting Hyperparameter Tuning...")
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# 7. Evaluate Best Model
predictions = best_model.predict(X_test)
print(f"\n--- Best Model Evaluation ---")
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
print("Predicted prices:", np.round(predictions, 2))
print("Actual prices:", np.round(y_test.values, 2))
