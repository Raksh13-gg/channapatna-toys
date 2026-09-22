import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FILE_PATH   = "C:/Users/raksh/channapatna-toys/channapatna_complete_common_and_remaining (1).xlsx"
SHEET_NAME  = "All_Products"
TEST_SIZE   = 0.20 
RANDOM_SEED = 42
print("\n[STEP 2] Loading Excel file...")

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)

print(f"  Loaded sheet '{SHEET_NAME}'")
print(f"  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print("\n[STEP 3] Inspecting the raw data...")

print("\n  --- First 5 rows ---")
print(df.head().to_string())

print("\n  --- Column names ---")
print(list(df.columns))

print("\n  --- Data types ---")
print(df.dtypes.to_string())

print("\n  --- Basic statistics ---")
print(df.describe(include="all").to_string())

print("\n  --- Missing values (raw) ---")
print(df.isnull().sum().to_string())

print(f"\n  --- Duplicate rows: {df.duplicated().sum()} ---")

print("\n[STEP 4] Cleaning column names...")

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(f"  Cleaned columns: {list(df.columns)}")

print("\n[STEP 5] Removing duplicate rows...")

before = len(df)
df = df.drop_duplicates()
after  = len(df)

print(f"  Removed {before - after} duplicate(s) | Remaining rows: {after}")

print("\n[STEP 6] Cleaning text columns...")

# These are the text/categorical columns in All_Products
TEXT_COLS = ["section", "product_name", "category", "material", "size", "platform"]

# Only process columns that actually exist in the sheet
TEXT_COLS = [c for c in TEXT_COLS if c in df.columns]

for col in TEXT_COLS:
    df[col] = df[col].astype("string").str.strip()
    # Normalise blank strings to NaN so fillna works properly
    df[col] = df[col].replace("", pd.NA)

print(f"  Stripped whitespace in: {TEXT_COLS}")

print("\n[STEP 7] Filling missing categorical values with 'Unknown'...")

CATEGORICAL_COLS = ["section", "category", "material", "size", "platform"]
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]

for col in CATEGORICAL_COLS:
    missing_before = df[col].isnull().sum()
    df[col] = df[col].fillna("Unknown")
    print(f"  {col:15s} - filled {missing_before} missing value(s)")

print("\n[STEP 7b] Inferring category & material from product name keywords...")

# ---- CATEGORY keyword map (order matters: first match wins) ----
CATEGORY_KEYWORDS = [
    ("Rocking Toy",                   ["rocking horse", "rocking toy", "rocker"]),
    ("Educational Toy",               ["educational", "learning", "alphabet", "abacus", "puzzle",
                                       "montessori", "number", "clock", "counting", "shape"]),
    ("Open-ended / Pretend Play / Educational Toys", ["peg doll", "pretend play", "open-ended",
                                                       "open ended", "figurine play"]),
    ("Baby Rattle",                   ["rattle", "roly poly", "roly-poly", "wobble"]),
    ("Building Blocks",               ["block", "stacking", "stack", "stacker", "interlocking"]),
    ("Animal Toy",                    ["animal", "elephant", "horse", "giraffe", "lion",
                                       "tiger", "bird", "fish", "dinosaur", "bear"]),
    ("Art & Craft",                   ["art", "craft", "drawing", "paint", "colour", "color kit"]),
    ("Spinning Top",                  ["spinning top", "spin top", "top toy", "lattu"]),
    ("Keychain",                      ["keychain", "key chain", "key ring"]),
    ("Home Decor",                    ["home decor", "wall hanging", "decorative", "figurine",
                                       "idol", "showpiece", "decor", "pot", "vase", "jar"]),
    ("Kitchen / Storage",             ["masala", "spice", "container", "storage", "kitchen"]),
    ("Activity Toy",                  ["activity", "sensory", "development", "developmental",
                                       "baby gym", "playgym"]),
    ("Rocking Toy",                   ["rocking"]),
    ("Board Game",                    ["board game", "chess", "ludo", "carrom"]),
    ("Baby Grasping Toy",             ["grasping", "teether", "infant", "newborn", "new born"]),
    ("Doll",                          ["doll", "peg doll", "couple", "family"]),
]

# ---- MATERIAL keyword map ----
MATERIAL_KEYWORDS = [
    ("Natural Wood",   ["natural wood"]),
    ("Mango Wood",     ["mango wood"]),
    ("Solid Wood",     ["solid wood"]),
    ("Teak Wood",      ["teak"]),
    ("Bamboo",         ["bamboo"]),
    ("Wood",           ["wood", "wooden"]),
]

def infer_category(name):
    name_lower = str(name).lower()
    for cat, keywords in CATEGORY_KEYWORDS:
        if any(kw in name_lower for kw in keywords):
            return cat
    return "Unknown"

def infer_material(name):
    name_lower = str(name).lower()
    for mat, keywords in MATERIAL_KEYWORDS:
        if any(kw in name_lower for kw in keywords):
            return mat
    return "Unknown"

cat_filled = 0
mat_filled = 0

for idx, row in df.iterrows():
    if row["category"] == "Unknown" and pd.notna(row["product_name"]):
        inferred = infer_category(row["product_name"])
        if inferred != "Unknown":
            df.at[idx, "category"] = inferred
            cat_filled += 1
    if row["material"] == "Unknown" and pd.notna(row["product_name"]):
        inferred = infer_material(row["product_name"])
        if inferred != "Unknown":
            df.at[idx, "material"] = inferred
            mat_filled += 1

print(f"  category : inferred {cat_filled} values  | still Unknown: {(df['category'] == 'Unknown').sum()}")
print(f"  material : inferred {mat_filled} values  | still Unknown: {(df['material'] == 'Unknown').sum()}")

print("\n[STEP 7c] Cleaning dirty size values and converting to numeric...")

import re

def extract_numeric_size(name):
    """Extract maximum dimension in cm from product name."""
    name = str(name).lower()
    
    # Skip set/pack quantities as they aren't physical sizes
    if "set of" in name or "pack of" in name:
        return np.nan

    # Extract all numbers from the string
    nums = re.findall(r'\d+(?:\.\d+)?', name)
    if not nums:
        return np.nan
        
    nums = [float(n) for n in nums]
    
    # Take the maximum dimension as the proxy for size
    val = max(nums)
    
    # Convert inches to cm
    if "inch" in name or '"' in name:
        val *= 2.54
        
    return val

# Create a numeric size column by inspecting the product name
df["size_numeric"] = df["product_name"].apply(extract_numeric_size)

# Calculate mean of known sizes
mean_size = df["size_numeric"].mean()

# Synthetically fill missing (NaN) sizes with the mean approximation
missing_count = df["size_numeric"].isnull().sum()
df["size_numeric"] = df["size_numeric"].fillna(mean_size)

print(f"  Extracted numeric size for {len(df) - missing_count} products.")
print(f"  Synthetically filled {missing_count} missing sizes with mean value: {mean_size:.2f} cm")

# Drop the old categorical 'size' column since we now have 'size_numeric'
if "size" in df.columns:
    df = df.drop(columns=["size"])
    print("  Dropped old categorical 'size' column")


print("\n[STEP 8] Dropping rows with no product name...")


if "product_name" in df.columns:
    before = len(df)
    df = df.dropna(subset=["product_name"])
    print(f"  Dropped {before - len(df)} row(s) with no product name")
else:
    print("  'product_name' column not found - skipping")

print("\n[STEP 8b] Dropping rows where category or material is still Unknown...")

before = len(df)
mask_unknown = (df["category"] == "Unknown") | (df["material"] == "Unknown")
df = df[~mask_unknown]
after = len(df)
print(f"  Dropped {before - after} row(s) with Unknown category or material")
print(f"  Remaining rows: {after}")

print("\n[STEP 9] Cleaning the 'price' column...")

# Force numeric; non-numeric values become NaN
df["price"] = pd.to_numeric(df["price"], errors="coerce")

invalid_prices = df["price"].isnull().sum()
print(f"  Non-numeric price values found: {invalid_prices}")

# Drop rows where price is NaN
df = df.dropna(subset=["price"])

# Drop zero or negative prices (meaningless for a product)
neg_prices = (df["price"] <= 0).sum()
df = df[df["price"] > 0]
print(f"  Removed {invalid_prices} non-numeric + {neg_prices} invalid (<=0) price rows")
print(f"  Remaining rows: {len(df)}")
print("\n[STEP 10] Computing NumPy-based features...")

# log1p(price) is useful when price distribution is skewed (right-tailed)
# We add it for reference but will NOT include it in X (to avoid data leakage)
df["log_price"] = np.log1p(df["price"])

print("  Sample price vs log_price:")
print(df[["price", "log_price"]].head(5).to_string(index=False))
print("  log_price column added (for reference / optional use as target)")

print("\n[STEP 11] Post-cleaning verification...")

print(f"  Shape       : {df.shape}")
print("\n  Missing values after cleaning:")
print(df.isnull().sum().to_string())
print("\n  Sample cleaned rows:")
print(df.head(3).to_string())

print("\n[STEP 12] Saving cleaned dataset...")

df.to_csv("cleaned_products_v4.csv", index=False, encoding="utf-8-sig")
print("  Saved -> cleaned_products_v4.csv")


print("\n[STEP 13] Defining features X and target y...")

# Target: product price
y = df["price"].copy()

# Features: everything except price and log_price (derived from price)
X = df.drop(columns=["price", "log_price"])

# Drop product_name - too many unique values for simple one-hot encoding.
# (Later you can use TF-IDF on product_name for a richer model.)
if "product_name" in X.columns:
    X = X.drop(columns=["product_name"])
    print("  Dropped 'product_name' (too many unique values - use TF-IDF later)")

print(f"\n  Features (X) columns : {list(X.columns)}")
print(f"  X shape              : {X.shape}")
print(f"  y shape              : {y.shape}")


print("\n[STEP 14] One-hot encoding categorical columns...")

ENCODE_COLS = ["section", "category", "material", "size", "platform"]
ENCODE_COLS = [c for c in ENCODE_COLS if c in X.columns]

print(f"  Columns to encode: {ENCODE_COLS}")

X = pd.get_dummies(X, columns=ENCODE_COLS, drop_first=False)

print(f"  X shape after encoding: {X.shape}")
print(f"  Sample encoded columns: {list(X.columns[:10])} ...")

print("\n[STEP 15] Converting all feature values to float...")

X = X.astype(float)

print("  Data types after conversion:")
print(X.dtypes.value_counts().to_string())

print(f"\n[STEP 16] Splitting data - {int((1-TEST_SIZE)*100)}% train / {int(TEST_SIZE*100)}% test ...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED
)

print(f"\n  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")


print("\n[STEP 17] Saving train/test split files...")

X_train.to_csv("X_train_v4.csv", index=False, encoding="utf-8-sig")
X_test.to_csv("X_test_v4.csv",  index=False, encoding="utf-8-sig")
y_train.to_csv("y_train_v4.csv", index=False, encoding="utf-8-sig")
y_test.to_csv("y_test_v4.csv",  index=False, encoding="utf-8-sig")

print("  Saved -> X_train_v4.csv")
print("  Saved -> X_test_v4.csv")
print("  Saved -> y_train_v4.csv")
print("  Saved -> y_test_v4.csv")

print("\n" + "=" * 55)
print("  PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 55)
print("""
  Output files:
    cleaned_products.csv  <- full cleaned dataset
    X_train.csv           <- training features  (80%)
    X_test.csv            <- testing  features  (20%)
    y_train.csv           <- training prices    (80%)
    y_test.csv            <- testing  prices    (20%)

  Next steps:
    1. Feature Scaling   -> StandardScaler / MinMaxScaler
    2. Train a Model     -> LinearRegression / RandomForest
    3. Evaluate          -> MAE, MSE, R-squared score
    4. (Optional) TF-IDF -> encode product_name as text features
""")
