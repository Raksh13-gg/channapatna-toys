from pathlib import Path
import hashlib
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


MASTER_PATH = Path("master_channapatna_toys_v2.csv")
FEATURES = ["category", "material", "platform"]
TARGET = "price"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main():
    if not MASTER_PATH.exists():
        raise FileNotFoundError(f"Missing required dataset: {MASTER_PATH}")

    master_hash_before = hashlib.sha256(MASTER_PATH.read_bytes()).hexdigest()
    data = pd.read_csv(MASTER_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if data[required_columns].isna().any().any():
        raise ValueError("The selected features and target must not contain missing values.")

    X = data[FEATURES].copy()
    y = data[TARGET].copy().rename(TARGET)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=float,
    )
    X_train_values = encoder.fit_transform(X_train_raw)
    X_test_values = encoder.transform(X_test_raw)
    encoded_columns = encoder.get_feature_names_out(FEATURES)

    X_train = pd.DataFrame(X_train_values, columns=encoded_columns)
    X_test = pd.DataFrame(X_test_values, columns=encoded_columns)

    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
    with Path("feature_encoder.pkl").open("wb") as file:
        pickle.dump(encoder, file)

    master_hash_after = hashlib.sha256(MASTER_PATH.read_bytes()).hexdigest()
    if master_hash_before != master_hash_after:
        raise RuntimeError("The master dataset changed during preprocessing.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise RuntimeError("Feature and target row counts do not match.")
    if set(X_train_raw.index).intersection(X_test_raw.index):
        raise RuntimeError("Train and test rows overlap.")
    if list(X_train.columns) != list(X_test.columns):
        raise RuntimeError("Encoded train and test columns do not match.")

    forbidden_columns = {
        "price",
        "mrp",
        "discount",
        "rating",
        "review_count",
        "size",
        "length_cm",
        "width_cm",
        "height_cm",
        "product_name",
        "product_url",
        "seller",
        "date_collected",
        "verification_note",
    }
    leaked_columns = [
        column
        for column in X_train.columns
        if column.split("_", 1)[0] in forbidden_columns
    ]
    if leaked_columns:
        raise RuntimeError(f"Forbidden columns found in encoded features: {leaked_columns}")

    print(f"Total records: {len(data)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")
    print("Original X columns:")
    for feature in FEATURES:
        print(feature)
    print(f"Encoded feature count: {X_train.shape[1]}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print("Validation: passed")


if __name__ == "__main__":
    main()
