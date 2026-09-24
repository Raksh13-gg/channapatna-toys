from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


REQUIRED_FILES = [
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv",
    "feature_encoder.pkl",
]


def main():
    missing_files = [name for name in REQUIRED_FILES if not Path(name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing required split artifact(s): {missing_files}")

    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").squeeze("columns")
    y_test = pd.read_csv("y_test.csv").squeeze("columns")

    with Path("feature_encoder.pkl").open("rb") as file:
        encoder = pickle.load(file)

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("X_train and X_test columns do not match.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")
    if X_train.shape[1] != len(encoder.get_feature_names_out()):
        raise ValueError("Encoded feature count does not match feature_encoder.pkl.")

    allowed_prefixes = ("category_", "material_", "platform_")
    invalid_features = [
        column for column in X_train.columns
        if not column.startswith(allowed_prefixes)
    ]
    if invalid_features:
        raise ValueError(f"Unexpected input feature(s): {invalid_features}")
    if "price" in X_train.columns or "price" in X_test.columns:
        raise ValueError("Target leakage detected: price is present in X.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "model": "sklearn.linear_model.LinearRegression",
        "target": "price",
        "input_features": ["category", "material", "platform"],
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "encoded_features": int(X_train.shape[1]),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "mse": float(mse),
        "rmse": float(mse ** 0.5),
        "r2_score": float(r2_score(y_test, predictions)),
    }

    actual_prices = y_test.astype(float).reset_index(drop=True)
    predicted_prices = pd.Series(predictions, dtype=float)
    absolute_errors = (actual_prices - predicted_prices).abs()
    percentage_errors = absolute_errors.div(actual_prices).mul(100)
    test_predictions = pd.DataFrame(
        {
            "Actual_Price": actual_prices,
            "Predicted_Price": predicted_prices,
            "Absolute_Error": absolute_errors,
            "Percentage_Error": percentage_errors,
        }
    ).round(2)
    test_predictions.to_csv("mlr_test_predictions.csv", index=False)

    with Path("mlr_model.pkl").open("wb") as file:
        pickle.dump(model, file)
    Path("mlr_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("MLR ACTUAL VS PREDICTED PRICE")
    print("=" * 60)
    print()
    print("No. | Actual Price | Predicted Price | Absolute Error | Error %")
    print("-" * 64)
    for number, row in test_predictions.iterrows():
        print(
            f"{number + 1:<3} | ₹{row['Actual_Price']:>10.2f} | "
            f"₹{row['Predicted_Price']:>14.2f} | "
            f"₹{row['Absolute_Error']:>12.2f} | "
            f"{row['Percentage_Error']:>7.2f}%"
        )

    print()
    print("=" * 60)
    print("MULTIPLE LINEAR REGRESSION RESULTS")
    print("=" * 60)
    print(f"Training samples: {metrics['training_samples']}")
    print(f"Testing samples: {metrics['testing_samples']}")
    print(f"Encoded features: {metrics['encoded_features']}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"MSE: {metrics['mse']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R2 Score: {metrics['r2_score']:.6f}")
    print("MODEL SAVED:")
    print("mlr_model.pkl")
    print("mlr_metrics.json")
    print("mlr_test_predictions.csv")
    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)
    print("mlr_model.pkl")
    print("mlr_metrics.json")
    print("mlr_test_predictions.csv")


if __name__ == "__main__":
    main()
