"""Main script demonstrating a complete AI/ML Flood Detection workflow."""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from data_loader import load_dataset
from evaluate import evaluate_model, plot_confusion_matrix, plot_data_trends, plot_roc_curve
from model import predict_flood_risk, train_random_forest, train_xgboost
from preprocess import split_features_target, split_train_test


def create_sample_dataset(file_path: str, n_samples: int = 800) -> None:
    """
    Create a synthetic flood dataset for demonstration.
    """
    rng = np.random.default_rng(42)
    rainfall = rng.normal(120, 45, n_samples).clip(0, None)
    river_level = rng.normal(7.0, 2.0, n_samples).clip(0, None)
    humidity = rng.normal(70, 12, n_samples).clip(10, 100)
    temperature = rng.normal(28, 6, n_samples)
    soil_moisture = rng.normal(45, 18, n_samples).clip(0, 100)

    # Region acts as a categorical feature.
    region = rng.choice(["north", "south", "east", "west"], size=n_samples)

    # Construct a probabilistic flood target from feature interactions.
    flood_score = (
        0.03 * rainfall
        + 1.15 * river_level
        + 0.015 * humidity
        + 0.02 * soil_moisture
        - 0.03 * temperature
    )
    flood_prob = 1 / (1 + np.exp(-(flood_score - 12)))
    flood = (rng.random(n_samples) < flood_prob).astype(int)

    df = pd.DataFrame(
        {
            "rainfall": rainfall,
            "river_level": river_level,
            "humidity": humidity,
            "temperature": temperature,
            "soil_moisture": soil_moisture,
            "region": region,
            "flood": flood,
        }
    )

    # Add some missing values to show imputation handling.
    for col in ["rainfall", "humidity", "region"]:
        missing_idx = rng.choice(df.index, size=max(1, n_samples // 25), replace=False)
        df.loc[missing_idx, col] = np.nan

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def print_metrics(model_name: str, metrics: Dict[str, float]) -> None:
    """Pretty-print model metrics."""
    print(f"\n{model_name} Performance")
    print("-" * (len(model_name) + 12))
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"ROC AUC  : {metrics['roc_auc']:.4f}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to avoid whitespace/BOM mismatches."""
    clean_cols = []
    for col in df.columns:
        col_clean = str(col).strip().replace("\ufeff", "")
        clean_cols.append(col_clean)
    df.columns = clean_cols
    return df


def _ensure_binary_target(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Ensure a binary target column named `flood` exists.
    Supports Kaggle-style `FloodProbability` target by thresholding.
    """
    if "flood" in df.columns:
        return df

    # Case-insensitive lookup for probable target names.
    col_map = {c.lower(): c for c in df.columns}
    for key in ["floodprobability", "flood_probability", "probability", "target"]:
        if key in col_map:
            source_col = col_map[key]
            df["flood"] = (pd.to_numeric(df[source_col], errors="coerce") >= threshold).astype(int)
            return df

    raise KeyError(
        "No target column found. Expected `flood` or a probability column like `FloodProbability`."
    )


def main() -> None:
    dataset_path = "data/flood_data.csv"
    plots_dir = Path("outputs/plots")

    # Create demo data if a dataset file does not already exist.
    if not Path(dataset_path).exists():
        create_sample_dataset(dataset_path, n_samples=800)
        print(f"Created sample dataset at {dataset_path}")

    # 1) Load data from CSV/Excel.
    df = load_dataset(dataset_path)
    df = _normalize_columns(df)
    df = _ensure_binary_target(df, threshold=0.5)

    # 2) Basic trend visualization from raw data.
    plot_data_trends(df, str(plots_dir / "data_trends.png"))

    # 3) Split dataset into features and target.
    X, y = split_features_target(df, target_column="flood")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)

    # 4) Train RandomForest model.
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print_metrics("RandomForest", rf_metrics)
    plot_confusion_matrix(rf_model, X_test, y_test, str(plots_dir / "rf_confusion_matrix.png"))
    plot_roc_curve(rf_model, X_test, y_test, str(plots_dir / "rf_roc_curve.png"))

    # 5) Optionally train XGBoost if installed.
    xgb_model = train_xgboost(X_train, y_train)
    best_model = rf_model
    best_model_name = "RandomForest"
    best_score = rf_metrics["roc_auc"]

    if xgb_model is not None:
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        print_metrics("XGBoost", xgb_metrics)
        plot_confusion_matrix(xgb_model, X_test, y_test, str(plots_dir / "xgb_confusion_matrix.png"))
        plot_roc_curve(xgb_model, X_test, y_test, str(plots_dir / "xgb_roc_curve.png"))

        if xgb_metrics["roc_auc"] > best_score:
            best_model = xgb_model
            best_model_name = "XGBoost"
            best_score = xgb_metrics["roc_auc"]
    else:
        print("\nXGBoost not installed. Skipping XGBoost training.")

    # 6) Predict flood risk for new input data.
    # Use a real feature row from the test split so schema always matches user datasets.
    sample_input = X_test.head(1).copy()
    prediction_result = predict_flood_risk(best_model, sample_input)
    predicted_class = int(prediction_result["predictions"][0])
    predicted_probability = float(prediction_result["probabilities"][0])

    print("\nFlood Risk Prediction")
    print("---------------------")
    print(f"Best model used       : {best_model_name}")
    print(f"Predicted flood class : {predicted_class} (1=Flood, 0=No Flood)")
    print(f"Predicted probability : {predicted_probability:.4f}")
    print(f"\nSaved plots in: {plots_dir.resolve()}")


if __name__ == "__main__":
    main()
