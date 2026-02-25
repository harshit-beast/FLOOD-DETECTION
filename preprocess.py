"""Preprocessing utilities: cleaning, encoding, normalization, and feature selection."""

from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(
    df: pd.DataFrame,
    target_column: str = "flood",
    feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target labels (y).
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    if feature_columns:
        missing = [col for col in feature_columns if col not in X.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}")
        X = X[feature_columns]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing transformer:
    - Numeric: median imputation + standard scaling
    - Categorical: mode imputation + one-hot encoding
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def build_selector(k_features: str | int = "all") -> SelectKBest:
    """
    Build a feature selector using mutual information.
    Set k_features='all' to keep all transformed features.
    """
    return SelectKBest(score_func=mutual_info_classif, k=k_features)


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split with stratification for classification stability."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
