"""Model training and prediction utilities for flood risk classification."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor, build_selector

try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam

    HAS_TENSORFLOW = True
except Exception:
    HAS_TENSORFLOW = False


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a RandomForest model wrapped in an end-to-end preprocessing pipeline.
    """
    preprocessor = build_preprocessor(X_train)
    selector = build_selector(k_features="all")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("classifier", clf),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Optional[Pipeline]:
    """
    Train an XGBoost model in the same preprocessing pipeline.
    Returns None if xgboost is not installed.
    """
    if not HAS_XGBOOST:
        return None

    preprocessor = build_preprocessor(X_train)
    selector = build_selector(k_features="all")
    clf = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        eval_metric="logloss",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", selector),
            ("classifier", clf),
        ]
    )
    model.fit(X_train, y_train)
    return model


def predict_flood_risk(model: Pipeline, new_data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Predict flood risk labels and probabilities for new input data.
    """
    predictions = model.predict(new_data)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_data)[:, 1]
    else:
        # Fallback for models without predict_proba.
        probabilities = np.zeros(shape=(len(new_data),), dtype=float)

    return {"predictions": predictions, "probabilities": probabilities}


def train_lstm_for_timeseries(
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    time_steps: int,
    epochs: int = 10,
    batch_size: int = 32,
) -> Optional["Sequential"]:
    """
    Optional LSTM trainer for time-series input shaped as:
      X_train_seq: (samples, time_steps, features)
      y_train_seq: (samples,)

    Returns None if TensorFlow/Keras is not installed.
    """
    if not HAS_TENSORFLOW:
        return None

    model = Sequential(
        [
            LSTM(64, input_shape=(time_steps, X_train_seq.shape[2]), return_sequences=False),
            Dropout(0.25),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, verbose=1)
    return model
