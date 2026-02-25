"""Evaluation and visualization helpers for flood detection models."""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Compute standard classification metrics:
    accuracy, precision, recall, and ROC-AUC.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0,
    }
    return metrics


def plot_confusion_matrix(model, X_test: pd.DataFrame, y_test: pd.Series, output_path: str) -> None:
    """Plot and save confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_roc_curve(model, X_test: pd.DataFrame, y_test: pd.Series, output_path: str) -> None:
    """Plot and save ROC curve."""
    if not hasattr(model, "predict_proba"):
        return

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc_val:.3f})", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_data_trends(df: pd.DataFrame, output_path: str) -> None:
    """
    Plot simple data trends for key numeric weather/river features.
    Expects columns commonly used in flood datasets:
    rainfall, river_level, humidity, temperature.
    """
    candidate_cols = ["rainfall", "river_level", "humidity", "temperature"]
    cols = [c for c in candidate_cols if c in df.columns]
    if not cols:
        return

    plt.figure(figsize=(10, 6))
    for col in cols:
        sns.lineplot(x=np.arange(len(df)), y=df[col], label=col)
    plt.xlabel("Record Index")
    plt.ylabel("Value (scaled by original units)")
    plt.title("Weather and River Data Trends")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
