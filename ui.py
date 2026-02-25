"""Streamlit dashboard for the Flood Detection System."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from model import predict_flood_risk, train_random_forest, train_xgboost
from preprocess import split_features_target, split_train_test


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and hidden BOM characters from column names."""
    cleaned = df.copy()
    cleaned.columns = [str(c).strip().replace("\ufeff", "") for c in cleaned.columns]
    return cleaned


def ensure_binary_target(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Ensure binary target `flood` exists, deriving it from probability when needed."""
    if "flood" in df.columns:
        return df

    col_map = {c.lower(): c for c in df.columns}
    for key in ["floodprobability", "flood_probability", "probability", "target"]:
        if key in col_map:
            src = col_map[key]
            out = df.copy()
            out["flood"] = (pd.to_numeric(out[src], errors="coerce") >= threshold).astype(int)
            return out
    raise KeyError("Target column missing. Provide `flood` or `FloodProbability`.")


def load_uploaded_file(uploaded: Any) -> pd.DataFrame:
    """Load uploaded CSV/Excel file."""
    name = uploaded.name.lower()
    content = uploaded.read()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(content), sep=None, engine="python")
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(BytesIO(content))
    raise ValueError("Unsupported file. Upload CSV or Excel.")


def render_header() -> None:
    """Render dashboard title and intro."""
    st.title("Flood Detection Dashboard")
    st.caption("Upload data, train models, compare results, and run flood risk predictions.")


def train_models(df: pd.DataFrame) -> None:
    """Train models and cache results in session state."""
    X, y = split_features_target(df, target_column="flood")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)

    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    best_model = rf_model
    best_name = "RandomForest"
    best_score = rf_metrics["roc_auc"]

    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = None
    if xgb_model is not None:
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        if xgb_metrics["roc_auc"] > best_score:
            best_model = xgb_model
            best_name = "XGBoost"

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    conf_path = plots_dir / "ui_confusion_matrix.png"
    roc_path = plots_dir / "ui_roc_curve.png"
    plot_confusion_matrix(best_model, X_test, y_test, str(conf_path))
    plot_roc_curve(best_model, X_test, y_test, str(roc_path))

    st.session_state["model"] = best_model
    st.session_state["feature_columns"] = X.columns.tolist()
    st.session_state["feature_defaults"] = X.median(numeric_only=True).to_dict()
    st.session_state["feature_choices"] = {
        col: sorted(X[col].dropna().astype(str).unique().tolist())
        for col in X.columns
        if X[col].dtype == "object"
    }
    st.session_state["rf_metrics"] = rf_metrics
    st.session_state["xgb_metrics"] = xgb_metrics
    st.session_state["best_model_name"] = best_name
    st.session_state["confusion_plot"] = str(conf_path)
    st.session_state["roc_plot"] = str(roc_path)


def render_data_tab(df: pd.DataFrame) -> None:
    """Data overview tab."""
    st.subheader("Dataset Overview")
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df.head(20), use_container_width=True)
    with right:
        st.metric("Rows", f"{df.shape[0]}")
        st.metric("Columns", f"{df.shape[1]}")
        st.metric("Flood Rate", f"{df['flood'].mean():.2%}")

    with st.expander("Column Details"):
        info = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
        st.dataframe(info, use_container_width=True)


def render_training_tab(df: pd.DataFrame) -> None:
    """Model training and evaluation tab."""
    st.subheader("Training")
    if st.button("Train / Retrain Models", type="primary", use_container_width=True):
        try:
            with st.spinner("Training models..."):
                train_models(df)
            st.success(f"Training complete. Best model: {st.session_state['best_model_name']}")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    if "rf_metrics" not in st.session_state:
        st.info("Train a model to view metrics and charts.")
        return

    rf_metrics = st.session_state["rf_metrics"]
    xgb_metrics = st.session_state["xgb_metrics"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{rf_metrics['accuracy']:.4f}")
    m2.metric("Precision", f"{rf_metrics['precision']:.4f}")
    m3.metric("Recall", f"{rf_metrics['recall']:.4f}")
    m4.metric("ROC-AUC", f"{rf_metrics['roc_auc']:.4f}")
    st.caption("Metrics shown above are for RandomForest.")

    if xgb_metrics is not None:
        st.write("XGBoost metrics:", {k: round(v, 4) for k, v in xgb_metrics.items()})
    else:
        st.caption("XGBoost not installed. Install `xgboost` to enable comparison.")

    p1, p2 = st.columns(2)
    with p1:
        st.image(st.session_state["confusion_plot"], caption="Confusion Matrix", use_container_width=True)
    with p2:
        st.image(st.session_state["roc_plot"], caption="ROC Curve", use_container_width=True)


def render_prediction_tab(df: pd.DataFrame) -> None:
    """Prediction input and output tab."""
    st.subheader("Predict Flood Risk")
    if "model" not in st.session_state or "feature_columns" not in st.session_state:
        st.info("Train models first in the Training tab.")
        return

    feature_columns = st.session_state["feature_columns"]
    numeric_defaults = st.session_state.get("feature_defaults", {})
    feature_choices = st.session_state.get("feature_choices", {})

    with st.form("prediction_form"):
        input_values = {}
        cols = st.columns(2)
        for idx, col in enumerate(feature_columns):
            current_col = cols[idx % 2]
            with current_col:
                if col in numeric_defaults:
                    input_values[col] = st.number_input(col, value=float(numeric_defaults[col]))
                else:
                    choices = feature_choices.get(col, [])
                    input_values[col] = st.selectbox(col, choices, index=0) if choices else ""
        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        return

    sample = pd.DataFrame([input_values])
    result = predict_flood_risk(st.session_state["model"], sample)
    pred = int(result["predictions"][0])
    prob = float(result["probabilities"][0])
    st.session_state["last_prediction"] = {"class": pred, "probability": prob}

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted Class", f"{pred}")
        st.caption("1 = Flood, 0 = No Flood")
    with c2:
        st.metric("Flood Probability", f"{prob:.4f}")

    if pred == 1:
        st.error("High flood risk predicted for this input.")
    else:
        st.success("Low flood risk predicted for this input.")

    st.caption(f"Model used: {st.session_state['best_model_name']}")


def chatbot_response(user_text: str, df: pd.DataFrame) -> str:
    """Return a contextual chatbot reply using dashboard state."""
    text = user_text.lower().strip()
    flood_rate = float(df["flood"].mean())
    rows, cols = df.shape

    if any(k in text for k in ["hello", "hi", "hey"]):
        return "Hello. Ask me about dataset quality, model performance, or flood risk interpretation."

    if any(k in text for k in ["dataset", "data summary", "rows", "columns"]):
        return (
            f"Dataset has {rows} rows and {cols} columns. "
            f"Observed flood rate is {flood_rate:.2%}. "
            "Use the Data tab to inspect feature types and missing values."
        )

    if any(k in text for k in ["target", "floodprobability", "threshold"]):
        return (
            "This app expects a binary target `flood` (0/1). "
            "If your dataset contains `FloodProbability`, it is converted to `flood` using the selected threshold."
        )

    if any(k in text for k in ["metric", "accuracy", "precision", "recall", "roc", "performance"]):
        if "rf_metrics" not in st.session_state:
            return "Train a model first in the Training tab. I will then explain all metrics."
        m = st.session_state["rf_metrics"]
        return (
            f"RandomForest metrics: Accuracy {m['accuracy']:.4f}, Precision {m['precision']:.4f}, "
            f"Recall {m['recall']:.4f}, ROC-AUC {m['roc_auc']:.4f}. "
            "Higher ROC-AUC means better ranking between flood and non-flood cases."
        )

    if any(k in text for k in ["best model", "which model", "xgboost", "randomforest"]):
        if "best_model_name" not in st.session_state:
            return "No best model yet. Train in the Training tab first."
        return f"Current best model is {st.session_state['best_model_name']} based on validation ROC-AUC."

    if any(k in text for k in ["predict", "prediction", "probability", "risk"]):
        last = st.session_state.get("last_prediction")
        if not last:
            return "Run a prediction in the Prediction tab first, then I can interpret that result."
        risk_label = "High" if last["class"] == 1 else "Low"
        return (
            f"Latest prediction: class={last['class']} and probability={last['probability']:.4f}. "
            f"Interpreted risk level: {risk_label}. "
            "You can test scenarios by changing feature values in the Prediction tab."
        )

    if any(k in text for k in ["improve", "better", "increase accuracy", "reduce error"]):
        return (
            "To improve quality: clean outliers, add more historical data, validate class balance, "
            "try XGBoost, and tune hyperparameters with cross-validation."
        )

    if any(k in text for k in ["what can you do", "help", "commands"]):
        return (
            "I can explain dataset stats, target conversion, model metrics, best model selection, "
            "and latest prediction interpretation."
        )

    return (
        "I did not fully understand that. Ask about dataset summary, metrics, best model, "
        "or prediction interpretation."
    )


def render_chatbot_tab(df: pd.DataFrame) -> None:
    """Simple local chatbot experience for dashboard guidance."""
    st.subheader("Flood Assistant")
    st.caption("Ask questions about your data, model results, and predictions.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": "Ask me: dataset summary, model performance, best model, or latest prediction.",
            }
        ]

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Type your question...")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        reply = chatbot_response(prompt, df)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)


def main() -> None:
    st.set_page_config(page_title="Flood Detection Dashboard", layout="wide")
    render_header()

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
        threshold = st.slider("FloodProbability Threshold", 0.0, 1.0, 0.5, 0.05)
        st.caption("If your dataset has `FloodProbability`, this threshold creates binary `flood`.")

    if uploaded is None:
        st.info("Upload a dataset from the sidebar to begin.")
        return

    try:
        df = load_uploaded_file(uploaded)
        df = normalize_columns(df)
        df = ensure_binary_target(df, threshold=threshold)
    except Exception as exc:
        st.error(f"Dataset load error: {exc}")
        return

    tab_data, tab_training, tab_prediction, tab_chatbot = st.tabs(
        ["Data", "Training", "Prediction", "Chatbot"]
    )
    with tab_data:
        render_data_tab(df)
    with tab_training:
        render_training_tab(df)
    with tab_prediction:
        render_prediction_tab(df)
    with tab_chatbot:
        render_chatbot_tab(df)


if __name__ == "__main__":
    main()
