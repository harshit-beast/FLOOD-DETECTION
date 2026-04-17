"""Streamlit dashboard for the Flood Detection System."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from data_loader import load_dataset
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from model import predict_flood_risk, train_random_forest, train_xgboost
from preprocess import split_features_target, split_train_test
from weather_api import fetch_weather_for_location


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


def load_default_dataset() -> pd.DataFrame:
    """Load the bundled dataset used for quick start and real-time demo mode."""
    default_path = Path("data/flood_data.csv")
    if not default_path.exists():
        # Deploy targets may not include local datasets from .gitignore.
        from main import create_sample_dataset

        create_sample_dataset(str(default_path), n_samples=800)
    return load_dataset(str(default_path))


def build_feature_baseline(X: pd.DataFrame) -> Dict[str, Any]:
    """Create a baseline value for each feature when live weather cannot map it directly."""
    baseline: Dict[str, Any] = {}
    for col in X.columns:
        series = X[col].dropna()
        if series.empty:
            baseline[col] = 0.0
            continue

        if pd.api.types.is_numeric_dtype(X[col]):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            baseline[col] = float(numeric.median()) if not numeric.empty else 0.0
        else:
            mode = series.mode()
            baseline[col] = str(mode.iloc[0]) if not mode.empty else str(series.iloc[0])

    return baseline


def build_numeric_feature_stats(X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute min/max bounds for numeric feature clipping."""
    stats: Dict[str, Dict[str, float]] = {}
    for col in X.select_dtypes(include=["number"]).columns:
        numeric = pd.to_numeric(X[col], errors="coerce").dropna()
        if numeric.empty:
            continue
        stats[col] = {"min": float(numeric.min()), "max": float(numeric.max())}
    return stats


def normalize_feature_name(name: str) -> str:
    """Normalize feature names for loose weather-to-feature matching."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def derive_weather_value_for_feature(feature_name: str, weather: Dict[str, float]) -> float | None:
    """Derive a best-effort feature value from live weather readings."""
    key = normalize_feature_name(feature_name)

    rainfall = float(weather.get("rainfall_24h_mm", 0.0))
    precipitation = float(weather.get("precipitation_mm", 0.0))
    humidity = float(weather.get("humidity_pct", 0.0))
    temperature = float(weather.get("temperature_c", 0.0))
    soil_moisture = float(weather.get("soil_moisture_pct", 0.0))
    wind_speed = float(weather.get("wind_speed_kmh", 0.0))
    pressure = float(weather.get("surface_pressure_hpa", 0.0))

    if any(token in key for token in ["rain", "precip"]):
        return max(rainfall, precipitation)
    if "monsoon" in key:
        return max(0.0, min(12.0, rainfall / 2.5))
    if "humid" in key:
        return humidity
    if "temp" in key:
        return temperature
    if "soilmoist" in key:
        return soil_moisture
    if "riverlevel" in key:
        return 2.5 + 0.05 * rainfall + 0.05 * soil_moisture
    if "wind" in key:
        return wind_speed
    if "pressure" in key:
        return pressure
    if "landslide" in key:
        return max(0.0, min(12.0, 0.2 * rainfall + 0.08 * soil_moisture))
    if "coastalvulnerability" in key:
        return max(0.0, min(12.0, 0.12 * rainfall + 0.25 * wind_speed))
    if "climatechange" in key:
        return max(0.0, min(12.0, (temperature - 10.0) * 0.35))
    return None


def build_realtime_feature_row(
    feature_columns: list[str],
    feature_baseline: Dict[str, Any],
    feature_stats: Dict[str, Dict[str, float]],
    weather: Dict[str, float],
) -> tuple[Dict[str, Any], Dict[str, float]]:
    """Create a model-ready input row from weather signals plus learned defaults."""
    row: Dict[str, Any] = {}
    weather_mapped: Dict[str, float] = {}

    for col in feature_columns:
        fallback = feature_baseline.get(col, 0.0)
        derived = derive_weather_value_for_feature(col, weather)
        if derived is None or isinstance(fallback, str):
            row[col] = fallback
            continue

        if col in feature_stats:
            bounds = feature_stats[col]
            derived = max(bounds["min"], min(bounds["max"], derived))

        row[col] = float(derived)
        weather_mapped[col] = float(derived)

    return row, weather_mapped


def render_header() -> None:
    """Render dashboard title and intro."""
    st.title("Flood Detection Dashboard")
    st.caption("Train flood models, run manual predictions, and fetch real-time weather-based risk.")


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
    st.session_state["feature_baseline"] = build_feature_baseline(X)
    st.session_state["feature_stats"] = build_numeric_feature_stats(X)
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


def render_realtime_tab() -> None:
    """Fetch live weather by location and run an immediate flood-risk prediction."""
    st.subheader("Real-Time Weather Prediction")
    st.caption("Uses Open-Meteo API and maps weather signals into the trained model features.")

    required_keys = {"model", "feature_columns", "feature_baseline", "feature_stats"}
    if not required_keys.issubset(st.session_state.keys()):
        st.info("Train models first in the Training tab.")
        return

    default_location = st.session_state.get("realtime_location", "New York")
    left, right = st.columns([3, 1])
    with left:
        location_query = st.text_input("City / Area", value=default_location, key="realtime_location")
    with right:
        fetch_now = st.button("Fetch & Predict", type="primary", use_container_width=True)

    if fetch_now:
        if not location_query.strip():
            st.error("Enter a location to fetch weather.")
            return

        try:
            with st.spinner("Fetching live weather and running prediction..."):
                weather = fetch_weather_for_location(location_query.strip())
                feature_row, mapped = build_realtime_feature_row(
                    feature_columns=st.session_state["feature_columns"],
                    feature_baseline=st.session_state["feature_baseline"],
                    feature_stats=st.session_state["feature_stats"],
                    weather=weather,
                )
                sample = pd.DataFrame([feature_row])
                result = predict_flood_risk(st.session_state["model"], sample)
                pred = int(result["predictions"][0])
                prob = float(result["probabilities"][0])

            st.session_state["last_realtime_prediction"] = {
                "prediction_class": pred,
                "prediction_probability": prob,
                "weather": weather,
                "mapped_features": mapped,
            }
            st.session_state["last_prediction"] = {"class": pred, "probability": prob}
        except Exception as exc:
            st.error(f"Live weather prediction failed: {exc}")
            return

    if "last_realtime_prediction" not in st.session_state:
        st.info("Click `Fetch & Predict` to run a live weather-based prediction.")
        return

    payload = st.session_state["last_realtime_prediction"]
    weather = payload["weather"]
    mapped_features = payload["mapped_features"]
    pred = payload["prediction_class"]
    prob = payload["prediction_probability"]

    st.write(f"Location: **{weather['location']}** ({weather['latitude']:.3f}, {weather['longitude']:.3f})")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Temperature (C)", f"{weather['temperature_c']:.1f}")
    w2.metric("Humidity (%)", f"{weather['humidity_pct']:.1f}")
    w3.metric("Rain (24h mm)", f"{weather['rainfall_24h_mm']:.1f}")
    w4.metric("Wind (km/h)", f"{weather['wind_speed_kmh']:.1f}")

    p1, p2 = st.columns(2)
    p1.metric("Predicted Class", f"{pred}")
    p2.metric("Flood Probability", f"{prob:.4f}")

    if pred == 1:
        st.error("High flood risk predicted for current weather conditions.")
    else:
        st.success("Low flood risk predicted for current weather conditions.")

    total_features = len(st.session_state["feature_columns"])
    mapped_count = len(mapped_features)
    st.caption(
        f"Weather mapped into {mapped_count}/{total_features} model features. "
        "Remaining features used training-data baseline values."
    )

    with st.expander("Mapped Feature Values"):
        st.dataframe(
            pd.DataFrame(
                [{"feature": k, "value": v} for k, v in sorted(mapped_features.items())]
            ),
            use_container_width=True,
        )


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

    if any(k in text for k in ["real-time", "realtime", "weather api", "live weather"]):
        if "last_realtime_prediction" not in st.session_state:
            return "Use the Realtime tab and click `Fetch & Predict` to run live weather inference."
        live = st.session_state["last_realtime_prediction"]
        weather = live["weather"]
        return (
            f"Latest live run for {weather['location']}: class={live['prediction_class']}, "
            f"probability={live['prediction_probability']:.4f}. "
            "Refresh in Realtime tab to pull new weather."
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
        data_source = st.radio("Dataset Source", ["Built-in dataset", "Upload CSV/Excel"], index=0)
        uploaded = None
        if data_source == "Upload CSV/Excel":
            uploaded = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx", "xls"])
        threshold = st.slider("FloodProbability Threshold", 0.0, 1.0, 0.5, 0.05)
        st.caption("If your dataset has `FloodProbability`, this threshold creates binary `flood`.")

    try:
        if data_source == "Upload CSV/Excel":
            if uploaded is None:
                st.info("Upload a dataset from the sidebar to begin.")
                return
            df = load_uploaded_file(uploaded)
        else:
            df = load_default_dataset()

        df = normalize_columns(df)
        df = ensure_binary_target(df, threshold=threshold)
    except Exception as exc:
        st.error(f"Dataset load error: {exc}")
        return

    tab_data, tab_training, tab_prediction, tab_realtime, tab_chatbot = st.tabs(
        ["Data", "Training", "Prediction", "Realtime", "Chatbot"]
    )
    with tab_data:
        render_data_tab(df)
    with tab_training:
        render_training_tab(df)
    with tab_prediction:
        render_prediction_tab(df)
    with tab_realtime:
        render_realtime_tab()
    with tab_chatbot:
        render_chatbot_tab(df)


if __name__ == "__main__":
    main()
