"""Streamlit dashboard for the Flood Detection System."""

from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

from data_loader import load_dataset
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from model import predict_flood_risk, train_random_forest, train_xgboost
from preprocess import split_features_target, split_train_test
from weather_api import (
    WeatherAPIRateLimitError,
    fetch_daily_weather_history,
    fetch_live_weather,
    fetch_weather_for_location,
)


REASON_FACTOR_MAP = {
    "Heavy Rain": 1.0,
    "River Overflow": 1.2,
    "Urban Drainage Issue": 1.1,
    "Coastal Storm Surge": 1.15,
    "Landslide-prone Area": 1.05,
}

QUICK_AREA_OPTIONS = [
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Kolkata",
    "Chennai",
    "Hyderabad",
    "New York",
    "London",
    "Tokyo",
    "Custom",
]

AREA_COORDINATES = {
    "Mumbai": {"latitude": 19.076, "longitude": 72.878, "location": "Mumbai, Maharashtra, India"},
    "Delhi": {"latitude": 28.614, "longitude": 77.209, "location": "Delhi, India"},
    "Bengaluru": {"latitude": 12.972, "longitude": 77.594, "location": "Bengaluru, Karnataka, India"},
    "Kolkata": {"latitude": 22.572, "longitude": 88.364, "location": "Kolkata, West Bengal, India"},
    "Chennai": {"latitude": 13.083, "longitude": 80.270, "location": "Chennai, Tamil Nadu, India"},
    "Hyderabad": {"latitude": 17.385, "longitude": 78.486, "location": "Hyderabad, Telangana, India"},
    "New York": {"latitude": 40.713, "longitude": -74.006, "location": "New York, New York, USA"},
    "London": {"latitude": 51.507, "longitude": -0.128, "location": "London, England, UK"},
    "Tokyo": {"latitude": 35.676, "longitude": 139.650, "location": "Tokyo, Japan"},
}


@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch_weather_for_location(location_name: str) -> Dict[str, Any]:
    """Cache recent live weather fetches to reduce API calls."""
    return fetch_weather_for_location(location_name)


@st.cache_data(ttl=600, show_spinner=False)
def cached_fetch_live_weather(latitude: float, longitude: float) -> Dict[str, float]:
    """Cache live weather by coordinates for quick-area presets."""
    return fetch_live_weather(latitude=latitude, longitude=longitude)


@st.cache_data(ttl=3600 * 6, show_spinner=False)
def cached_fetch_daily_weather_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Cache daily historical weather to minimize archive API pressure."""
    return fetch_daily_weather_history(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
    )


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


def build_api_training_dataset(hourly_history: Dict[str, list[Any]], reason: str) -> pd.DataFrame:
    """Build daily train rows from historical weather API payload and generate labels."""
    df = pd.DataFrame(hourly_history)
    if df.empty:
        raise ValueError("Historical weather response is empty.")

    if "precipitation_sum" in df.columns:
        # Daily schema from archive `daily=...` query.
        daily = pd.DataFrame(
            {
                "date": pd.to_datetime(df.get("time"), errors="coerce").dt.date,
                "rainfall_24h_mm": pd.to_numeric(df.get("precipitation_sum"), errors="coerce").fillna(0.0),
                "temperature_c": pd.to_numeric(df.get("temperature_2m_mean"), errors="coerce").fillna(0.0),
                "humidity_pct": pd.to_numeric(df.get("relative_humidity_2m_mean"), errors="coerce").fillna(0.0),
                "wind_speed_kmh": pd.to_numeric(df.get("wind_speed_10m_max"), errors="coerce").fillna(0.0),
                "surface_pressure_hpa": pd.to_numeric(df.get("surface_pressure_mean"), errors="coerce").fillna(0.0),
            }
        ).dropna(subset=["date"])
    else:
        # Backward-compatible hourly schema.
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"])
        if df.empty:
            raise ValueError("Historical weather response is empty after parsing timestamps.")

        df["date"] = df["time"].dt.date
        numeric_cols = [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

        daily = (
            df.groupby("date", as_index=False)
            .agg(
                rainfall_24h_mm=("precipitation", "sum"),
                temperature_c=("temperature_2m", "mean"),
                humidity_pct=("relative_humidity_2m", "mean"),
                wind_speed_kmh=("wind_speed_10m", "max"),
                surface_pressure_hpa=("surface_pressure", "mean"),
            )
            .sort_values("date")
        )

    reason_factor = float(REASON_FACTOR_MAP.get(reason, 1.0))
    daily["soil_moisture_proxy"] = (0.7 * daily["rainfall_24h_mm"] + 0.3 * (daily["humidity_pct"] / 2.0)).clip(
        lower=0.0
    )
    daily["runoff_index"] = (
        0.65 * daily["rainfall_24h_mm"] + 0.2 * daily["humidity_pct"] + 0.15 * daily["wind_speed_kmh"]
    ).clip(lower=0.0)
    daily["reason_factor"] = reason_factor

    daily["flood_score"] = (
        0.11 * daily["rainfall_24h_mm"]
        + 0.03 * daily["humidity_pct"]
        + 0.06 * daily["soil_moisture_proxy"]
        + 0.02 * daily["runoff_index"]
        + 0.01 * daily["wind_speed_kmh"]
        - 0.02 * daily["temperature_c"]
        + 3.0 * daily["reason_factor"]
    )
    threshold = float(daily["flood_score"].quantile(0.70))
    daily["flood"] = (daily["flood_score"] >= threshold).astype(int)
    if daily["flood"].nunique() < 2:
        fallback_threshold = float(daily["flood_score"].median())
        daily["flood"] = (daily["flood_score"] >= fallback_threshold).astype(int)
        if daily["flood"].nunique() < 2:
            raise ValueError("Unable to create a trainable target from fetched weather history.")

    return daily[
        [
            "rainfall_24h_mm",
            "temperature_c",
            "humidity_pct",
            "wind_speed_kmh",
            "surface_pressure_hpa",
            "soil_moisture_proxy",
            "runoff_index",
            "reason_factor",
            "flood",
        ]
    ]


def build_live_feature_row(weather: Dict[str, float], reason: str) -> pd.DataFrame:
    """Build a single-row feature frame for current prediction from live weather values."""
    rainfall = float(weather.get("rainfall_24h_mm", 0.0))
    humidity = float(weather.get("humidity_pct", 0.0))
    temperature = float(weather.get("temperature_c", 0.0))
    wind_speed = float(weather.get("wind_speed_kmh", 0.0))
    pressure = float(weather.get("surface_pressure_hpa", 0.0))
    reason_factor = float(REASON_FACTOR_MAP.get(reason, 1.0))

    soil_moisture_proxy = max(0.0, 0.7 * rainfall + 0.3 * (humidity / 2.0))
    runoff_index = max(0.0, 0.65 * rainfall + 0.2 * humidity + 0.15 * wind_speed)

    return pd.DataFrame(
        [
            {
                "rainfall_24h_mm": rainfall,
                "temperature_c": temperature,
                "humidity_pct": humidity,
                "wind_speed_kmh": wind_speed,
                "surface_pressure_hpa": pressure,
                "soil_moisture_proxy": soil_moisture_proxy,
                "runoff_index": runoff_index,
                "reason_factor": reason_factor,
            }
        ]
    )


def build_offline_weather_snapshot(location: str) -> Dict[str, Any]:
    """Create deterministic fallback weather when live API calls are unavailable."""
    preset = AREA_COORDINATES.get(location)
    if preset:
        latitude = float(preset["latitude"])
        longitude = float(preset["longitude"])
        location_label = str(preset["location"])
    else:
        latitude = 0.0
        longitude = 0.0
        location_label = location

    seed = sum(ord(ch) for ch in location.lower()) % 10_000
    rng = np.random.default_rng(seed)
    rainfall = float(rng.uniform(2.0, 32.0))
    humidity = float(rng.uniform(55.0, 92.0))
    temperature = float(rng.uniform(20.0, 34.0))
    wind_speed = float(rng.uniform(6.0, 28.0))
    pressure = float(rng.uniform(995.0, 1018.0))
    soil = float(max(0.0, min(100.0, 0.9 * rainfall + 0.35 * humidity)))

    return {
        "location": f"{location_label} (offline estimate)",
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "precipitation_mm": rainfall / 24.0,
        "rainfall_24h_mm": rainfall,
        "soil_moisture_pct": soil,
        "wind_speed_kmh": wind_speed,
        "surface_pressure_hpa": pressure,
    }


def build_synthetic_daily_history(weather: Dict[str, Any], days: int = 120) -> Dict[str, list[Any]]:
    """Generate lightweight synthetic daily history for resilient training fallback."""
    base_temp = float(weather.get("temperature_c", 28.0))
    base_humidity = float(weather.get("humidity_pct", 75.0))
    base_rain = float(weather.get("rainfall_24h_mm", 12.0))
    base_wind = float(weather.get("wind_speed_kmh", 12.0))
    base_pressure = float(weather.get("surface_pressure_hpa", 1008.0))

    seed = int(abs(base_temp * 100 + base_humidity * 10 + base_rain * 7 + base_wind * 5))
    rng = np.random.default_rng(seed)
    end_dt = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=max(30, days) - 1)
    dates = pd.date_range(start_dt, end_dt, freq="D")
    phase = np.linspace(0, 4 * np.pi, len(dates))

    rain_series = np.clip(base_rain + 8 * np.sin(phase) + rng.normal(0, 3, len(dates)), 0, None)
    humidity_series = np.clip(base_humidity + 10 * np.sin(phase + 0.6) + rng.normal(0, 4, len(dates)), 35, 100)
    temp_series = np.clip(base_temp + 4 * np.cos(phase) + rng.normal(0, 1.5, len(dates)), 5, 45)
    wind_series = np.clip(base_wind + 3 * np.sin(phase + 1.2) + rng.normal(0, 1.2, len(dates)), 0, None)
    pressure_series = np.clip(base_pressure + 4 * np.cos(phase + 0.8) + rng.normal(0, 1.0, len(dates)), 970, 1040)

    return {
        "time": [d.date().isoformat() for d in dates],
        "precipitation_sum": rain_series.round(3).tolist(),
        "relative_humidity_2m_mean": humidity_series.round(3).tolist(),
        "temperature_2m_mean": temp_series.round(3).tolist(),
        "wind_speed_10m_max": wind_series.round(3).tolist(),
        "surface_pressure_mean": pressure_series.round(3).tolist(),
    }


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
    """Train from fetched weather history and predict risk for current conditions."""
    st.subheader("Real-Time Smart Prediction")
    st.caption("Select area and reason. The app fetches weather history, trains automatically, then predicts.")

    chosen_area = st.selectbox("Select Area", QUICK_AREA_OPTIONS, index=0, key="quick_area")
    custom_area = ""
    if chosen_area == "Custom":
        custom_area = st.text_input("Enter Custom Area", value="", key="custom_area")

    selected_reason = st.selectbox("Primary Risk Reason", list(REASON_FACTOR_MAP.keys()), index=0)

    with st.expander("Advanced", expanded=False):
        history_days = st.slider("Training lookback days", min_value=30, max_value=365, value=120, step=15)

    if st.button("Train from API & Predict", type="primary", use_container_width=True):
        location = custom_area.strip() if chosen_area == "Custom" else chosen_area
        if not location:
            st.error("Please enter an area name.")
            return

        try:
            with st.spinner("Fetching weather history, training model, and predicting..."):
                fallback_mode = ""
                end_dt = date.today() - timedelta(days=1)
                start_dt = end_dt - timedelta(days=history_days - 1)

                try:
                    if chosen_area != "Custom" and chosen_area in AREA_COORDINATES:
                        preset = AREA_COORDINATES[chosen_area]
                        live_values = cached_fetch_live_weather(
                            latitude=float(preset["latitude"]),
                            longitude=float(preset["longitude"]),
                        )
                        weather = {
                            "location": str(preset["location"]),
                            "latitude": float(preset["latitude"]),
                            "longitude": float(preset["longitude"]),
                            **live_values,
                        }
                    else:
                        weather = cached_fetch_weather_for_location(location)

                    history = cached_fetch_daily_weather_history(
                        latitude=float(weather["latitude"]),
                        longitude=float(weather["longitude"]),
                        start_date=start_dt.isoformat(),
                        end_date=end_dt.isoformat(),
                    )
                except WeatherAPIRateLimitError:
                    fallback_mode = "rate_limited"
                    weather = build_offline_weather_snapshot(location)
                    history = build_synthetic_daily_history(weather, days=history_days)
                except Exception:
                    fallback_mode = "api_unavailable"
                    weather = build_offline_weather_snapshot(location)
                    history = build_synthetic_daily_history(weather, days=history_days)

                train_df = build_api_training_dataset(history, selected_reason)
                X, y = split_features_target(train_df, target_column="flood")
                X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, random_state=42)
                live_model = train_random_forest(X_train, y_train)
                live_metrics = evaluate_model(live_model, X_test, y_test)

                live_row = build_live_feature_row(weather, selected_reason)
                live_result = predict_flood_risk(live_model, live_row)
                pred = int(live_result["predictions"][0])
                prob = float(live_result["probabilities"][0])

            st.session_state["last_realtime_prediction"] = {
                "prediction_class": pred,
                "prediction_probability": prob,
                "weather": weather,
                "metrics": live_metrics,
                "training_rows": int(len(train_df)),
                "reason": selected_reason,
                "date_window": f"{start_dt.isoformat()} to {end_dt.isoformat()}",
                "fallback_mode": fallback_mode,
            }
            st.session_state["last_prediction"] = {"class": pred, "probability": prob}
        except Exception as exc:
            st.error(f"Live weather training/prediction failed: {exc}")
            return

    if "last_realtime_prediction" not in st.session_state:
        st.info("Pick area and reason, then click `Train from API & Predict`.")
        return

    payload = st.session_state["last_realtime_prediction"]
    weather = payload["weather"]
    pred = payload["prediction_class"]
    prob = payload["prediction_probability"]
    metrics = payload["metrics"]
    fallback_mode = payload.get("fallback_mode", "")

    if fallback_mode == "rate_limited":
        st.warning("Weather API rate-limited the request. Showing resilient offline estimate for now.")
    elif fallback_mode == "api_unavailable":
        st.warning("Weather API is temporarily unavailable. Showing resilient offline estimate for now.")

    st.write(f"Location: **{weather['location']}** ({weather['latitude']:.3f}, {weather['longitude']:.3f})")
    st.caption(f"Reason: {payload['reason']} | Training window: {payload['date_window']}")

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

    st.caption(f"Model retrained on {payload['training_rows']} daily records fetched from weather API.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Validation Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Validation Precision", f"{metrics['precision']:.3f}")
    m3.metric("Validation Recall", f"{metrics['recall']:.3f}")
    m4.metric("Validation ROC-AUC", f"{metrics['roc_auc']:.3f}")


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
