# AI/ML Flood Detection System

A modular Python project for flood risk prediction using tabular weather/river data.

## Features

- Load datasets from CSV or Excel.
- Preprocess data (missing values, scaling, encoding, feature selection).
- Train machine learning models:
  - RandomForest (default)
  - XGBoost (optional)
  - LSTM helper included (optional, TensorFlow)
- Evaluate performance with:
  - Accuracy
  - Precision
  - Recall
  - ROC-AUC
  - Confusion Matrix
  - ROC Curve
- Predict flood risk for new data.
- Streamlit dashboard for interactive upload, training, manual prediction, and real-time weather prediction via API.

## Project Structure

```text
New project/
├── data_loader.py
├── preprocess.py
├── model.py
├── evaluate.py
├── main.py
├── ui.py
├── requirements.txt
└── README.md
```

## Installation

1. Open terminal in project root:

```bash
cd "/Users/harshit/Documents/New project"
```

2. Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (Script Mode)

```bash
python3 main.py
```

What happens:
- Loads dataset from `data/flood_data.csv`.
- If dataset is missing, creates a sample dataset automatically.
- Handles Kaggle-style `FloodProbability` by converting to binary `flood` (threshold `0.5`).
- Trains model(s), prints metrics, saves plots to `outputs/plots`.

## Run (Dashboard Mode)

```bash
streamlit run ui.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

Dashboard tabs:
- `Data`: dataset preview and column details
- `Training`: train/retrain and view model metrics/plots
- `Prediction`: enter feature values and get flood risk prediction
- `Realtime`: select area + risk reason, auto-train on fetched weather history, and predict instantly

Realtime notes:
- Uses [Open-Meteo](https://open-meteo.com/) geocoding + forecast API.
- No API key is required.
- Automatically fetches daily weather history for the selected area and retrains a local model before prediction.
- User only needs to select area and reason (plus optional lookback days in Advanced settings).
- App includes retry/backoff + caching to handle temporary API rate limiting.
- If weather API is unavailable/rate-limited, app switches to offline estimated weather + synthetic history so prediction still works.
- If `data/flood_data.csv` is missing on deploy, the app auto-generates a sample dataset.

## Public Deployment

### Option 1: Streamlit Community Cloud (fastest)

1. Push this repo to GitHub.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app** and select:
   - Repository: your `FLOOD-DETECTION` repo
   - Branch: `main`
   - Main file path: `ui.py`
4. Click **Deploy**.

Your app will be publicly available on a `*.streamlit.app` URL.

### Option 2: Render

This repo includes `render.yaml` for one-click setup.

1. Push the latest code to GitHub.
2. In Render, click **New +** -> **Blueprint**.
3. Select this repository.
4. Render will create `flood-detection-dashboard` automatically and deploy it.

Your app will be publicly available on a `*.onrender.com` URL.

## Dataset Format

### Supported target columns

- `flood` (binary: `0` or `1`)  
or
- `FloodProbability` (continuous; app converts to binary using threshold)

### Example columns

`MonsoonIntensity`, `TopographyDrainage`, `RiverManagement`, ..., `FloodProbability`

## Optional Dependencies

- `xgboost` for XGBoost model:

```bash
pip install xgboost
```

- `tensorflow` for LSTM/deep learning experiments:

```bash
pip install tensorflow
```

## Notes

- TensorFlow is optional and not required for current tabular workflow.
- XGBoost is optional; project works fully with RandomForest.
