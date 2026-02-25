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
- Streamlit dashboard for interactive upload, training, and prediction.

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
