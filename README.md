# Electricity Consumption Forecasting Web App

A time-aware machine learning Streamlit application that predicts next-hour electricity consumption and provides appliance-level usage and bill estimation.

## Live Demo

- App URL: https://electric-power.onrender.com

## Video Demo

- Walkthrough URL: https://drive.google.com/drive/folders/1dcA5JZDJYqgL6vshr3MwkdeJmzaKaUPU?usp=drive_link

## Dataset

- Name: Individual Household Electric Power Consumption
- Kaggle: https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set
- Expected file name: household_power_consumption.txt

The training script checks these paths in order:
1. data/household_power_consumption.txt
2. household_power_consumption.txt

## Project Structure

```text
electricity-forecast-app/
|
|- app.py
|- train.py
|- utils.py
|- model.pkl                # generated after training
|- scaler.pkl               # generated after training
|- hourly_usage.csv         # generated after training
|- model_comparison.csv     # generated after training
|- requirements.txt
|- README.md
|- data/
   |- household_power_consumption.txt
```

## End-to-End Pipeline

1. Load semicolon-separated txt dataset
2. Replace missing markers and impute numeric missing values
3. Merge Date and Time into Datetime and set Datetime as index
4. Convert numeric columns to float
5. Resample minute-level records to hourly mean
6. Create time and lag features:
   - hour
   - day
   - lag_1
   - lag_24
   - rolling_mean_24
7. Drop null rows created by lag and rolling windows
8. Set target as Global_active_power
9. Scale features using StandardScaler
10. Perform forward feature selection (top 5)
11. Train and validate with TimeSeriesSplit (5 folds)
12. Evaluate Ridge, Lasso, PCR, and PLS using MSE
13. Select the lowest-MSE model and save artifacts

## Models Applied

- Ridge Regression
- Lasso Regression
- Principal Component Regression
- Partial Least Squares Regression

## Outputs Generated After Training

- model.pkl
- scaler.pkl
- hourly_usage.csv
- model_comparison.csv

## MSE Comparison

MSE values are stored in model_comparison.csv and displayed in the Model Details tab in the app.

## Streamlit Features

- Wide responsive layout with sections:
  - Prediction Studio
  - Usage and Bill
  - Visual Analytics
  - Model Details
- Auto-fill current hour and day
- Model selection dropdown
- Manual feature input and recent usage input
- Appliance-driven estimation (AC, refrigerator, TV, fan, and custom entries)
- Quick add appliance workflow and reset defaults
- Next-hour prediction
- Next 24-hour forecast
- Usage distribution and hourly behavior charts
- Bill summary with tariff, fixed charge, and tax
- Optional custom dataset upload for visualization

## Run Locally

1. Activate virtual environment (example):

```bash
house\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train models:

```bash
python train.py
```

4. Launch Streamlit app:

```bash
streamlit run app.py
```

## Deployment

Platform options:
- Streamlit Community Cloud
- Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
streamlit run app.py --server.port $PORT
```

## Notes

- Keep the dataset file available in the project root or inside the data folder.
- Retrain whenever you change feature engineering or model settings.
- Replace demo links with your real deployed app URL and final video URL.
