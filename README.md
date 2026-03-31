# Time Series Forecasting

A modular time series forecasting pipeline implementing and comparing ARIMA and LSTM models for sequential data prediction. Includes data loading, stationarity analysis, model training, evaluation, and visualisation.

## Features

- **ARIMA** — classical statistical forecasting with automatic order selection
- **LSTM** — deep learning sequence model for capturing long-range dependencies
- **Analysis** — stationarity tests, ACF/PACF plots, decomposition
- **Evaluation** — MAE, RMSE, MAPE, and residual diagnostics
- **Visualisations** — forecast plots, confidence intervals, model comparison charts

## Project Structure

```
time_series_forecasting/
├── main.py                # Entry point — run full pipeline
├── data_loader.py         # Load and preprocess time series data
├── ts_analysis.py         # Stationarity tests, ACF/PACF, decomposition
├── arima_model.py         # ARIMA model training and forecasting
├── lstm_model.py          # LSTM model definition and training
├── forecast_evaluator.py  # Metrics: MAE, RMSE, MAPE
├── data/                  # Raw and processed datasets
├── models/                # Saved model checkpoints
└── visualizations/        # Output charts and plots
```

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python time_series_forecasting/main.py
```

## Tech Stack

- **Language**: Python 3
- **Libraries**: pandas, numpy, statsmodels (ARIMA), PyTorch (LSTM), matplotlib, scikit-learn
