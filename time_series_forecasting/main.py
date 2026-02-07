
import pandas as pd
import numpy as np
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from .data_loader import TimeSeriesDataLoader
from .ts_analysis import TimeSeriesAnalyzer
from .arima_model import ARIMAForecaster
from .lstm_model import LSTMForecaster
from .forecast_evaluator import ForecastEvaluator

class ForecastPipeline:
    """
    Orchestrates the entire time series forecasting workflow.
    """
    def __init__(self, data_path, product_id, train_end_date='2025-09-30'):
        self.data_path = data_path
        self.product_id = product_id
        self.train_end_date = train_end_date
        self.output_paths = {
            'models': 'time_series_forecasting/models',
            'visualizations': 'time_series_forecasting/visualizations',
            'data': 'time_series_forecasting/data'
        }
        self._create_dirs()
        
        self.data_loader = TimeSeriesDataLoader()
        self.df = self.data_loader.load_data(self.data_path)
        
        self.product_df = self.df[self.df['product_id'] == self.product_id].set_index('date').sort_index()
        self.train_df = self.product_df[self.product_df.index <= self.train_end_date]
        self.test_df = self.product_df[self.product_df.index > self.train_end_date]
        
        self.arima_model = None
        self.lstm_model = None
        self.metrics = {}

    def _create_dirs(self):
        for path in self.output_paths.values():
            os.makedirs(path, exist_ok=True)

    def run_exploratory_analysis(self):
        print(f"\n--- Running Exploratory Analysis for {self.product_id} ---")
        analyzer = TimeSeriesAnalyzer(self.df, self.product_id)
        analyzer.plot_series(save_path=self.output_paths['visualizations'])
        analyzer.test_stationarity()
        analyzer.decompose_series(save_path=self.output_paths['visualizations'])
        analyzer.plot_acf_pacf(save_path=self.output_paths['visualizations'])
        print("Exploratory analysis complete. Plots saved.")

    def train_arima_model(self):
        print(f"\n--- Training ARIMA Model for {self.product_id} ---")
        full_train_df = self.df[self.df['date'] <= self.train_end_date]
        
        self.arima_model = ARIMAForecaster(full_train_df, self.product_id)
        self.arima_model.fit()
        self.arima_model.diagnose(save_path=self.output_paths['visualizations'])
        self.arima_model.save_model(f"{self.output_paths['models']}/arima_{self.product_id}.pkl")

    def train_lstm_model(self, lookback=90, epochs=50, batch_size=32):
        print(f"\n--- Training LSTM Model for {self.product_id} ---")
        self.lstm_model = LSTMForecaster(lookback=lookback)
        
        # Scale data
        scaled_data = self.lstm_model.scaler.fit_transform(self.train_df[['sales']])
        
        # Create sequences
        X_train, y_train = self.lstm_model.create_sequences(scaled_data)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Build and train
        self.lstm_model.build_model(n_features=1)
        self.lstm_model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
        self.lstm_model.save(f"{self.output_paths['models']}/lstm_{self.product_id}")

    def evaluate_models(self):
        print(f"\n--- Evaluating Models for {self.product_id} ---")
        # ARIMA Evaluation
        arima_forecast, _ = self.arima_model.forecast(steps=len(self.test_df))
        arima_preds = pd.Series(arima_forecast, index=self.test_df.index)
        self.metrics['ARIMA'] = ForecastEvaluator.calculate_metrics(self.test_df['sales'], arima_preds)
        ForecastEvaluator.plot_forecast(self.test_df['sales'], arima_preds, 'ARIMA', self.product_id, self.output_paths['visualizations'])

        # LSTM Evaluation
        scaled_full_data = self.lstm_model.scaler.transform(self.product_df[['sales']])
        X, _ = self.lstm_model.create_sequences(scaled_full_data)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Get the part of X that corresponds to the test set
        test_start_index = len(self.train_df) - self.lstm_model.lookback
        X_test_lstm = X[test_start_index:]
        
        lstm_preds_scaled = self.lstm_model.predict(X_test_lstm)
        lstm_preds = self.lstm_model.scaler.inverse_transform(lstm_preds_scaled).flatten()
        
        self.metrics['LSTM'] = ForecastEvaluator.calculate_metrics(self.test_df['sales'], lstm_preds)
        ForecastEvaluator.plot_forecast(self.test_df['sales'], lstm_preds, 'LSTM', self.product_id, self.output_paths['visualizations'])
        
        # Compare
        ForecastEvaluator.compare_models(self.metrics, self.product_id)
    
    def run_complete_pipeline(self):
        """
        Executes all steps of the forecasting pipeline.
        """
        print(f"======================================================")
        print(f"STARTING FORECAST PIPELINE FOR: {self.product_id}")
        print(f"======================================================")
        
        self.run_exploratory_analysis()
        self.train_arima_model()
        self.train_lstm_model()
        self.evaluate_models()
        
        print(f"\n======================================================")
        print(f"PIPELINE COMPLETED FOR: {self.product_id}")
        print(f"======================================================")

def main():
    DATA_FILE = 'time_series_forecasting/data/synthetic_sales.csv'
    
    # Generate data if it doesn't exist
    if not os.path.exists(DATA_FILE):
        print("Synthetic data not found. Generating new data...")
        TimeSeriesDataLoader().generate_sales_data(to_csv_path=DATA_FILE)
    
    # Run pipeline for a single product
    product_to_forecast = 'Product_5'
    pipeline = ForecastPipeline(data_path=DATA_FILE, product_id=product_to_forecast)
    pipeline.run_complete_pipeline()

if __name__ == '__main__':
    main()
