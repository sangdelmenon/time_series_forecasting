
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ForecastEvaluator:
    """
    Evaluates the performance of forecasting models.
    """
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculates key forecasting metrics.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Avoid division by zero for MAPE
        y_true_safe = np.where(y_true == 0, 1e-8, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

    @staticmethod
    def plot_forecast(y_true, y_pred, model_name, product_id, save_path=None):
        """
        Visualizes the forecast against the actual values.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(y_true.index, y_true, label='Actual Sales', color='blue')
        plt.plot(y_true.index, y_pred, label=f'{model_name} Forecast', color='red', linestyle='--')
        plt.title(f'Forecast vs Actuals for {product_id} using {model_name}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(f"{save_path}/{model_name}_forecast_{product_id}.png")
            plt.close()
        else:
            plt.show()

    @staticmethod
    def compare_models(metrics_dict, product_id):
        """
        Compares multiple models based on their metrics.
        
        Args:
            metrics_dict (dict): e.g., {'ARIMA': {'MAE': 10, ...}, 'LSTM': {'MAE': 12, ...}}
        """
        print(f"\n--- Model Comparison for {product_id} ---")
        metrics_df = pd.DataFrame(metrics_dict).T
        print(metrics_df)
        return metrics_df


if __name__ == '__main__':
    # Example Usage
    from data_loader import TimeSeriesDataLoader
    from arima_model import ARIMAForecaster
    
    try:
        # Load data
        df = pd.read_csv('time_series_forecasting/data/synthetic_sales.csv', parse_dates=['date'])
        product_id = 'Product_1'
        product_df = df[df['product_id'] == product_id].set_index('date')

        # Split data
        train_df = product_df[product_df.index < '2025-06-01']
        test_df = product_df[product_df.index >= '2025-06-01']

        # --- ARIMA Example ---
        arima_forecaster = ARIMAForecaster(df[df['date'] < '2025-06-01'], product_id=product_id)
        arima_forecaster.fit()
        
        # Forecast
        arima_forecast, _ = arima_forecaster.forecast(steps=len(test_df))
        arima_forecast_series = pd.Series(arima_forecast, index=test_df.index)

        # Evaluate
        arima_metrics = ForecastEvaluator.calculate_metrics(test_df['sales'], arima_forecast_series)
        
        print("\n--- ARIMA Evaluation ---")
        print(arima_metrics)

        # Plot
        ForecastEvaluator.plot_forecast(test_df['sales'], arima_forecast_series, 'ARIMA', product_id, save_path='time_series_forecasting/visualizations')

        # --- Dummy LSTM Example (for comparison) ---
        # In a real scenario, you would train an LSTM model here
        dummy_lstm_forecast = test_df['sales'] * 0.9  # Simple dummy forecast
        lstm_metrics = ForecastEvaluator.calculate_metrics(test_df['sales'], dummy_lstm_forecast)
        
        print("\n--- Dummy LSTM Evaluation ---")
        print(lstm_metrics)
        
        # --- Compare Models ---
        all_metrics = {'ARIMA': arima_metrics, 'LSTM (Dummy)': lstm_metrics}
        ForecastEvaluator.compare_models(all_metrics, product_id)
        
        print("\nSuccessfully ran ForecastEvaluator.")

    except FileNotFoundError:
        print("Please run data_loader.py first to generate the synthetic sales data.")
    except Exception as e:
        print(f"An error occurred: {e}")

