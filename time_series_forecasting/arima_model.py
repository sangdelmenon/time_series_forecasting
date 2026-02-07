
import pandas as pd
import pmdarima as pm
from pmdarima import model_selection
import matplotlib.pyplot as plt
import joblib

class ARIMAForecaster:
    """
    Handles statistical forecasting using ARIMA models.
    """
    def __init__(self, train_data, product_id, sales_column='sales', date_column='date'):
        self.train_data = train_data[train_data['product_id'] == product_id].set_index(date_column)[sales_column]
        self.product_id = product_id
        self.model = None

    def auto_arima(self, seasonal=True, m=7, stepwise=True):
        """
        Automatically selects the best ARIMA model parameters.
        """
        print(f"Finding best ARIMA for {self.product_id}...")
        self.model = pm.auto_arima(
            self.train_data,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            m=m,              # Weekly seasonality
            seasonal=seasonal,
            d=None,           # Let the model determine 'd'
            D=None,           # Let the model determine 'D'
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=stepwise # Use stepwise algorithm for faster search
        )
        print(self.model.summary())
        return self.model

    def fit(self, order=None, seasonal_order=None):
        """
        Fits the ARIMA model to the data. If order is not specified, runs auto_arima.
        """
        if self.model is None:
            if order:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                self.model = SARIMAX(self.train_data, order=order, seasonal_order=seasonal_order).fit(disp=False)
            else:
                self.auto_arima()
        else:
            self.model.fit(self.train_data)
        
        print(f"ARIMA model fitted for {self.product_id}.")
        return self.model

    def forecast(self, steps):
        """
        Generates forecasts for a given number of steps.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' first.")
        
        forecasts, conf_int = self.model.predict(n_periods=steps, return_conf_int=True)
        return forecasts, conf_int

    def diagnose(self, save_path=None):
        """
        Performs residual analysis on the fitted model.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        
        fig = self.model.plot_diagnostics(figsize=(15, 12))
        
        if save_path:
            fig.savefig(f"{save_path}/arima_diagnostics_{self.product_id}.png")
            plt.close()
        else:
            plt.show()

    def save_model(self, path):
        """
        Saves the trained model to a file.
        """
        joblib.dump(self.model, path)
        print(f"Model for {self.product_id} saved to {path}")

    @staticmethod
    def load_model(path):
        """
        Loads a trained model from a file.
        """
        return joblib.load(path)

if __name__ == '__main__':
    # Example Usage
    try:
        df = pd.read_csv('time_series_forecasting/data/synthetic_sales.csv', parse_dates=['date'])
        
        # Split data for training
        train_df = df[df['date'] < '2025-01-01']
        
        arima_forecaster = ARIMAForecaster(train_df, product_id='Product_1')
        
        # Fit model using auto_arima
        arima_forecaster.fit()
        
        # Generate forecast
        forecasts, conf_int = arima_forecaster.forecast(steps=30)
        print("\nForecasted Sales for next 30 days:")
        print(forecasts)
        
        # Diagnose model
        arima_forecaster.diagnose(save_path='time_series_forecasting/visualizations')
        
        # Save the model
        arima_forecaster.save_model(f'time_series_forecasting/models/arima_{arima_forecaster.product_id}.pkl')

        print("\nSuccessfully ran ARIMAForecaster.")

    except FileNotFoundError:
        print("Please run data_loader.py first to generate the synthetic sales data.")

