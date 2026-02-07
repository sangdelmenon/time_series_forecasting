
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

class TimeSeriesAnalyzer:
    """
    Performs exploratory analysis of time series data.
    """
    def __init__(self, data, product_id, sales_column='sales', date_column='date'):
        self.data = data[data['product_id'] == product_id].set_index(date_column)[sales_column]
        self.product_id = product_id

    def test_stationarity(self, print_results=True):
        """
        Performs the Augmented Dickey-Fuller test for stationarity.
        """
        result = adfuller(self.data.dropna())
        if print_results:
            print(f'--- ADF Test for {self.product_id} ---')
            print(f'ADF Statistic: {result[0]}')
            print(f'p-value: {result[1]}')
            print('Critical Values:')
            for key, value in result[4].items():
                print(f'\t{key}: {value}')
            
            if result[1] <= 0.05:
                print("Result: The series is likely stationary.")
            else:
                print("Result: The series is likely non-stationary.")
        return result

    def decompose_series(self, seasonal=7, plot=True, save_path=None):
        """
        Decomposes the time series using STL.
        """
        stl = STL(self.data, seasonal=seasonal)
        result = stl.fit()

        if plot:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
            axes[0].plot(self.data)
            axes[0].set_title(f'Original Series - {self.product_id}')
            axes[1].plot(result.trend)
            axes[1].set_title('Trend')
            axes[2].plot(result.seasonal)
            axes[2].set_title('Seasonal')
            axes[3].plot(result.resid)
            axes[3].set_title('Residual')
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/stl_decomposition_{self.product_id}.png")
                plt.close()
            else:
                plt.show()

        return result

    def plot_acf_pacf(self, lags=40, save_path=None):
        """
        Plots the Autocorrelation and Partial Autocorrelation functions.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        plot_acf(self.data.dropna(), lags=lags, ax=ax1)
        ax1.set_title(f'ACF - {self.product_id}')
        
        plot_pacf(self.data.dropna(), lags=lags, ax=ax2)
        ax2.set_title(f'PACF - {self.product_id}')
        
        if save_path:
            plt.savefig(f"{save_path}/acf_pacf_{self.product_id}.png")
            plt.close()
        else:
            plt.show()

    def plot_series(self, save_path=None):
        """
        Plots the time series data.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(self.data)
        plt.title(f'Sales for {self.product_id}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.grid(True)
        
        if save_path:
            plt.savefig(f"{save_path}/sales_plot_{self.product_id}.png")
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    # Example Usage
    # Ensure you have 'time_series_forecasting/data/synthetic_sales.csv' generated
    try:
        df = pd.read_csv('time_series_forecasting/data/synthetic_sales.csv', parse_dates=['date'])
        
        analyzer = TimeSeriesAnalyzer(df, product_id='Product_1')
        
        # Plot the series
        analyzer.plot_series(save_path='time_series_forecasting/visualizations')
        
        # Test for stationarity
        analyzer.test_stationarity()
        
        # Decompose the series
        analyzer.decompose_series(save_path='time_series_forecasting/visualizations')
        
        # Plot ACF and PACF
        analyzer.plot_acf_pacf(save_path='time_series_forecasting/visualizations')
        
        print("\nSuccessfully ran TimeSeriesAnalyzer and saved plots to 'visualizations' folder.")

    except FileNotFoundError:
        print("Please run data_loader.py first to generate the synthetic sales data.")

