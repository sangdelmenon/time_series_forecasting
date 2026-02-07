
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataLoader:
    """
    Handles data generation, loading, and preprocessing for time series forecasting.
    """
    def __init__(self, n_products=50, n_days=1095):
        self.n_products = n_products
        self.n_days = n_days
        self.data = None

    def generate_sales_data(self, to_csv_path=None):
        """
        Generates synthetic daily sales data for multiple products.
        """
        dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=self.n_days, freq='D'))
        product_ids = [f'Product_{i}' for i in range(1, self.n_products + 1)]
        
        df = pd.DataFrame(columns=['date', 'product_id', 'sales'])
        
        for product in product_ids:
            # Base sales
            base_sales = np.random.randint(50, 200)
            
            # Trend
            trend_factor = np.linspace(1, np.random.uniform(1.2, 1.8), self.n_days)
            
            # Seasonality
            day_of_week_effect = dates.dayofweek.isin([5, 6]) * np.random.uniform(0.2, 0.5) # Weekend spike
            month_effect = np.sin(dates.month * 2 * np.pi / 12) * np.random.uniform(0.1, 0.3)
            
            # Promotions and holidays
            promo_days = np.random.choice(self.n_days, 30, replace=False)
            promo_effect = np.zeros(self.n_days)
            promo_effect[promo_days] = np.random.uniform(0.5, 1.5)

            # Noise
            noise = np.random.normal(0, 0.1, self.n_days)
            
            sales = base_sales * trend_factor * (1 + day_of_week_effect + month_effect + promo_effect + noise)
            
            temp_df = pd.DataFrame({'date': dates, 'product_id': product, 'sales': sales})
            df = pd.concat([df, temp_df])

        df['sales'] = df['sales'].astype(int)
        
        if to_csv_path:
            df.to_csv(to_csv_path, index=False)
            print(f"Data generated and saved to {to_csv_path}")

        self.data = df
        return df

    def load_data(self, file_path):
        """
        Loads data from a CSV file.
        """
        self.data = pd.read_csv(file_path, parse_dates=['date'])
        return self.data

    def handle_missing_values(self, df, method='ffill'):
        """
        Handles missing values in the dataframe.
        """
        if method == 'ffill':
            return df.ffill()
        elif method == 'interpolate':
            return df.interpolate(method='time')
        return df

    def remove_outliers(self, df, column='sales'):
        """
        Removes outliers using the IQR method.
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def create_features(self, df, lags=[1, 7, 30], rolling_windows=[7, 30]):
        """
        Creates lag, rolling statistics, and time-based features.
        """
        df_featured = df.copy()
        
        # Lag features
        for lag in lags:
            df_featured[f'sales_lag_{lag}'] = df_featured.groupby('product_id')['sales'].shift(lag)
            
        # Rolling statistics
        for window in rolling_windows:
            df_featured[f'sales_rolling_mean_{window}'] = df_featured.groupby('product_id')['sales'].shift(1).rolling(window).mean()
            df_featured[f'sales_rolling_std_{window}'] = df_featured.groupby('product_id')['sales'].shift(1).rolling(window).std()

        # Time-based features
        df_featured['day_of_week'] = df_featured['date'].dt.dayofweek
        df_featured['month'] = df_featured['date'].dt.month
        df_featured['quarter'] = df_featured['date'].dt.quarter
        df_featured['year'] = df_featured['date'].dt.year
        
        return df_featured.dropna()

    def normalize_data(self, df, columns_to_scale):
        """
        Normalizes specified columns using MinMaxScaler.
        """
        scalers = {}
        df_scaled = df.copy()
        
        for col in columns_to_scale:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
            scalers[col] = scaler
            
        return df_scaled, scalers

if __name__ == '__main__':
    # Example usage:
    data_loader = TimeSeriesDataLoader()
    
    # Generate and save data
    sales_df = data_loader.generate_sales_data(to_csv_path='time_series_forecasting/data/synthetic_sales.csv')
    
    # Load data
    loaded_df = data_loader.load_data('time_series_forecasting/data/synthetic_sales.csv')
    
    # Process for a single product
    product_df = loaded_df[loaded_df['product_id'] == 'Product_1'].copy()
    
    # Handle missing values (if any)
    product_df = data_loader.handle_missing_values(product_df)
    
    # Remove outliers
    product_df = data_loader.remove_outliers(product_df)
    
    # Create features
    product_df_featured = data_loader.create_features(product_df)
    
    # Normalize data
    columns_to_scale = ['sales'] + [col for col in product_df_featured.columns if 'sales_' in col]
    product_df_scaled, scalers = data_loader.normalize_data(product_df_featured, columns_to_scale)
    
    print("Data Loader Example:")
    print(product_df_scaled.head())
    print("\nScalers used for normalization:")
    print(scalers)
