
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model, load_model

class LSTMForecaster:
    """
    Handles deep learning based forecasting using LSTM networks.
    """
    def __init__(self, lookback=90):
        self.lookback = lookback
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def create_sequences(self, data):
        """
        Creates sequences of data for LSTM training.
        """
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def build_model(self, n_features, units1=64, units2=32, dropout=0.2):
        """
        Builds the LSTM model architecture.
        """
        model = Sequential([
            LSTM(units1, return_sequences=True, input_shape=(self.lookback, n_features)),
            Dropout(dropout),
            LSTM(units2),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model
        print(self.model.summary())
        return self.model

    def train(self, X_train, y_train, epochs=100, batch_size=32, patience=10):
        """
        Trains the LSTM model with early stopping.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call 'build_model' first.")
        
        early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def predict(self, data):
        """
        Generates predictions from the trained model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(data)

    def save(self, filepath):
        """Saves the model and scaler."""
        self.model.save(f"{filepath}_model.h5")
        import joblib
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        print(f"LSTM model and scaler saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads the model and scaler."""
        forecaster = cls()
        forecaster.model = load_model(f"{filepath}_model.h5")
        import joblib
        forecaster.scaler = joblib.load(f"{filepath}_scaler.pkl")
        print(f"LSTM model and scaler loaded from {filepath}")
        return forecaster


if __name__ == '__main__':
    # Example Usage
    from data_loader import TimeSeriesDataLoader

    try:
        # 1. Load and prepare data
        data_loader = TimeSeriesDataLoader()
        df = data_loader.load_data('time_series_forecasting/data/synthetic_sales.csv')
        
        product_df = df[df['product_id'] == 'Product_1'][['date', 'sales']].set_index('date')
        
        # 2. Scale data
        lstm_forecaster = LSTMForecaster(lookback=90)
        scaled_data = lstm_forecaster.scaler.fit_transform(product_df)

        # 3. Create sequences
        X, y = lstm_forecaster.create_sequences(scaled_data)

        # Reshape X for LSTM [samples, time_steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 4. Build and train model
        lstm_forecaster.build_model(n_features=1)
        lstm_forecaster.train(X_train, y_train)

        # 5. Make predictions
        predictions_scaled = lstm_forecaster.predict(X_test)
        predictions = lstm_forecaster.scaler.inverse_transform(predictions_scaled)

        print("\nSample of Predictions (first 5):")
        print(predictions[:5].flatten())

        # 6. Save the model
        lstm_forecaster.save('time_series_forecasting/models/lstm_Product_1')
        
        # 7. Load the model (example)
        loaded_forecaster = LSTMForecaster.load('time_series_forecasting/models/lstm_Product_1')
        loaded_predictions = loaded_forecaster.scaler.inverse_transform(loaded_forecaster.predict(X_test))
        print("\nSample of Predictions from loaded model (first 5):")
        print(loaded_predictions[:5].flatten())


    except FileNotFoundError:
        print("Please run data_loader.py first to generate the synthetic sales data.")

