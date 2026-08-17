import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


class EMDBPNN:
    """
    EMD-BPNN hybrid model for time series forecasting.
    Each EMD component (high_freq, low_freq, trend) gets its own BPNN.
    Final prediction = direct sum of the three component predictions.
    """

    def __init__(self, lookback=10):
        self.lookback   = lookback
        self.models     = {}
        self.scalers    = {
            'high_freq': MinMaxScaler(feature_range=(0, 1)),
            'low_freq':  MinMaxScaler(feature_range=(0, 1)),
            'trend':     MinMaxScaler(feature_range=(0, 1)),
        }
        self.predictions = {}

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------

    def create_component_model(self, input_dim, component_name):
        """Build and register a small BPNN for one EMD component."""
        model = Sequential([
            Dense(32, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(1),
        ])
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=0.001))
        self.models[component_name] = model
        return model

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def create_dataset(self, data, train_size, component_name):
        """Scale data and build sliding-window sequences."""
        scaled = self.scalers[component_name].fit_transform(
            data.reshape(-1, 1)).flatten()

        X, y = [], []
        for i in range(len(scaled) - self.lookback):
            X.append(scaled[i: i + self.lookback])
            y.append(scaled[i + self.lookback])

        X = np.array(X)
        y = np.array(y)

        split = train_size - self.lookback
        return X[:split], y[:split], X[split:], y[split:]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_component(self, data, train_size, component_name,
                        epochs=100, batch_size=32):
        """Train the BPNN for a single EMD component."""
        X_tr, y_tr, X_te, y_te = self.create_dataset(
            data, train_size, component_name)

        if component_name not in self.models:
            self.create_component_model(self.lookback, component_name)

        early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                   restore_best_weights=True)

        history = self.models[component_name].fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_te, y_te),
            callbacks=[early_stop],
            verbose=1,
        )
        return history

    # kept for backward-compat with app.py (called with epochs=1 as no-op)
    def train_combined_model(self, original_data, train_size,
                             epochs=1, batch_size=32):
        """
        Legacy stub – the new architecture uses direct summation,
        so this method does nothing meaningful.  Returns a dummy object
        whose .history attribute mimics a Keras History so the template
        doesn't break.
        """
        class _FakeHistory:
            history = {'loss': [0.0], 'val_loss': [0.0]}

        return _FakeHistory()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_component(self, data, train_size, component_name):
        """Generate test-set predictions for one component."""
        scaled = self.scalers[component_name].transform(
            data.reshape(-1, 1)).flatten()

        X_test = []
        for i in range(train_size - self.lookback,
                       len(scaled) - self.lookback):
            X_test.append(scaled[i: i + self.lookback])

        X_test = np.array(X_test)
        scaled_pred = self.models[component_name].predict(X_test)
        predictions = self.scalers[component_name].inverse_transform(
            scaled_pred).flatten()

        self.predictions[component_name] = predictions
        return predictions

    def predict(self, high_freq, low_freq, trend, train_size):
        """
        Predict on the test set via **direct summation** of components.
        No combined network required.
        """
        hf_pred = self.predict_component(high_freq, train_size, 'high_freq')
        lf_pred = self.predict_component(low_freq,  train_size, 'low_freq')
        tr_pred = self.predict_component(trend,     train_size, 'trend')

        # Align lengths (safety guard)
        min_len = min(len(hf_pred), len(lf_pred), len(tr_pred))
        return hf_pred[:min_len] + lf_pred[:min_len] + tr_pred[:min_len]

    # ------------------------------------------------------------------
    # Forecasting
    # ------------------------------------------------------------------

    def forecast_component(self, data, forecast_days, component_name):
        """Auto-regressive multi-step forecast for one component."""
        scaled = self.scalers[component_name].transform(
            data.reshape(-1, 1)).flatten()

        curr_input = scaled[-self.lookback:].copy()
        forecasts_scaled = []

        for _ in range(forecast_days):
            x = curr_input.reshape(1, self.lookback)
            next_val = self.models[component_name].predict(x, verbose=0)[0][0]
            forecasts_scaled.append(next_val)
            curr_input = np.append(curr_input[1:], next_val)

        forecasts = self.scalers[component_name].inverse_transform(
            np.array(forecasts_scaled).reshape(-1, 1))
        return forecasts.flatten()

    def forecast(self, high_freq, low_freq, trend, forecast_days):
        """
        Generate future forecasts via **direct summation** of components.
        """
        hf_fc = self.forecast_component(high_freq, forecast_days, 'high_freq')
        lf_fc = self.forecast_component(low_freq,  forecast_days, 'low_freq')
        tr_fc = self.forecast_component(trend,     forecast_days, 'trend')
        return hf_fc + lf_fc + tr_fc