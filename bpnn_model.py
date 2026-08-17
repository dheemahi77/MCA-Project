import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class BPNNModel:
    """
    Back Propagation Neural Network Model for time series forecasting
    """
    
    def __init__(self, lookback=10):
        """
        Initialize the BPNN model
        
        Parameters:
        -----------
        lookback : int
            Number of previous time steps to use as input features
        """
        self.model = None
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def create_model(self, input_dim):
        """
        Create a neural network model
        
        Parameters:
        -----------
        input_dim : int
            Input dimension (number of features)
        """
        self.model = Sequential()
        
        # First hidden layer
        self.model.add(Dense(64, activation='relu', input_dim=input_dim))
        self.model.add(Dropout(0.2))
        
        # Second hidden layer
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        
        # Output layer
        self.model.add(Dense(1))
        
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
        
        return self.model
    
    def create_dataset(self, data, train_size=None):
        """
        Create sequences for time series prediction
        
        Parameters:
        -----------
        data : numpy.ndarray
            Time series data
        train_size : int, optional
            Size of training set. If None, all data will be used.
        
        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test)
        """
        # Scale data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        X = []
        y = []
        
        # Create sequences
        for i in range(len(scaled_data) - self.lookback):
            X.append(scaled_data[i:i+self.lookback])
            y.append(scaled_data[i+self.lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        if train_size is None:
            return X, y, None, None
        else:
            X_train, X_test = X[:train_size-self.lookback], X[train_size-self.lookback:]
            y_train, y_test = y[:train_size-self.lookback], y[train_size-self.lookback:]
            
            return X_train, y_train, X_test, y_test
    
    def train(self, data, train_size, epochs=100, batch_size=32):
        """
        Train the model
        
        Parameters:
        -----------
        data : numpy.ndarray
            Time series data
        train_size : int
            Size of training set
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        history : tensorflow.keras.callbacks.History
            Training history
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self.create_dataset(data, train_size)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history
    
    def predict(self, data, train_size):
        """
        Generate predictions
        
        Parameters:
        -----------
        data : numpy.ndarray
            Time series data
        train_size : int
            Size of training set
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Scale data
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Generate test sequences
        X_test = []
        for i in range(train_size - self.lookback, len(scaled_data) - self.lookback):
            X_test.append(scaled_data[i:i+self.lookback])
        
        X_test = np.array(X_test)
        
        # Generate predictions
        scaled_predictions = self.model.predict(X_test)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(scaled_predictions)
        
        return predictions.flatten()
    
    def forecast(self, data, forecast_days):
        """
        Generate future forecasts
        
        Parameters:
        -----------
        data : numpy.ndarray
            Historical time series data
        forecast_days : int
            Number of days to forecast
            
        Returns:
        --------
        numpy.ndarray
            Forecasted values
        """
        # Scale the entire dataset
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        # Use the last 'lookback' points as initial input
        curr_input = scaled_data[-self.lookback:]
        
        # Generate forecasts
        forecasts_scaled = []
        
        for _ in range(forecast_days):
            # Reshape input
            curr_input_reshaped = curr_input.reshape(1, self.lookback)
            
            # Get prediction
            next_pred = self.model.predict(curr_input_reshaped)[0][0]
            
            # Store prediction
            forecasts_scaled.append(next_pred)
            
            # Update input (remove oldest value, add newest prediction)
            curr_input = np.append(curr_input[1:], next_pred)
        
        # Convert forecasts back to original scale
        forecasts = self.scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1))
        
        return forecasts.flatten()
