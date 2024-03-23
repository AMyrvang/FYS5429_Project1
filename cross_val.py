import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
import tensorflow as tf
import random as python_random

from models import SimpleRNNRegression, LSTMRegression, GRURegression

# Set seed for reproducibility
python_random.seed(2024)
np.random.seed(2024)
tf.random.set_seed(2024)

# Load the dataset
df = pd.read_csv("Data/Processed_Data/World_temperature_change.csv")

def create_dataset(data, n_lag=1):
    """
    Prepares dataset for LSTM/RNN models with a specified lag.
    
    Parameters:
        data: Scaled dataset as numpy array.
        n_lag: Number of lagged time steps.
        
    Returns:
        Tuple of numpy arrays: (X, y).
    """
    X, y = [], []
    for i in range(len(data) - n_lag):
        X.append(data[i:(i + n_lag), 0])
        y.append(data[i + n_lag, 0])
    return np.array(X), np.array(y)

# Scale the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[["Temperature Change"]].values)

# Create dataset with lag
n_lag = 12
X, y = create_dataset(data_scaled, n_lag)
X = X.reshape(X.shape[0], 1, X.shape[1])

def evaluate_model(model_class, X, y, scaler, n_splits=6, num_epochs=100, learning_rate=0.001):
    """
    Trains the model and evaluates it using Time Series Cross-Validation.
    Parameters:
        model_class: The model class to be evaluated (LSTMRegression or SimpleRNNRegression).
        X: Feature dataset.
        y: Target dataset.
        scaler: Scaler object for inverse transformation.
        n_splits: Number of splits for TimeSeriesSplit.
        num_epochs: Number of epochs for model training.
        learning_rate: Learning rate for the optimizer.
    Returns:
        List of RMSE scores for each split.
    """
    rmse_scores = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = model_class(units=50, num_epochs=num_epochs, learning_rate=learning_rate)
        model.fit(X_train, y_train)
        test_predict = model.predict(X_test)
        
        test_predict = scaler.inverse_transform(test_predict)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        rmse = sqrt(mean_squared_error(y_test_inv, test_predict))
        rmse_scores.append(rmse)
        
    return rmse_scores

# Run the models
rnn_rmse_scores = evaluate_model(SimpleRNNRegression, X, y, scaler, num_epochs=100)
lstm_rmse_scores = evaluate_model(LSTMRegression, X, y, scaler, num_epochs=100)
gru_rmse_scores = evaluate_model(GRURegression, X, y, scaler, num_epochs=100)

# Evaluate RNN Model
print("RNN RMSE scores:", rnn_rmse_scores)
print("RNN Average RMSE:", np.mean(rnn_rmse_scores))

# Evaluate LSTM Model
print("LSTM RMSE scores:", lstm_rmse_scores)
print("LSTM Average RMSE:", np.mean(lstm_rmse_scores))

# Evaluate GRU Model
print("GRU RMSE scores:", gru_rmse_scores)
print("GRU Average RMSE:", np.mean(gru_rmse_scores))
