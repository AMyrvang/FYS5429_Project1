import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random as python_random
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from models import LSTMRegression, SimpleRNNRegression, GRURegression


# Set the seaborn theme
sns.set_theme("notebook","whitegrid", palette="colorblind")
cm = 1/2.54

# Define plot parameters
params = {
    "legend.fontsize": "9",
    "font.size": "9",
    "figure.figsize": (8.647 * cm, 12.0 * cm),  # figsize for two-column latex document
    "axes.labelsize": "9",
    "axes.titlesize": "9",
    "xtick.labelsize": "9",
    "ytick.labelsize": "9",
    "legend.fontsize": "7",
    "lines.markersize": "3.0",
    "lines.linewidth": "1.5",
}
plt.rcParams.update(params)


# Seed setting for reproducibility
python_random.seed(2024)
np.random.seed(2024)
tf.random.set_seed(2024)

# Load the dataset
df = pd.read_csv("Data/Processed_Data/World_temperature_change.csv")

def create_dataset(data, n_lag=1):
    """
    Creates dataset with specified lag.
    Parameters:
        data: Input dataset.
        n_lag: Number of lag periods.
    Returns:
        Tuple of numpy arrays (X, y).
    """
    X, y = [], []
    for i in range(len(data) - n_lag - 1):
        a = data[i:(i + n_lag), 0]
        X.append(a)
        y.append(data[i + n_lag, 0])
    return np.array(X), np.array(y)

# Data preparation
value_col = "Temperature Change"
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[[value_col]].values)

# Splitting the dataset
train_size = int(len(data_scaled) * 0.8136)
test_size = len(data_scaled) - train_size
train, test = data_scaled[0:train_size, :], data_scaled[train_size:len(data_scaled), :]

look_back = 18
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
y_test = np.reshape(y_test, (y_test.shape[0], 1))

def train_and_predict(model, X_train, y_train, X_test, scaler):
    """
    Trains the model and predicts on training and test data.
    Parameters:
        model: The model to train.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        scaler: Scaler object for inverse transformation.
    Returns:
        Tuple of numpy arrays (train_predict, test_predict).
    """
    model.fit(X_train, y_train)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    return train_predict, test_predict

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the mean absolute percentage error.
    Parameters:
        y_true: Actual values.
        y_pred: Predicted values.
    Returns:
        MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Training and evaluating models
rnn_model = SimpleRNNRegression(units=12, num_epochs=1000, learning_rate=0.001)
rnn_train_predict, rnn_test_predict = train_and_predict(rnn_model, X_train, y_train, X_test, scaler)

lstm_model = LSTMRegression(units=12, num_epochs=1000, learning_rate=0.001)
lstm_train_predict, lstm_test_predict = train_and_predict(lstm_model, X_train, y_train, X_test, scaler)

gru_model = GRURegression(units=12, num_epochs=1000, learning_rate=0.001)
gru_train_predict, gru_test_predict = train_and_predict(gru_model, X_train, y_train, X_test, scaler)

# Metrics calculation
rnn_mse = mean_squared_error(scaler.inverse_transform(y_test), rnn_test_predict)
rnn_rmse = sqrt(rnn_mse)
rnn_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_test), rnn_test_predict)
print(f"RNN MSE: {rnn_mse}, RMSE: {rnn_rmse}, MAPE: {rnn_mape}%")

lstm_mse = mean_squared_error(scaler.inverse_transform(y_test), lstm_test_predict)
lstm_rmse = sqrt(lstm_mse)
lstm_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_test), lstm_test_predict)
print(f"LSTM MSE: {lstm_mse}, RMSE: {lstm_rmse}, MAPE: {lstm_mape}%")

gru_mse = mean_squared_error(scaler.inverse_transform(y_test), gru_test_predict)
gru_rmse = sqrt(gru_mse)
gru_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_test), gru_test_predict)
print(f"GRU MSE: {gru_mse}, RMSE: {gru_rmse}, MAPE: {gru_mape}%")


# RNN Predictions plot
plt.subplot(3, 1, 1)
plt.plot(scaler.inverse_transform(y_test), color="black")
plt.plot(rnn_test_predict, color="blue")
plt.title("RNN")

# LSTM Predictions plot
plt.subplot(3, 1, 2)
plt.plot(scaler.inverse_transform(y_test), color="black")
plt.plot(lstm_test_predict, color="red")
plt.title("LSTM")
plt.ylabel("Temperature change [\u00B0C]")

# GRU Predictions plot
plt.subplot(3, 1, 3)
plt.plot(scaler.inverse_transform(y_test), color="black")
plt.plot(gru_test_predict, color="green")
plt.title("GRU")
plt.xlabel("Time [Months]")


plt.tight_layout() 
plt.savefig("Figs/RNN_LSTM_and_GRU.png")

