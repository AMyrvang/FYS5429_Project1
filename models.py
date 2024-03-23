from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, GRU, Dense

class SimpleRNNRegression:
    """
    A regression model using SimpleRNN for univariate time series forecasting.
    Attributes:
        units (int): The number of units in the RNN layer.
        num_epochs (int): The number of epochs for training the model.
        learning_rate (float): The learning rate for the optimizer.
        verbose (int): Verbosity mode.
    """
    def __init__(self, units=100, num_epochs=10, learning_rate=0.001, verbose=1):
        self.units = units
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X_train, y_train):
        """
        Fits the model to the training data.
        Parameters:
            X_train: Training data features.
            y_train: Training data target.
        """
        model = Sequential()
        model.add(SimpleRNN(self.units, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose)
        self.model = model

    def predict(self, X_test):
        """
        Predicts using the trained model on the test data.
        Parameters:
            X_test: Test data for prediction.
        """
        return self.model.predict(X_test)

class LSTMRegression:
    """
    A regression model using LSTM for univariate time series forecasting.
    Attributes are similar to SimpleRNNRegression.
    """
    def __init__(self, units=100, num_epochs=10, learning_rate=0.001, verbose=1):
        self.units = units
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X_train, y_train):
        model = Sequential()
        model.add(LSTM(self.units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(self.units))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose)
        self.model = model

    def predict(self, X_test):
        return self.model.predict(X_test)

class GRURegression:
    """
    A regression model using GRU for univariate time series forecasting.
    Attributes are similar to SimpleRNNRegression.
    """
    def __init__(self, units=100, num_epochs=10, learning_rate=0.001, verbose=1):
        self.units = units
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

    def fit(self, X_train, y_train):
        model = Sequential()
        model.add(GRU(self.units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(GRU(self.units))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train, y_train, epochs=self.num_epochs, verbose=self.verbose)
        self.model = model

    def predict(self, X_test):
        return self.model.predict(X_test)
