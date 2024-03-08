#from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#Taken from project 1 in FYS4155
def mse(data, model):
	"""Calculate the mean square error of the fit.
	
	Args:
	    data (np.ndarray): data to fit
		model (np.ndarray): predicted model
		
	Returns:
	    np.ndarray: mean square error of fit
	"""
	return np.sum((data - model)**2) / len(data)


def r2(data, model):
	"""Calculate the R2 score of the fit

    Args:
        data (np.ndarray): original data to fit
        model (np.ndarray): predicted model
		
	Returns:
	    np.ndarray: R2 score of fit
    """
	return 1 - np.sum((data - model)**2) / np.sum((data - np.mean(data))**2)


file_path = ["Data/Temperature_Data/FAOSTAT_data_1-10-2022.csv"]
data = pd.read_csv(file_path)
data.head()












"""
# This is the data file i should use i think. "Data/Temperature_Data/FAOSTAT_data_1-10-2022.csv"
def read_data():
data = pd.read_csv(file_path, encoding='ISO-8859-1')
# Filter for annual temperature changes for Norway
norway_data = data[(data['Area'] == 'Norway') & (data['Element'] == 'Temperature change') & (data['Months'] == 'Meteorological year')]
# Select only the yearly columns
yearly_cols = norway_data.columns[norway_data.columns.str.startswith('Y')]
norway_temp_changes = norway_data[yearly_cols].values.flatten()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
norway_temp_changes_scaled = scaler.fit_transform(norway_temp_changes.reshape(-1, 1)).flatten()

# Function to create sequences
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data[i:(i + sequence_length)]
        label = data[i + sequence_length]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

# Define sequence length (e.g., use the past 5 years to predict the next year)
sequence_length = 5
X, y = create_sequences(norway_temp_changes_scaled, sequence_length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape input to be [samples, time steps, features] which is required for RNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

X_train.shape, X_test.shape, y_train.shape, y_test.shape
"""

