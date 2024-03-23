# FYS5429 Project 1

This project aims to forecast global temperature changes using deep learning techniques, specifically Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Unit (GRU) networks. It utilizes temperature data from global meteorological stations from 1961 to 2019 to evaluate these models' effectiveness in capturing the temporal dynamics of climate change.

### Requirements
To run the Python programs, the following Python packages must be installed:
- TensorFlow
- Keras
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Structure
- `data_processing.py`: Script for preprocessing raw temperature data, including cleaning, normalization, and preparation for model training.
- `models.py`: Contains the implementation of RNN, LSTM, and GRU models using TensorFlow and Keras.
- `cross_val.py`: Script for evaluating model performance using time series cross-validation.
- `run_models.py`: Main script that orchestrates data preprocessing, model training, prediction, and evaluation.

### Run code
To successfully execute the code, please note that you might need to modify the file path in the script to correctly access the data file located in the 'Data' folder. Ensure that all required packages are installed, and then enter the following command in the terminal to run the codes: 

```bash
python3 data_processing.py
```

```bash
python3 run_models.py
```

```bash
python3 cross_val.py
```

### Dataset
The dataset used in this project includes temperature changes recorded at various global meteorological stations from 1961 to 2019. The original dataset, titled "Environment Temperature change," is sourced from Kaggle and can be found here: [Temperature Change Dataset](https://www.kaggle.com/datasets/sevgisarac/temperature-change).

