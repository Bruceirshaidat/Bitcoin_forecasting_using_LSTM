# Bitcoin Forecasting using LSTM
This repository contains a project for forecasting the closing price of Bitcoin using a Long Short-Term Memory (LSTM) neural network. The model is built with TensorFlow and Keras and trained on historical Bitcoin price data.

## Overview

The project follows a comprehensive workflow for time series forecasting:
1.  **Data Loading and Inspection**: The `bitcoin.csv` dataset, containing 15-minute interval data, is loaded, cleaned, and explored.
2.  **Data Preprocessing**: The 'close' price is selected for forecasting and normalized using `MinMaxScaler`.
3.  **Train-Test Split**: The data is chronologically split into training and testing sets.
4.  **Data Sequencing**: The time series data is transformed into sequences using a sliding window of 60 time steps.
5.  **Model Building**: A stacked LSTM model is constructed to capture temporal dependencies.
6.  **Training and Evaluation**: The model is trained on the training data and evaluated on the test set to measure its performance.
7.  **Forecasting**: The trained model is used to predict future Bitcoin prices.

## Dataset

The model is trained on a dataset containing historical Bitcoin prices with the following columns: `timestamp`, `open`, `high`, `low`, `close`, and `volume`. The data is preprocessed to ensure it's clean, sorted by time, and has no missing values. The target variable for this project is the `close` price.

The data is split into a training set and a test set, where the test data begins from January 1, 2025.



## Model Architecture

A Sequential model is implemented using Keras, featuring two LSTM layers followed by Dense layers for the final prediction.

```python
model = Sequential([
    LSTM(50, return_sequences= True, input_shape= (x_train.shape[1], 1)),
    LSTM(64, return_sequences= False),
    Dense(32),
    Dense(16),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=["mean_absolute_error"])
```
The model is trained for 100 epochs with a batch size of 64.

## Results

The model achieves high accuracy on the test set.

*   **Test MAPE**: 0.60%
*   **Test Accuracy (1 - MAPE)**: 99.40%
*   **RMSE**: 548.46

### Model Performance

The following chart illustrates the model's performance by comparing the predicted closing prices against the actual values for the test set.



The predictions (red) closely follow the actual test data (blue), demonstrating the model's effectiveness.

### Future Price Forecast

The model was also used to forecast the next 30 time steps beyond the available data.



## How to Use

### Prerequisites

You need the following libraries installed to run the notebook:
- `numpy`
- `pandas`
- `tensorflow`
- `scikit-learn`
- `matplotlib`

You can install them using pip:
```bash
pip install numpy pandas tensorflow scikit-learn matplotlib
```

### Running the Project

1.  Clone this repository:
    ```bash
    git clone https://github.com/Bruceirshaidat/Bitcoin_forecasting_using_LSTM.git
    cd Bitcoin_forecasting_using_LSTM
    ```
2.  Ensure you have the `bitcoin.csv` file in the root directory.
3.  Open and run the `Bitcoin_forecasting_using_LSTM.ipynb` notebook in a Jupyter environment or Google Colab.

The notebook will execute all steps from data loading to forecasting and will save the trained model as `my_model.h5`.
