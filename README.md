# Time series forecasting: ARIMA vs Deep Learning

## Project Overview

This project, completed as part of the MVA 2023/2024 course, investigates the effectiveness of various time series models in forecasting and analyzing repairable system failure data. The main focus is on comparing traditional ARIMA models with modern deep learning approaches, including Artificial Neural Networks (ANN) and Long Short-Term Memory (LSTM) networks.

## Project Objectives

1. **Compare ARIMA with Deep Learning Models**: Evaluate the performance of ARIMA, ANN, and LSTM models in forecasting tasks.
2. **Use Case Applications**: Apply these models to specific use cases such as financial stock forecasting and predictive maintenance tasks.
3. **Performance Metrics**: Compare models based on metrics like predictive errors and the ability to detect trends and reversals.

## Author

- **Name**: Mohamed Khalil Braham
- **Email**: [khalil.braham@telecom-paris.fr](mailto:khalil.braham@telecom-paris.fr)

## Introduction

The report delves into the traditional ARIMA model's usage in time series forecasting and its comparison with ANN and LSTM models. It aims to bridge the gap in literature regarding the application of these models in reliability analysis and predictive maintenance.

## Methodology

### ARIMA (Autoregressive Integrated Moving Average)

- **Components**:
  - **AR**: Autoregression, using the dependency between observations.
  - **I**: Integration, making the series stationary by differencing.
  - **MA**: Moving Average, using the dependency between an observation and residual errors from a moving average model.
- **Model Parameters**:
  - `p`: Order of the autoregression.
  - `d`: Degree of differencing.
  - `q`: Order of the moving average.

### ANN (Artificial Neural Network)

- **Structure**: Composed of interconnected processing elements (nodes) that work simultaneously to solve problems.
- **Usage**: Suitable for modeling nonlinear relationships in time series data.
- **Model Parameters**: Includes hidden layers and activation functions to model complex patterns.

### LSTM (Long Short-Term Memory)

- **Structure**: Complex neural network module designed to remember long-term dependencies.
- **Components**:
  - **Forget Gate**: Decides what information to discard.
  - **Input Gate**: Updates the cell state with new information.
  - **Output Gate**: Decides the output based on the cell state.
- **Model Parameters**: Includes cell memory and hidden state for each time step.

## Data

### YFinance Stock Data

- **Description**: Historical stock prices of 'AAPL' from January 1, 2019, to January 1, 2023.
- **Purpose**: Compare different forecasting techniques on a dataset with variability and trends.

### Azure Predictive Maintenance Dataset

- **Description**: Data includes telemetry, failures, and maintenance records for machines.
- **Purpose**: Evaluate models on predicting machine failures using sensor data.

## Data Diagnosis

- **Stationarity**: Used techniques like differencing and the ADF test to ensure the data is stationary.
- **Seasonality**: Analyzed using seasonal decomposition to understand trends and residuals.

## Results

### YFinance Stock Dataset

- **Qualitative Results**: Visual comparison of ARIMA, LSTM, and ANN predictions for short-term and long-term forecasting.
- **Quantitative Results**: Comparison based on Root Mean Squared Error (RMSE).
  - **Short-term RMSE**: 
    - ARIMA: 3.34
    - ANN: 3.6
    - LSTM: 2.61
  - **Long-term RMSE**: 
    - ARIMA: 2.8
    - ANN: 8.2
    - LSTM: 2.3

### Predictive Maintenance Dataset

- **Challenges**: Dataset imbalance with few failure instances made classification difficult.
- **Findings**: Models consistently predicted non-failure due to the skewed dataset.

## Conclusion

- **LSTM Performance**: Demonstrated superior performance in both short-term and long-term forecasting compared to ARIMA and ANN.
- **Dataset Suitability**: Highlighted the importance of balanced datasets for meaningful predictive maintenance and failure detection.

## References

- A comparative study of neural network and Box-Jenkins ARIMA modeling in time series prediction, Siong Lin Ho, Ngee Ann Polytechnic.
- A Review of ARIMA vs. Machine Learning Approaches for Time Series Forecasting in Data Driven Networks, Vaia I. Kontopoulou, Athanasios D. Panagopoulos, Ioannis Kakkos, and George K. Matsopoulos.
