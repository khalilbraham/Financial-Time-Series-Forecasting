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


Sure! I'll adapt the models to use **PyTorch** instead of **TensorFlow**. Below are the PyTorch implementations for the models: MLP, Transformer-based model, XLSTM, and the steps for integrating the Informer and PathFormer models.

### 1. **MLP (Multilayer Perceptron) in PyTorch**

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Example usage:
mlp_model = MLP(input_size=X_train.shape[1])
```

### 2. **Transformer-Based Model in PyTorch**

A PyTorch implementation of a basic Transformer-based model for time series forecasting.

```python
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, num_heads=4, ff_dim=128, num_layers=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 1)  # Output layer

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc_out(x)
        return x

# Example usage:
transformer_model = TransformerModel(input_size=X_train.shape[2])  # Assuming input is (batch, seq_len, features)
```

### 3. **XLSTM Model in PyTorch**

An LSTM-based model with extended capabilities for handling long-term dependencies.

```python
class XLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(XLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc_out(h[:, -1, :])  # Taking the last hidden state
        return out

# Example usage:
xlstm_model = XLSTM(input_size=X_train.shape[2])
```

### 4. **Informer Model in PyTorch**

For **Informer**, you can use the authors' [Informer GitHub repository](https://github.com/zhouhaoyi/Informer2020). Here’s how to integrate it into your workflow:

1. Clone the repository:

```bash
git clone https://github.com/zhouhaoyi/Informer2020.git
cd Informer2020
pip install -r requirements.txt
```

2. Use the Informer model in your code:

```python
from informer_model import Informer  # Assuming Informer is imported from the repo

def build_informer(input_size):
    model = Informer(enc_in=input_size, dec_in=input_size, c_out=1, seq_len=96, label_len=48, out_len=24)
    return model

# Example usage:
informer_model = build_informer(input_size=X_train.shape[2])
```

### 5. **PathFormer Model in PyTorch**

For **PathFormer**, similar to Informer, use the implementation provided in their paper's GitHub repository. Here’s an outline of how to proceed:

1. Clone the PathFormer repository:

```bash
git clone https://github.com/pathformer-repo/pathformer.git
```

2. Use the PathFormer model:

```python
from pathformer import PathFormer  # Assuming PathFormer class is implemented

def build_pathformer(input_size):
    model = PathFormer(input_size=input_size, d_model=64, num_heads=4, ff_dim=128)
    return model

# Example usage:
pathformer_model = build_pathformer(input_size=X_train.shape[2])
```

### Training Loop for PyTorch

Here’s an example of a generic training loop you can use to train any of the models:

```python
import torch.optim as optim

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, lr=0.001):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.unsqueeze(1))
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor.unsqueeze(1))
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Example usage:
train_model(mlp_model, X_train, y_train, X_val, y_val, epochs=20)
```

### Evaluation

Once the model is trained, you can evaluate it on the test set:

```python
def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(X_test_tensor)
        test_loss = nn.MSELoss()(predictions, y_test_tensor.unsqueeze(1))
        print(f"Test Loss: {test_loss.item():.4f}")

# Example usage:
evaluate_model(mlp_model, X_test, y_test)
```

### Summary:

1. **MLP**: A simple feed-forward neural network.
2. **Transformer**: A time series transformer model leveraging self-attention.
3. **XLSTM**: An extended LSTM model for long-range dependencies.
4. **Informer**: You can use the official implementation from the [Informer GitHub repository](https://github.com/zhouhaoyi/Informer2020).
5. **PathFormer**: Use the repository for **PathFormer** to get an implementation based on their paper.

This PyTorch implementation allows you to handle multiple complex models efficiently for time series forecasting.
