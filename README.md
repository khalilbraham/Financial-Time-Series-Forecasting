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


## Generating Signals Positively Correlated with a Target Signal: Techniques and Approaches

In financial time series forecasting, the goal is often not only to forecast the exact future values of a time series but also to generate signals that are positively correlated with a target signal, such as stock prices or other market indicators. Positively correlated signals can help in applications such as portfolio management, risk modeling, and algorithmic trading, where the aim is to capture the general direction of the market, even if the exact forecast may not be perfect.

In this article, we explore various sophisticated techniques that can generate signals positively correlated with a target signal. We will discuss traditional machine learning, deep learning, reinforcement learning, adversarial training, and constrained optimization approaches, each with its own strengths and use cases.

### 1. **Why Positive Correlation Matters in Financial Forecasting**

In finance, many strategies rely on capturing the **trend** or **direction** of a target signal rather than its exact value. A positively correlated signal moves in the same direction as the target signal, which is particularly important for strategies like **trend-following** or **momentum-based trading**. If a model can consistently generate signals that are positively correlated with the target signal (e.g., stock prices), it can drive actionable trading decisions.

### 2. **Traditional Machine Learning Approach**

A traditional machine learning model can be trained to forecast time series, but to specifically optimize for **positive correlation** with the target signal, we need to tweak the objective function.

#### Technique: **Correlation-Optimized Loss Function**

One approach is to modify the training objective by using a **correlation-based loss function**. Rather than focusing solely on minimizing the prediction error (e.g., MSE), we add a term to the loss function that maximizes the **Pearson correlation** between the model's predictions and the target signal.

#### Custom Loss Function:
```python
import torch
import torch.nn.functional as F

def correlation_loss(y_true, y_pred):
    y_true_centered = y_true - torch.mean(y_true)
    y_pred_centered = y_pred - torch.mean(y_pred)
    correlation = torch.sum(y_true_centered * y_pred_centered) / torch.sqrt(
        torch.sum(y_true_centered ** 2) * torch.sum(y_pred_centered ** 2)
    )
    return -correlation  # We maximize correlation, so we minimize the negative correlation

def combined_loss(y_true, y_pred, alpha=0.5):
    mse_loss = F.mse_loss(y_pred, y_true)
    corr_loss = correlation_loss(y_true, y_pred)
    return mse_loss + alpha * corr_loss
```

This loss function combines **MSE** (to ensure the predictions are reasonably accurate) and **negative correlation** (to encourage a strong positive correlation with the target). This is a simple yet effective way to generate signals positively correlated with the target.

### 3. **Reinforcement Learning Approach**

**Reinforcement Learning (RL)** can be a powerful approach to train models to generate positively correlated signals. In RL, the model is treated as an agent that interacts with an environment (the financial data), and its goal is to maximize a **reward** based on positive correlation with the target signal.

#### Steps:
1. **Agent**: The model (e.g., a neural network) that generates signals.
2. **Action**: The output of the model at each time step (e.g., a buy, sell, or hold signal).
3. **Reward**: The reward is based on the **correlation** between the predicted signal and the target signal. A higher correlation leads to a higher reward.

#### Reinforcement Learning Workflow:
```python
import torch
import torch.nn as nn
import torch.optim as optim

class RLModel(nn.Module):
    def __init__(self, input_size):
        super(RLModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)  # Output signal

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def reward_function(y_true, y_pred):
    return correlation_loss(y_true, y_pred)

# Basic RL training loop
def train_rl_model(model, X_train, y_train, num_episodes=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    gamma = 0.99  # Discount factor for future rewards

    for episode in range(num_episodes):
        model.train()
        optimizer.zero_grad()

        # Forward pass: model predicts signals
        y_pred = model(X_train)
        
        # Compute reward (correlation between prediction and target)
        reward = reward_function(y_train, y_pred)

        # RL-based loss: maximize reward
        loss = -reward
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}, Reward: {reward.item():.4f}")

# Train RL Model
model = RLModel(input_size=X_train.shape[1])
train_rl_model(model, X_train, y_train, num_episodes=100)
```

In this approach, the model continuously interacts with the data and updates its signals to maximize the positive correlation with the target signal.

### 4. **Adversarial Learning Approach**

In **adversarial training**, we employ a generator-discriminator framework. The generator is responsible for generating signals, while the discriminator evaluates whether the generated signals are positively correlated with the target signal. The goal of the generator is to "fool" the discriminator into thinking that its signals are highly correlated with the target.

#### Steps:
1. **Generator**: A model that generates signals based on input features.
2. **Discriminator**: A model that evaluates whether the generated signals are positively correlated with the target signal.
3. **Adversarial Loss**: The generator tries to maximize correlation, while the discriminator tries to distinguish between good and bad signals based on their correlation.

#### Adversarial Training Workflow:
```python
class Generator(nn.Module):
    def __init__(self, input_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def adversarial_loss(discriminator, generator_output, real_correlation):
    pred_correlation = discriminator(generator_output)
    return F.binary_cross_entropy(pred_correlation, real_correlation)

# Training adversarially
def train_adversarial(generator, discriminator, X_train, y_train, num_epochs=100, lr=0.001):
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # Step 1: Train discriminator
        discriminator.train()
        generator.eval()
        
        y_pred = generator(X_train)
        real_correlation = torch.ones(y_pred.size(0), 1)  # Positive correlation target
        
        d_optimizer.zero_grad()
        d_loss = adversarial_loss(discriminator, y_pred, real_correlation)
        d_loss.backward()
        d_optimizer.step()

        # Step 2: Train generator
        generator.train()
        discriminator.eval()

        g_optimizer.zero_grad()
        y_pred = generator(X_train)
        g_loss = -correlation_loss(y_train, y_pred)  # Maximize positive correlation
        g_loss.backward()
        g_optimizer.step()

        print(f"Epoch {epoch+1}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Example usage:
generator = Generator(input_size=X_train.shape[1])
discriminator = Discriminator(input_size=1)  # Takes predicted signal as input
train_adversarial(generator, discriminator, X_train, y_train, num_epochs=100)
```

### 5. **Constrained Optimization Approach**

A more sophisticated approach involves **constrained optimization**, where the model is trained to minimize the prediction error (e.g., MSE) under the constraint that the correlation with the target signal is above a certain threshold.

#### Technique: **Differentiable Correlation Constraint**

This approach penalizes the model if the correlation between its predictions and the target signal drops below a given threshold.

```python
def constrained_correlation_loss(y_true, y_pred, min_corr=0.1, penalty_weight=10):
    mse_loss = F.mse_loss(y_pred, y_true)
    corr = correlation_loss(y_true, y_pred)
    
    # If correlation is less than min_corr, add a penalty to the loss
    penalty = torch.relu(min_corr - corr) * penalty_weight
    
    return mse_loss + penalty
```

This method provides direct control over the degree of correlation and enforces constraints during the training process, ensuring that the generated signals maintain the desired correlation properties.

### Conclusion

Generating signals that are positively correlated with a target signal can be achieved through various sophisticated techniques:

- **Traditional approaches** modify the loss function to incorporate correlation objectives.
- **Reinforcement learning** models treat

### Meta-Learning for Generalization to Out-of-Sample Data

Meta-learning, or "learning to learn," is a machine learning paradigm designed to help models generalize better across tasks and unseen data by training them to quickly adapt to new tasks using only a few examples. Meta-learning is particularly effective in scenarios where a model encounters new environments (e.g., financial markets with changing conditions), as it can adapt faster and generalize better compared to traditional models.

There are different approaches to meta-learning, and they can be leveraged to improve a model's performance on out-of-sample data (data that differs from the training data in terms of distributions or characteristics). Meta-learning achieves this by focusing not just on the model's performance on a single task but on its ability to adapt to various tasks. This ability can translate to better generalization when applied to new, unseen data distributions, such as live financial data in Ahmed's case.

### Key Meta-Learning Approaches

1. **Model-Agnostic Meta-Learning (MAML)**:
   MAML is a gradient-based meta-learning approach where the model learns an initialization that can be fine-tuned quickly with a few gradient steps on new tasks. In financial forecasting, this means the model can start from a well-initialized state and adjust to new market conditions (e.g., changes in volatility or correlations) after seeing a few new data points.

   - **How MAML Works**:
     - **Meta-Training**: The model trains on a variety of related tasks, updating its parameters such that it learns a good initialization that works across tasks.
     - **Adaptation (Inner Loop)**: For each task, the model takes a few gradient descent steps to fine-tune to the specific task.
     - **Meta-Update (Outer Loop)**: The model updates its parameters to improve its performance across all tasks after the adaptation.

   - **Application to Finance**: In financial time series forecasting, MAML could be trained across different market regimes (e.g., high volatility, low volatility, different asset classes). When new live data arrives, the model can quickly adapt to the new conditions using a few steps of gradient descent, improving its generalization to the out-of-sample data.

   **Reference**:
   - Finn, C., Abbeel, P., & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." [Paper Link](https://arxiv.org/abs/1703.03400)

2. **Recurrent Neural Networks (RNN) for Meta-Learning (RL²)**:
   In this approach, the model itself is a recurrent network (such as LSTM or GRU) that learns how to update its internal states based on past experience. This can help in sequential decision-making and prediction tasks where the data distribution changes over time.

   - **How RL² Works**:
     - The RNN is trained across multiple tasks or sequences, and it learns to encode experiences into its hidden states.
     - When applied to new data, the RNN can adapt to the changing patterns by learning how to modify its internal states based on new inputs.
     
   - **Application to Finance**: This approach is useful when the financial market evolves over time, and the model needs to continuously learn and adapt. The internal states of the RNN can "remember" how to handle certain market regimes and can dynamically adjust to new regimes when the market changes.

   **Reference**:
   - Wang, J. X., Kurth-Nelson, Z., Tirumala, D., et al. (2016). "Learning to Reinforcement Learn." [Paper Link](https://arxiv.org/abs/1611.05763)

3. **Metric-Based Meta-Learning (Prototypical Networks)**:
   This approach involves learning a metric space where tasks are represented by prototypes, and new tasks can be classified by comparing them to these prototypes. This is useful for few-shot learning, where we don’t have much data to adapt to new conditions.

   - **How Prototypical Networks Work**:
     - During meta-training, the model learns to represent each task by a prototype, typically the mean of the embeddings for the support set (examples from a task).
     - During inference on a new task, the model compares new examples to these prototypes and predicts the task based on similarity (in an embedding space).

   - **Application to Finance**: Financial data is often dynamic and regime-based (bull markets, bear markets, etc.). A prototypical network can help by learning prototypes for different regimes and quickly classifying new market data based on how similar it is to previously learned regimes.

   **Reference**:
   - Snell, J., Swersky, K., & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning." [Paper Link](https://arxiv.org/abs/1703.05175)

4. **Memory-Augmented Neural Networks (MANNs)**:
   In MANNs, the model is augmented with an external memory component that stores knowledge about different tasks and allows the model to recall this knowledge when needed. This can enable rapid learning and adaptation to new tasks by retrieving relevant patterns from memory.

   - **How MANNs Work**:
     - The model consists of a neural network and a memory bank where important task-related information is stored.
     - When new data comes in, the model can access this memory bank to retrieve information about similar tasks and apply it to adapt quickly.
   
   - **Application to Finance**: Financial time series can change abruptly due to macroeconomic events, regime shifts, or market anomalies. A MANN could store patterns from previous shifts and anomalies and recall them when similar events occur in live data, allowing for fast adaptation.

   **Reference**:
   - Santoro, A., Bartunov, S., Botvinick, M., et al. (2016). "One-shot Learning with Memory-Augmented Neural Networks." [Paper Link](https://arxiv.org/abs/1605.06065)

### Meta-Learning in Financial Applications

In financial forecasting, data distributions can shift frequently due to economic conditions, geopolitical events, and market sentiment changes. Traditional machine learning models often struggle with these changes because they are trained on fixed historical data that may no longer be representative of current market conditions. Meta-learning approaches can help by training models to:

- **Adapt to New Regimes**: The financial markets exhibit different regimes (e.g., periods of high or low volatility). Meta-learning allows the model to adapt to these new regimes with minimal retraining.
  
- **Handle Few-Shot Scenarios**: In financial time series, there may be limited data in certain market conditions (e.g., after a crash). Meta-learning can help models quickly adapt to such scenarios by leveraging few-shot learning techniques.

- **Robust Generalization**: Because meta-learning focuses on generalizing across tasks, models trained with meta-learning can better handle out-of-sample data that does not follow the same distribution as the training data. This is particularly important for live financial data, which often differs from both in-sample and out-of-sample validation data.

### Meta-Learning Implementation Strategy for Financial Forecasting

1. **Task Sampling**: Create multiple tasks (e.g., forecasting different asset classes or different time periods under various economic regimes) to train the meta-learner. These tasks can be different in terms of volatility, trend, or the underlying asset.

2. **Training Process**:
   - For gradient-based approaches (e.g., MAML), you can perform task-specific training (inner loop) followed by a meta-update step (outer loop).
   - For memory-based approaches, the model will store and retrieve relevant market information from the memory bank when adapting to new market regimes.

3. **Adaptation**: When the model encounters new, unseen financial data, it will quickly adapt using a few gradient updates (in the case of MAML) or by accessing stored knowledge from similar past regimes (in the case of MANNs).

4. **Evaluation on Live Data**: Once the model is trained using meta-learning, you can evaluate it on live, out-of-sample financial data to assess how well it generalizes. For example, you can test the model’s ability to handle market regime shifts, such as moving from a bull market to a bear market.

### Example Workflow with MAML (in PyTorch)

Here’s a simplified workflow of how you could apply MAML for financial forecasting using PyTorch:

```python
import torch
from torch import nn, optim
import numpy as np

# Sample Task: Suppose we have multiple tasks (different financial markets)
class FinancialTask:
    def __init__(self, data):
        self.data = data  # Time-series data

    def sample(self):
        # Simulate sampling of time series data for meta-learning
        X, y = self.data[:, :-1], self.data[:, -1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Define the model (MLP or LSTM for time series prediction)
class MetaLearner(nn.Module):
    def __init__(self, input_size):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# MAML Inner and Outer Loops
def maml_inner_loop(model, task, learning_rate):
    X_train, y_train = task.sample()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Inner loop: One or more steps of gradient descent on task-specific data
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    return model

def maml_outer_loop(model, tasks, meta_lr):
    meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    for epoch in range(epochs):
        meta_optimizer.zero_grad()

        # Simulate meta-training over multiple tasks
        for task in tasks:
            # Clone the model to avoid overwriting it
            task_model = deepcopy(model)
            task_model = maml_inner_loop(task_model, task, learning_rate=0.01)

            # Calculate loss after adaptation
            X_test, y_test = task.sample()  # Sample test data
            y_pred = task_model(X_test)
            task_loss = nn.MSELoss()(y_pred, y_test)
            task_loss.backward()

        # Meta-update
        meta_optimizer.step()
```

### Conclusion

Meta-learning, particularly approaches like MAML, RL², and memory-augmented networks, is a powerful strategy for improving generalization in financial forecasting. By focusing on the model's ability to adapt to new tasks or changing environments, meta-learning techniques can make models more resilient to shifts in market conditions and out-of-sample data.

This flexibility is crucial in financial markets, where the data distribution is often non-stationary and subject to sudden changes. By leveraging meta-learning, you can create models that not only perform well on historical data but also quickly adapt to live market data, enhancing their robustness and performance.

#### References for Further Reading:
1. **Finn, C., Abbeel, P., & Levine, S. (2017)**: "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks." [Paper Link](https://arxiv.org/abs/1703.03400)
2. **Snell, J., Swersky, K., & Zemel, R. (2017)**: "Prototypical Networks for Few-shot Learning." [Paper Link](https://arxiv.org/abs/1703.05175)
3. **Santoro, A., Bartunov, S., Botvinick, M., et al. (2016)**: "One-shot Learning with Memory-Augmented Neural Networks." [Paper Link](https://arxiv.org/abs/1605.06065)
4. **Wang, J. X., Kurth-Nelson, Z., Tirumala, D., et al. (2016)**: "Learning to Reinforcement Learn." [Paper Link](https://arxiv.org/abs/1611.05763)
