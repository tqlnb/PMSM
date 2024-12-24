import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('data/merged.csv')

# Step 2: Generate time step feature (assuming data is sorted by 'unique_time')
df['time_step'] = np.arange(len(df)) * 0.1  # 每一步 0.1s 的固定时间间隔

# Features and targets
features = ['iq', 'id', 'vd', 'vq', 'time_step']
target = ['torque', 'speed']

# Step 3: Normalize the features and targets
X = df[features].values
y = df[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Step 4: Sliding window for time series data
def create_sliding_window(data, target, window_size):
    """
    Creates sliding window data for time series modeling.

    Args:
        data (np.array): Feature data.
        target (np.array): Target data.
        window_size (int): Number of time steps in the window.

    Returns:
        X, y: Features and targets for the model.
    """
    X_window, y_window = [], []
    for i in range(len(data) - window_size):
        X_window.append(data[i:i + window_size])  # Use the past `window_size` steps
        y_window.append(target[i + window_size])  # Predict the next time step's target
    return np.array(X_window), np.array(y_window)

# Parameters
window_size = 10  # Use the past 10 time steps

# Prepare data
X_window, y_window = create_sliding_window(X, y, window_size)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.2, random_state=42)

# Reshape for 1D-CNN (samples, channels, sequence_length)
X_train = X_train.transpose(0, 2, 1)  # Shape: (samples, channels, sequence_length)
X_test = X_test.transpose(0, 2, 1)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 5: Define the 1D-CNN model
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=len(features), out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * window_size, 64)
        self.fc2 = nn.Linear(64, len(target))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Step 6: Initialize model, loss, and optimizer
model = CNN1D()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train the model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Step 8: Final evaluation
#model.eval()
#with torch.no_grad():
#    predictions = model(X_test)
#    mse = criterion(predictions, y_test).item()
#    print(f"Final Test MSE: {mse:.4f}")
#
## Step 9: Inverse transform the predictions (optional)
#predictions = scaler_y.inverse_transform(predictions.numpy())
#y_test = scaler_y.inverse_transform(y_test.numpy())

from sklearn.metrics import r2_score


# Step 8: Final evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test).item()

    # Convert to NumPy arrays for R^2 calculation
    predictions_np = predictions.numpy()
    y_test_np = y_test.numpy()

    # Calculate R^2
    r2 = r2_score(y_test_np, predictions_np)

print(f"Final Test MSE: {mse:.4f}")
print(f"Final Test R^2 Score: {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(y_train.numpy().flatten(), bins=30, alpha=0.7, label='Train')
plt.hist(y_test.numpy().flatten(), bins=30, alpha=0.7, label='Test')
plt.legend()
plt.title("Distribution of Target Values")
plt.subplot(1, 2, 2)
plt.boxplot([y_train.numpy().flatten(), y_test.numpy().flatten()], labels=["Train", "Test"])
plt.title("Boxplot of Target Values")
plt.show()
