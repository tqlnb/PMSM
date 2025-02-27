import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('data/merged.csv')

# Features and target
features = ['iq', 'id', 'vd', 'vq']
target = ['torque', 'speed']

# Prepare data 准备数据
X = df[features].values
y = df[target].values

# Normalize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Define a simple feedforward neural network
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model, loss function, and optimizer
model = PINN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Inverse transform for original scale
y_pred = scaler_y.inverse_transform(test_outputs.numpy())
y_true = scaler_y.inverse_transform(y_test_tensor.numpy())

# You can add more evaluations, such as computing R^2 score or visualizing results
from sklearn.metrics import mean_squared_error, r2_score

# Compute MSE and R2 score
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'Test Mean Squared Error: {mse:.4f}')
print(f'Test R^2 Score: {r2:.4f}')

# Compute residuals
residuals = np.abs(y_true - y_pred)

import matplotlib.pyplot as plt

# Plot residuals
plt.figure(figsize=(12, 6))
plt.hist(residuals.flatten(), bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(y_train.flatten(), bins=30, alpha=0.7, label='Train')
plt.hist(y_test.flatten(), bins=30, alpha=0.7, label='Test')
plt.legend()
plt.title("Distribution of Target Values")
plt.subplot(1, 2, 2)
plt.boxplot([y_train.flatten(), y_test.flatten()], labels=["Train", "Test"])
plt.title("Boxplot of Target Values")
plt.show()
