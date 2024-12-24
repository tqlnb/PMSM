import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('input/measures_v2.csv')

# Features and target
features = ['u_q', 'u_d', 'i_d', 'i_q']
target = ['torque', 'motor_speed']

# Prepare data
X = df[features].values
y = df[target].values


# Convert to PyTorch tensors and create dataset
class MotorDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# Create dataset and dataloader
dataset = MotorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# Define the PINN Model
class PINNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PINNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Instantiate model, optimizer, and loss function
model = PINNModel(input_size=4, output_size=2)  # 4 inputs, 2 outputs (torque and motor_speed)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


# 物理约束损失函数：基于PMSM的d轴和q轴方程，以及电磁转矩方程
def physics_informed_loss(predictions, targets, inputs, R=0.01, L_d=0.005, L_q=0.006, lambda_f=0.1, p=4):
    # 神经网络的预测输出
    motor_speed_pred, torque_pred = predictions[:, 0], predictions[:, 1]
    motor_speed_true, torque_true = targets[:, 0], targets[:, 1]

    # 输入：d轴和q轴电压、电流
    u_q, u_d, i_d, i_q = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]

    # 计算 d轴 和 q轴 的电压方程
    # ω 是电机的转速（在这里使用预测的转速）
    omega = motor_speed_pred

    # d轴电压方程 (PMSM d-axis equation)
    u_d_pred = R * i_d + L_d * torch.gradient(i_d)[0] - omega * L_q * i_q

    # q轴电压方程 (PMSM q-axis equation)
    u_q_pred = R * i_q + L_q * torch.gradient(i_q)[0] + omega * L_d * i_d + omega * lambda_f

    # 电磁转矩方程 (Electromagnetic torque equation)
    torque_pred_physics = (3 / 2) * p * (lambda_f * i_q + (L_d - L_q) * i_d * i_q)

    # 基本的 MSE 损失 (MSE loss)
    mse_loss = criterion(predictions, targets)

    # 物理约束损失：电压方程和转矩方程的误差
    physics_loss = torch.mean((u_d - u_d_pred) ** 2) + torch.mean((u_q - u_q_pred) ** 2) + torch.mean(
        (torque_pred - torque_pred_physics) ** 2)

    # 总损失
    total_loss = mse_loss + physics_loss
    return total_loss


# Train the PINN Model
def train_model(model, dataloader, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = physics_informed_loss(predictions, targets, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')


# Train the model
train_model(model, dataloader, optimizer)

# Train Quantile Regression Forest (QRF) for uncertainty estimation
qrf_model = RandomForestRegressor(n_estimators=100, min_samples_split=5)


def train_qrf(qrf_model, model, dataloader):
    # Collect all inputs and predictions for training QRF
    all_inputs = []
    all_predictions = []
    all_targets = []

    model.eval()  # Switch model to evaluation mode
    with torch.no_grad():
        for inputs, targets in dataloader:
            predictions = model(inputs)
            all_inputs.append(inputs.numpy())
            all_predictions.append(predictions.numpy())
            all_targets.append(targets.numpy())

    # Stack all data for QRF training
    all_inputs = np.vstack(all_inputs)
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)

    # Train QRF using the inputs and model predictions/targets
    qrf_model.fit(all_inputs, all_targets)
    return qrf_model


# Train QRF for uncertainty estimation
qrf_model = train_qrf(qrf_model, model, dataloader)


# Prediction with uncertainty using QRF
def predict_with_uncertainty(qrf_model, inputs):
    # Predict using the trained QRF model
    preds_50 = qrf_model.predict(inputs)  # 50% quantile (median)
    preds_90 = np.percentile(qrf_model.predict(inputs), 90, axis=1)  # 90% quantile
    preds_10 = np.percentile(qrf_model.predict(inputs), 10, axis=1)  # 10% quantile

    return preds_50, preds_10, preds_90


# Test predictions
test_inputs = np.random.rand(10, 4)  # 10 random test samples
mean_preds, lower_bound, upper_bound = predict_with_uncertainty(qrf_model, test_inputs)
print("Mean Predictions:", mean_preds)
print("Uncertainty Bounds:", lower_bound, upper_bound)
