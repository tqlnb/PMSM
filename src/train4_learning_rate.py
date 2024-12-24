import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# 启用异常检测
torch.autograd.set_detect_anomaly(True)

# 加载数据集
df = pd.read_csv('data/merged.csv')

# 特征和目标
features = ['iq', 'id', 'vd', 'vq']
target = ['torque', 'speed']

# 准备数据
X = df[features].values
y = df[target].values

# 数据归一化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# 定义PINN模型
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


# 初始化模型、损失函数和优化器
model = PINN()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 电机物理参数（根据提供的数据）
K_t = 0.059  # 转矩常数 (Nm/A)
J = 0.08 * 1e-4  # 转动惯量 (kg·m²) —— 0.08 kg·cm² 转换为 kg·m²
lambda_phys = 1e-3  # 物理损失权重
alpha = 0.9  # 超参数，控制物理损失权重的更新速率

# 初始化物理损失权重
lambda_phys_curr = lambda_phys

# 训练循环
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(X_train_tensor)
    torque_pred, speed_pred = outputs[:, 0], outputs[:, 1]

    # 数据损失（MSE）
    data_loss = mse_loss(outputs, y_train_tensor)

    # 提取输入中的电流
    iq = X_train_tensor[:, 0]

    # 提取真实的负载扭矩 (假设 'torque' 是负载扭矩)
    T_load = y_train_tensor[:, 0]

    # 扭矩方程残差：T_pred = K_t * i_q
    res_torque = torque_pred - (K_t * iq)

    # 物理损失
    phys_loss = torch.mean(res_torque ** 2)  # 仅考虑扭矩方程残差

    # 计算物理损失的梯度
    phys_loss_grad = torch.autograd.grad(phys_loss, model.parameters(), create_graph=True)
    phys_loss_grad_flat = torch.cat([g.flatten() for g in phys_loss_grad])
    grad_phys_max = torch.max(torch.abs(phys_loss_grad_flat))

    # 计算数据损失的梯度
    data_loss_grad = torch.autograd.grad(data_loss, model.parameters(), create_graph=True)
    data_loss_grad_flat = torch.cat([g.flatten() for g in data_loss_grad])
    grad_data_mean = torch.mean(torch.abs(data_loss_grad_flat))

    # 计算物理损失权重
    lambda_phys_new = grad_phys_max / grad_data_mean

    # 更新物理损失权重
    lambda_phys_curr = (1 - alpha) * lambda_phys_curr + alpha * lambda_phys_new

    # 总损失
    total_loss = data_loss + lambda_phys_curr * phys_loss

    # 反向传播和优化
    total_loss.backward(retain_graph=True)
    optimizer.step()

    # 每10个epoch打印一次损失
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Data Loss: {data_loss.item():.6f}, '
              f'Physical Loss: {phys_loss.item():.6f}, Total Loss: {total_loss.item():.6f}')

# 评估模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = mse_loss(test_outputs, y_test_tensor)
    print(f'\nTest Loss: {test_loss.item():.6f}')
