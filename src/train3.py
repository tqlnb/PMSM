import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

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
K_t = 0.059      # 转矩常数 (Nm/A)
J = 0.08 * 1e-4  # 转动惯量 (kg·m²) —— 0.08 kg·cm² 转换为 kg·m²
lambda_phys = 1e-3  # 物理损失权重

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

    # 速度动态方程残差：T_pred = T_load（稳态条件下）
    res_speed = torque_pred - T_load

    # 物理损失
    phys_loss = torch.mean(res_torque**2) + torch.mean(res_speed**2)  # 速度动态方程残差
    # phys_loss = torch.mean(res_torque ** 2)  # 仅考虑扭矩方程残差

    # 总损失
    total_loss = data_loss + lambda_phys * phys_loss

    # 反向传播和优化
    total_loss.backward()
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

    # 反归一化预测值和真实值
    y_pred = scaler_y.inverse_transform(test_outputs.numpy())
    y_true = scaler_y.inverse_transform(y_test_tensor.numpy())

    # 计算MSE和R²分数
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f'Test Mean Squared Error: {mse:.6f}')
    print(f'Test R^2 Score: {r2:.6f}')

    # 计算残差
    residuals = np.abs(y_true - y_pred)

    # 绘制残差分布图
    plt.figure(figsize=(12, 6))
    plt.hist(residuals.flatten(), bins=50, edgecolor='k', alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.show()
