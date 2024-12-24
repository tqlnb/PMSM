import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # 添加用于保存预处理器

# 加载数据集
df = pd.read_csv('input/measures_v2.csv')

# 特征和目标
features = ['u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth', 'motor_speed',
            'i_d', 'i_q', 'pm', 'stator_yoke', 'ambient']
target = 'torque'

# 准备数据
X = df[features].values
y = df[target].values

# 数据归一化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义神经网络模型
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(len(features), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = PINN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 物理常数（需要根据电机参数设定）
K_t = 0.1  # 示例值

# 训练循环
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        mse_loss = criterion(outputs.squeeze(), batch_y)

        # 物理损失
        i_q = batch_X[:, features.index('i_q')]
        T_physics = K_t * i_q
        p_loss = torch.mean((outputs.squeeze() - T_physics) ** 2)

        # 总损失
        loss = mse_loss + p_loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

# ========== 新增：保存模型和预处理器 ==========
# 保存模型参数
torch.save(model.state_dict(), 'pinn_model.pth')
print("模型已保存为 'pinn_model.pth'")

# 保存预处理器
joblib.dump(scaler_X, 'scaler_X.save')
joblib.dump(scaler_y, 'scaler_y.save')
print("预处理器已保存为 'scaler_X.save' 和 'scaler_y.save'")
# ============================================

# 模型评估
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

# 逆归一化
y_pred = scaler_y.inverse_transform(test_outputs.numpy())
y_true = scaler_y.inverse_transform(y_test_tensor.numpy().reshape(-1, 1))

# 计算MSE和R^2得分
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'Test Mean Squared Error: {mse:.6f}')
print(f'Test R^2 Score: {r2:.6f}')

# 绘制残差分布
residuals = np.abs(y_true - y_pred)
plt.figure(figsize=(12, 6))
plt.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residuals Distribution')
plt.show()
