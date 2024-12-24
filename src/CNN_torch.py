import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

# 1. 加载数据集
df = pd.read_csv('input/measures_v2.csv')

# 2. 数据预处理
df = df.dropna()

# 3. 特征和目标变量
features = ['u_q', 'u_d', 'i_d', 'i_q']
target = ['torque', 'motor_speed']

X = df[features].values
y = df[target].values

# 4. 数据标准化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# 5. 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42)

# 6. 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 7. 重塑输入数据
X_train = X_train.unsqueeze(1)
X_test = X_test.unsqueeze(1)

# 8. 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# 9. 定义CNN模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=2)

        # 计算卷积后的序列长度
        seq_len = X_train.shape[2]
        L1 = seq_len - 2 + 1
        L2 = L1 - 2 + 1
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * L2, 50)
        self.fc2 = nn.Linear(50, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 10. 初始化模型、损失函数和优化器
model = CNNModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 11. 训练模型
num_epochs = 50
train_losses = []
val_losses = []

# early_stop_patience = 5
# best_val_loss = float('inf')
# epochs_no_improve = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # 验证集损失
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_running_loss += loss.item() * X_batch.size(0)

    val_loss = val_running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], 训练损失: {epoch_loss:.8f}, 验证损失: {val_loss:.8f}")

    # if val_loss < best_val_loss:
    #     best_val_loss = val_loss
    #     epochs_no_improve = 0
    #     # 保存最佳模型
    #     torch.save(model.state_dict(), 'best_model.pt')
    # else:
    #     epochs_no_improve += 1
    #     if epochs_no_improve >= early_stop_patience:
    #         print("Early stopping!")
    #         break


# 12. 可视化训练和验证损失
plt.plot(train_losses, label='train_loss')
plt.plot(val_losses, label='simp_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 13. 模型评估
model.eval()
test_loss = 0.0
all_predictions = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item() * X_batch.size(0)

        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())

test_loss = test_loss / len(test_loader.dataset)
print(f"测试集损失：{test_loss:.4f}")

# 14. 预测与可视化
all_predictions = np.vstack(all_predictions)
all_targets = np.vstack(all_targets)

# 逆缩放回原始尺度
predictions_inverse = scaler_y.inverse_transform(all_predictions)
targets_inverse = scaler_y.inverse_transform(all_targets)

# 绘制Torque预测结果
plt.figure(figsize=(10, 5))
plt.plot(targets_inverse[:, 0], label='reql_Torque')
plt.plot(predictions_inverse[:, 0], label='predict_Torque')
plt.legend()
plt.xlabel('label')
plt.ylabel('Torque')
plt.show()

# 绘制Motor Speed预测结果
plt.figure(figsize=(10, 5))
plt.plot(targets_inverse[:, 1], label='real_Motor Speed')
plt.plot(predictions_inverse[:, 1], label='predict_Motor Speed')
plt.legend()
plt.xlabel('sample')
plt.ylabel('Motor Speed')
plt.show()
