import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np

# 加载数据集
df = pd.read_csv('input/measures_v2.csv')

# 特征和目标变量
features = ['u_q', 'u_d', 'i_d', 'i_q']
target = ['torque', 'motor_speed']

# 准备数据
X = df[features].values
y = df[target].values

# 如果数据中没有时间信息，需要添加时间变量
# 假设数据按时间顺序排列，时间间隔为1单位
time_steps = np.arange(len(df))
X = np.hstack((X, time_steps.reshape(-1, 1)))  # 将时间作为特征添加

# 更新特征列表
features.append('time')


# 转换为PyTorch张量并创建数据集
class MotorDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


# 创建数据集和数据加载器
dataset = MotorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)  # 保持时间顺序


# 定义简化的PINN模型
class PINNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PINNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)  # 将神经元数量从64减少到32
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型、优化器和损失函数
model = PINNModel(input_size=5, output_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()



# 物理约束损失函数
def physics_informed_loss(predictions, targets, inputs, model, criterion, R=0.01, L_d=0.005, L_q=0.006, lambda_f=0.1,
                          p=4):
    # 预测的电机速度和转矩
    motor_speed_pred, torque_pred = predictions[:, 0], predictions[:, 1]
    motor_speed_true, torque_true = targets[:, 0], targets[:, 1]

    # 输入的电压、电流和时间
    u_q, u_d, i_d, i_q, t = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3], inputs[:, 4]

    # 需要对电流关于时间求导数，使用自动微分
    i_d.requires_grad_(True)
    i_q.requires_grad_(True)

    # 前向传播计算模型输出
    predictions = model(inputs)
    motor_speed_pred, torque_pred = predictions[:, 0], predictions[:, 1]

    # 计算电流对时间的导数
    di_d_dt = torch.autograd.grad(i_d, t, grad_outputs=torch.ones_like(i_d), create_graph=True)[0]
    di_q_dt = torch.autograd.grad(i_q, t, grad_outputs=torch.ones_like(i_q), create_graph=True)[0]

    # d轴和q轴电压方程
    omega = motor_speed_pred
    u_d_pred = R * i_d + L_d * di_d_dt - omega * L_q * i_q
    u_q_pred = R * i_q + L_q * di_q_dt + omega * L_d * i_d + omega * lambda_f

    # 电磁转矩方程
    torque_pred_physics = (3 / 2) * p * (lambda_f * i_q + (L_d - L_q) * i_d * i_q)

    # 计算损失
    mse_loss = criterion(predictions, targets)
    physics_loss = criterion(u_d, u_d_pred) + criterion(u_q, u_q_pred) + criterion(torque_pred, torque_pred_physics)

    # 总损失
    total_loss = mse_loss + physics_loss
    return total_loss


# 训练PINN模型
def train_model(model, dataloader, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs.requires_grad_(True)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = physics_informed_loss(predictions, targets, inputs, model, criterion)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')


# 开始训练
train_model(model, dataloader, optimizer)

# 使用支持分位数回归的模型进行不确定性估计
# 使用 GradientBoostingRegressor，并设置 loss='quantile'
from sklearn.multioutput import MultiOutputRegressor


def train_quantile_regressor(X, y):
    quantiles = [0.1, 0.5, 0.9]  # 需要预测的分位数
    models = {}
    for q in quantiles:
        # 为每个分位数训练一个模型
        gbr = GradientBoostingRegressor(loss='quantile', alpha=q, n_estimators=100)
        # 如果是多输出，需要使用 MultiOutputRegressor
        multi_gbr = MultiOutputRegressor(gbr)
        multi_gbr.fit(X, y)
        models[q] = multi_gbr
    return models


# 准备用于回归的不确定性数据
def prepare_uncertainty_data(model, dataloader):
    model.eval()
    all_inputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            all_inputs.append(inputs.numpy())
            all_targets.append(targets.numpy())
    all_inputs = np.vstack(all_inputs)
    all_targets = np.vstack(all_targets)
    return all_inputs, all_targets


# 训练分位数回归模型
X_uncertainty, y_uncertainty = prepare_uncertainty_data(model, dataloader)
quantile_models = train_quantile_regressor(X_uncertainty, y_uncertainty)


# 使用分位数模型进行预测
def predict_with_uncertainty(quantile_models, inputs):
    preds_10 = quantile_models[0.1].predict(inputs)
    preds_50 = quantile_models[0.5].predict(inputs)
    preds_90 = quantile_models[0.9].predict(inputs)
    return preds_50, preds_10, preds_90


# 测试预测
test_inputs = np.random.rand(10, 4)
# 添加时间特征
test_time_steps = np.arange(len(test_inputs))
test_inputs = np.hstack((test_inputs, test_time_steps.reshape(-1, 1)))

mean_preds, lower_bound, upper_bound = predict_with_uncertainty(quantile_models, test_inputs)
print("Mean Predictions:", mean_preds)
print("Uncertainty Bounds Lower:", lower_bound)
print("Uncertainty Bounds Upper:", upper_bound)
