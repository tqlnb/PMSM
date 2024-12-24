import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

# 定义 PINN 模型
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PINN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

# 定义 NTK 计算函数
def compute_ntk(model, inputs):
    inputs = inputs.requires_grad_(True)
    outputs = model(inputs)
    gradients = torch.autograd.grad(
        outputs, model.parameters(),
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )

    # 检查每个梯度的维度并跳过标量
    valid_gradients = [g.view(inputs.shape[0], -1) for g in gradients if g is not None and g.ndim > 1]
    if len(valid_gradients) == 0:
        raise RuntimeError("No valid gradients computed for NTK calculation.")

    gradients_flat = torch.cat(valid_gradients, dim=1)
    ntk = gradients_flat @ gradients_flat.T
    return ntk

# 定义物理约束损失函数
def physics_loss(model, inputs, P, psi_d, psi_q, T_load, B, J):
    ud, uq, id, iq = inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3]
    outputs = model(inputs)
    torque, omega = outputs[:, 0], outputs[:, 1]

    # 电磁关系约束
    torque_pred = (3 / 2) * P * (psi_d * iq - psi_q * id)
    loss_torque = torch.mean((torque - torque_pred) ** 2)

    # 机械动力学约束
    d_omega = (torque - T_load - B * omega) / J
    loss_dynamics = torch.mean(d_omega ** 2)

    return loss_torque + loss_dynamics

# 训练函数
def train_pinn(model, optimizer, inputs, P, psi_d, psi_q, T_load, B, J, epochs=1000):
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 计算物理约束损失
        loss = physics_loss(model, inputs, P, psi_d, psi_q, T_load, B, J)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 每隔100次迭代输出一次损失和 NTK 矩阵
        if epoch % 100 == 0:
            try:
                ntk_matrix = compute_ntk(model, inputs)
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                print(f"NTK Matrix (Trace): {torch.trace(ntk_matrix).item()}")
            except RuntimeError as e:
                print(f"Epoch {epoch}, Loss: {loss.item()}, NTK Calculation Error: {e}")

# 数据加载和预处理
df = pd.read_csv('input/measures_v2.csv')
df = df.dropna()

# 特征和目标变量
features = ['u_q', 'u_d', 'i_d', 'i_q']
target = ['torque', 'motor_speed']
X = df[features].values
y = df[target].values

# 数据划分
train_X, temp_X, train_y, temp_y = train_test_split(X, y, test_size=0.5, random_state=42)
val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.4, random_state=42)

# 转换为 PyTorch 张量
train_inputs = torch.tensor(train_X, dtype=torch.float32)
train_targets = torch.tensor(train_y, dtype=torch.float32)
val_inputs = torch.tensor(val_X, dtype=torch.float32)
val_targets = torch.tensor(val_y, dtype=torch.float32)
test_inputs = torch.tensor(test_X, dtype=torch.float32)
test_targets = torch.tensor(test_y, dtype=torch.float32)

# 参数设置
P = 4  # 极对数
psi_d = 4.3 / 1000  # d 轴磁链 (由反电势常数计算得出)
psi_q = 4.3 / 1000  # q 轴磁链
T_load = 0.2  # 额定负载扭矩 (Nm)
B = 0.01  # 阻尼系数
J = 0.0028 / 10000  # 转动惯量 (Kg.m^2, 转换为标准单位)

# 初始化网络和优化器
input_dim = 4
output_dim = 2
hidden_dim = 64
model = PINN(input_dim, output_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train_pinn(model, optimizer, train_inputs, P, psi_d, psi_q, T_load, B, J, epochs=1000)

# 测试模型
def test_model(model, test_inputs, test_targets):
    model.eval()
    with torch.no_grad():
        predictions = model(test_inputs)
        loss = torch.mean((predictions - test_targets) ** 2).item()
        print(f"Test Loss: {loss}")
    return predictions

# 使用测试集评估模型
predictions = test_model(model, test_inputs, test_targets)
