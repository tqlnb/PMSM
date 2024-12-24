import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. 加载数据集
df = pd.read_csv('input/measures_v2.csv')

# 2. 数据预处理
# 检查缺失值
print(df.isnull().sum())
# 删除缺失值
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

# 6. 数据重塑
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 7. 构建CNN模型
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                 input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(2))  # 输出层有2个神经元

# 8. 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 9. 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=64,
                    validation_data=(X_test, y_test))

# 10. 可视化训练结果
# 绘制损失曲线
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# 绘制MAE曲线
plt.plot(history.history['mae'], label='训练MAE')
plt.plot(history.history['val_mae'], label='验证MAE')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.show()

# 11. 模型评估
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"测试集损失：{test_loss}, 测试集MAE：{test_mae}")

# 12. 预测与可视化
predictions = model.predict(X_test)

# 逆缩放回原始尺度
y_test_inverse = scaler_y.inverse_transform(y_test)
predictions_inverse = scaler_y.inverse_transform(predictions)

# 绘制Torque预测结果
plt.figure(figsize=(10,5))
plt.plot(y_test_inverse[:, 0], label='真实Torque值')
plt.plot(predictions_inverse[:, 0], label='预测Torque值')
plt.legend()
plt.xlabel('样本')
plt.ylabel('Torque')
plt.show()

# 绘制Motor Speed预测结果
plt.figure(figsize=(10,5))
plt.plot(y_test_inverse[:, 1], label='真实Motor Speed值')
plt.plot(predictions_inverse[:, 1], label='预测Motor Speed值')
plt.legend()
plt.xlabel('样本')
plt.ylabel('Motor Speed')
plt.show()
