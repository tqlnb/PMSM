import matplotlib.pyplot as plt
import pandas as pd

# 数据文件路径
INPUT_FILE = "input/processed_data.csv"

# 读取数据
data = pd.read_csv(INPUT_FILE)

# 转换时间戳为相对时间（秒）
data['time'] = (data['time'] - data['time'][0]) / 1000.0

# 绘制图表
plt.figure(figsize=(12, 8))

# 绘制各变量趋势
plt.plot(data['time'], data['id'], label='Id', marker='o')
plt.plot(data['time'], data['iq'], label='Iq', marker='o')
plt.plot(data['time'], data['vd'], label='Vd', marker='o')
plt.plot(data['time'], data['vq'], label='Vq', marker='o')

# 添加图例和标签
plt.title('Data Trends Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()
