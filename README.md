# 收集数据

## 实验设备
使用STM32F103zeT6开发板，ST Motor Control Workbench5.4.4生成控制代码。设置如下

![image](https://github.com/user-attachments/assets/08bec522-4143-43ff-b24e-d2ae0bd4d08b)

![image](https://github.com/user-attachments/assets/153a7374-4448-45fd-8eb1-777de85c50ee)

![image](https://github.com/user-attachments/assets/84c8595d-dde3-4558-af81-c78906c7c023)

![image](https://github.com/user-attachments/assets/f16f6fbc-3e02-487f-9d1f-38231389ba34)

![image](https://github.com/user-attachments/assets/3f6a69eb-82d3-4c33-8ad1-91628c09e893)

![image](https://github.com/user-attachments/assets/ae368ec2-7aed-4c2c-af9a-ca8e48e9d180)

![image](https://github.com/user-attachments/assets/40a3bbf2-836e-4b66-b949-dc2e9c304737)

![image](https://github.com/user-attachments/assets/e712c63f-9519-4a45-aab2-ef75543f357f)

![image](https://github.com/user-attachments/assets/542b925e-afb2-42a8-8527-cbb0b3dba326)

![image](https://github.com/user-attachments/assets/8e8eb12b-c2a6-4208-b6d6-a2fffcda944c)

![image](https://github.com/user-attachments/assets/7fe9bf21-ec6b-4e85-9e9b-af93a057b4cc)

![image](https://github.com/user-attachments/assets/514bdded-1136-4ade-a60b-ead7902d3315)

![image](https://github.com/user-attachments/assets/a50f9118-5142-47a6-ba26-f317753814ee)

![image](https://github.com/user-attachments/assets/923f8dd5-3ece-4b8e-87a6-ce8aef01ca02)

![image](https://github.com/user-attachments/assets/62ce0629-28df-415e-b3a8-f3ca15e7eb48)

![image](https://github.com/user-attachments/assets/1163d469-efe9-496b-a339-6b07f149086e)

![image](https://github.com/user-attachments/assets/bcd76320-a1fe-4040-ab3b-4f856445d42d)

![image](https://github.com/user-attachments/assets/7c1ee390-3220-42e1-aa50-38ecda73dd55)

生成代码然后修改main.c方法加上串口数据传输（id,iq,ud,uq）

```c
/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_TIM1_Init();
  MX_TIM4_Init();
  MX_USART1_UART_Init();
  MX_MotorControl_Init();

  /* Initialize interrupts */
  MX_NVIC_Init();
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  // 获取FOCVars_t结构体
  FOCVars_t* vars = GetFOCVars(0);
  char buffer[64];
  while (1)
  {
    /* USER CODE END WHILE */
    // FOCVars_t* vars = GetFOCVars(0);
    // HAL_UART_Transmit(&huart1, (uint8_t*)"Hello World!\r\n", 14, 1000); 
    // 获取电流值
    int16_t current_id = vars->Iqd.d;
    int16_t current_iq = vars->Iqd.q;
    //获取电压值
    int16_t voltage_d = vars->Vqd.d;
    int16_t voltage_q = vars->Vqd.q;
    // 发送
    sprintf(buffer, "id:%d,iq:%d,vd:%d,vq:%d\r\n", current_id, current_iq, voltage_d, voltage_q);
    HAL_UART_Transmit(&huart1, buffer, strlen(buffer), 1000); 
    HAL_Delay(20);
    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

```

## id,iq,ud,uq来源
在电机控制中，\(i_{uvw}\)、\(u_{uvw}\) 是指定子三相电流和电压，\(i_{d}\)、\(i_{q}\)、\(u_{d}\)、\(u_{q}\) 是在旋转坐标系（dq坐标系）中的电流和电压分量。将三相坐标系 (\(uvw\)) 转换到两相旋转坐标系 (\(dq\))，需要进行坐标变换，这包括 **Clark变换** 和 **Park变换**。

以下是主要步骤：

---

1. **从三相坐标系到两相静止坐标系 - Clark变换**
Clark变换将三相信号投影到一个静止的两相平面上。假设三相电流为 \(i_u\), \(i_v\), \(i_w\)，我们有：

![image](https://github.com/user-attachments/assets/2f6e830d-c8ad-49e4-8e78-9b89a562e11e)

若三相电流满足平衡（\(i_u + i_v + i_w = 0\)），上述公式可以简化为：

![image](https://github.com/user-attachments/assets/d98e8954-3d83-4708-ad84-9b6db78f7e88)


同理适用于电压 (\(u_u, u_v, u_w\))。


2. **从静止坐标系到旋转坐标系（\(dq\)） - Park变换**
Park变换将 \(\alpha\beta\) 坐标系中的量投影到旋转坐标系 \(dq\)。旋转坐标系与电机转子的旋转位置同步，使用转子磁链位置角 \(\theta\) 进行变换：


![image](https://github.com/user-attachments/assets/2b7db792-fb77-4dba-9257-699ce3cbef83)


其中，\(\theta\) 通常由电机控制器中的位置传感器（如编码器）或观测器提供。


3. **总结**
- Clark变换：将三相信号投影到静止的 \(\alpha\beta\) 平面；
- Park变换：将静止的 \(\alpha\beta\) 信号投影到旋转的 \(dq\) 坐标系；
- \(i_d\)、\(i_q\) 分别对应直轴（d轴）和交轴（q轴）的电流分量，通常 \(i_d\) 用于控制磁链，\(i_q\) 用于控制电磁转矩。

完整流程为：

![image](https://github.com/user-attachments/assets/7bfcceab-e41b-4f52-94ce-c144ff5139be)

## 在电机运行时使用java程序收集串口数据


1.采集串口数据
```java
package tql.test.test32;

import jssc.SerialPort;
import jssc.SerialPortEvent;
import jssc.SerialPortEventListener;
import jssc.SerialPortException;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class SerialDataLogger {
    private static final String PORT_NAME = "COM5"; // 修改为实际串口名称
    private static final int BAUD_RATE = 115200;   // 根据实际设置波特率
    private static final String OUTPUT_FILE = "data_log.txt";
    private static BufferedWriter writer;

    public static void main(String[] args) throws IOException {
        SerialPort serialPort = new SerialPort(PORT_NAME);
        try {
            // 打开串口
            serialPort.openPort();
            serialPort.setParams(
                BAUD_RATE,
                SerialPort.DATABITS_8,
                SerialPort.STOPBITS_1,
                SerialPort.PARITY_NONE
            );

            // 添加串口监听器
            serialPort.addEventListener(new SerialPortEventListener() {

                {
                    try {
                        writer = new BufferedWriter(new FileWriter(OUTPUT_FILE, true));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void serialEvent(SerialPortEvent event) {
                    if (event.isRXCHAR() && event.getEventValue() > 0) {
                        try {

                            String data = serialPort.readString(event.getEventValue());
                            //System.out.println(data);
                            if(data.startsWith("id")) {
                                if(!data.endsWith("\n")) {
                                    data = data + "\n";
                                }
                                String s = System.currentTimeMillis() + ",Received:" + data;
                                System.out.print(s);

                                if (writer != null) {
                                    writer.write(s);
                                    writer.flush();
                                }
                            }
                        } catch (SerialPortException | IOException e) {
                            e.printStackTrace();
                        }
                    }
                }
            });
            while (true) {
                Thread.sleep(1000);
            }
        } catch (SerialPortException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        } finally {
            if (writer != null) {
                writer.close();
            }
        }
    }
}

```

2.处理数据，放到csv文件

```java
package tql.test.test32;

import java.io.*;
import java.util.Date;
import java.util.regex.*;

public class DataProcessor {
    private static final String INPUT_FILE = "data_log.txt";  // 原始数据文件路径
    private static final String OUTPUT_FILE = "id_iq_vd_vq.csv"; // 输出文件路径

    public static void main(String[] args) {
        try (BufferedReader reader = new BufferedReader(new FileReader(INPUT_FILE));
             BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_FILE))) {
            
            // 写入表头
            writer.write("time,id,iq,vd,vq");
            writer.newLine();
            int idValue = 0;
            int iqValue = 0;
            int vdValue = 0;
            int vqValue = 0;
            float id_real = 0;
            float iq_real = 0;
            float vd_real = 0;
            float vq_real = 0;
            Date date ;


            String line;
            while ((line = reader.readLine()) != null) {
                // 匹配时间戳和数据
                Pattern pattern = Pattern.compile("(-?\\d+),Received:id:(\\d+),iq:(-?\\d+),vd:(-?\\d+), vq:(-?\\d+)");
                Matcher matcher = pattern.matcher(line);

                if (matcher.find()) {
                    // 提取数据
                    String time = matcher.group(1);
                    String id = matcher.group(2);
                    String iq = matcher.group(3);
                    String vd = matcher.group(4);
                    String vq = matcher.group(5);
                    // 对数据进行处理
                    date = new Date(Long.parseLong(time));
                    idValue = Integer.parseInt(id);
                    iqValue = Integer.parseInt(iq);
                    vdValue = Integer.parseInt(vd);
                    vqValue = Integer.parseInt(vq);

                    id_real = convertToRealValue(idValue, 60);
                    iq_real = convertToRealValue(iqValue, 60);
                    vd_real = convertToRealValue(vdValue, 60);
                    vq_real = convertToRealValue(vqValue, 60);

                    // 写入 CSV 格式
                    //writer.write(String.join(",", time, id, iq, vd, vq));
                    java.text.SimpleDateFormat dateFormat = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                    writer.write(String.join(",", dateFormat.format(date), String.valueOf(id_real), String.valueOf(iq_real), String.valueOf(vd_real), String.valueOf(vq_real)));
                    //writer.write(String.join(",", time, String.valueOf(id_real), String.valueOf(iq_real), String.valueOf(vd_real), String.valueOf(vq_real)));
                    writer.newLine();
                }
            }
            
            System.out.println("Data processing complete. Output saved to: " + OUTPUT_FILE);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static float convertToRealValue(int value, float maxValue) {
        return value / 32767.0f * maxValue;
    }
}
```

3.扭矩传感器的数据，.xls文件（其实是文本文件），处理为.csv文件

```java
package tql.test.test32;

import java.io.*;
import java.text.ParseException;
import java.text.SimpleDateFormat;

public class TextToCSV {
    public static void main(String[] args) {
        // 输入文件路径
        String inputFilePath = "myexcel.txt";
        // 输出 CSV 文件路径
        String outputFilePath = "torque_speed_time.csv";

        try (BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {

            // 写入 CSV 文件表头
            writer.write("torque,speed,time\n");

            String line;
            SimpleDateFormat inputFormat = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
            SimpleDateFormat outputFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");

            while ((line = reader.readLine()) != null) {
                // 跳过空行
                if (line.trim().isEmpty()) continue;

                // 将制表符替换为逗号
                String[] parts = line.split("\t");
                if (parts.length == 3) {
                    String torque = parts[0];
                    String speed = parts[1];
                    String time = parts[2];

                    try {
                        // 转换时间格式
                        time = outputFormat.format(inputFormat.parse(time));
                    } catch (ParseException e) {
                        System.err.println("时间格式解析错误: " + time);
                    }

                    String csvLine = torque + "," + speed + "," + time;
                    writer.write(csvLine + "\n");
                }
            }

            System.out.println("CSV 文件已生成: " + outputFilePath);

        } catch (IOException e) {
            System.err.println("文件读写错误: " + e.getMessage());
        }
    }
}
```

4.汇总数据，使用python

```python
import pandas as pd

# 读取两个CSV文件
high_freq_df = pd.read_csv('data/id_iq_vd_vq.csv')  
low_freq_df = pd.read_csv('data/torque_speed_time.csv')  

# 设置时间列为索引
high_freq_df.set_index('time', inplace=True)
low_freq_df.set_index('time', inplace=True)

# 将时间列转换为 datetime 类型
high_freq_df.index = pd.to_datetime(high_freq_df.index)
low_freq_df.index = pd.to_datetime(low_freq_df.index)

# 处理重复的时间戳，通过添加毫秒来区分
high_freq_df['unique_time'] = high_freq_df.index
low_freq_df['unique_time'] = low_freq_df.index

# 为重复的时间戳添加毫秒信息
def add_milliseconds_to_duplicates(df):
    df['time_ms'] = df.groupby(df['unique_time']).cumcount()  # 计算重复时间戳的序号
    df['unique_time'] = df['unique_time'].dt.strftime('%Y-%m-%d %H:%M:%S') + '.' + df['time_ms'].astype(str).str.zfill(3)  # 将毫秒加到时间戳中
    df.drop(columns=['time_ms'], inplace=True)
    return df

high_freq_df = add_milliseconds_to_duplicates(high_freq_df)
low_freq_df = add_milliseconds_to_duplicates(low_freq_df)

# 设置新的唯一时间列为索引
high_freq_df.set_index('unique_time', inplace=True)
low_freq_df.set_index('unique_time', inplace=True)

# 找到共同的时间范围
common_index = high_freq_df.index.intersection(low_freq_df.index)

# 对低频数据进行线性插值
low_freq_df_interp = low_freq_df.reindex(high_freq_df.index).interpolate(method='linear')

# 合并数据
merged_df = pd.concat([high_freq_df, low_freq_df_interp], axis=1)

# 保存合并后的数据
merged_df.to_csv('data/merged.csv')  # 保存合并后的数据到文件
```

## 数据图像

**网上的标准数据**


![image](https://github.com/user-attachments/assets/a68de9c4-76f3-43c1-b339-afbafed0d674)

**我的数据**


![image](https://github.com/user-attachments/assets/e5aa61bb-fefa-4441-8245-3e61f61a14e8)


![image](https://github.com/user-attachments/assets/54886d8f-3c1b-4138-a088-e86622e407f5)


![image](https://github.com/user-attachments/assets/47f666d0-4167-4843-a064-95c7e00dcfab)


![image](https://github.com/user-attachments/assets/0e9ce2b5-a69a-4711-9bca-4012370951df)


![image](https://github.com/user-attachments/assets/c9d62080-a222-454a-a478-2e3af19a8e55)


# 训练的尝试
## 1.使用两层全连接层，输入为（id,iq,vd,vq）,输出为(speed,torque)

```python
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

```

输出

```

Epoch [10/1000], Loss: 0.9219
Epoch [20/1000], Loss: 0.7748
Epoch [30/1000], Loss: 0.6301
...
Epoch [510/1000], Loss: 0.0953
Epoch [520/1000], Loss: 0.0947
Epoch [530/1000], Loss: 0.0941
...
Epoch [980/1000], Loss: 0.0772
Epoch [990/1000], Loss: 0.0769
Epoch [1000/1000], Loss: 0.0766
Test Loss: 0.2169
Test Mean Squared Error: 3613.3276
Test R^2 Score: 0.4504


----------------------分界线，上面是我的数据，下面是网上的数据

Epoch [10/100], Loss: 0.6394
Epoch [20/100], Loss: 0.3311
Epoch [30/100], Loss: 0.1319
Epoch [40/100], Loss: 0.0442
Epoch [50/100], Loss: 0.0216
Epoch [60/100], Loss: 0.0120
Epoch [70/100], Loss: 0.0089
Epoch [80/100], Loss: 0.0063
Epoch [90/100], Loss: 0.0053
Epoch [100/100], Loss: 0.0047
Test Loss: 0.0046
Test Mean Squared Error: 27.6074
Test R^2 Score: 0.9954

```

网上的
![image](https://github.com/user-attachments/assets/332a2356-bcac-4ca0-8bbd-d88232852519)

![image](https://github.com/user-attachments/assets/d8c5130b-ad98-4f3a-b7f8-d3b5cb970c68)

我的
![image](https://github.com/user-attachments/assets/79614e70-1dee-4d60-8f0f-9a33f4bdcd27)

![image](https://github.com/user-attachments/assets/04112950-438d-43cb-925f-ee67e35dd4b9)

### 分析结果
从这些测试结果来看，你的模型在测试集上的性能可以从以下几个方面分析：

1. **Test Loss (测试损失)**: 
   - **值：0.2117**  
   这是模型的损失函数值，具体取决于你所使用的损失函数类型（例如均方误差 MSE 或交叉熵）。损失越小，表明模型的预测值与真实值之间的误差越小。当前损失值并不算高，但还需要结合目标任务和数据范围进行判断。

2. **Test Mean Squared Error (测试均方误差)**:
   - **值：3579.9895**  
   均方误差反映了模型预测值与真实值的平均偏差的平方。较高的均方误差（如这里的3579.99）表明预测误差较大。这个值需要结合你的目标变量的数值范围进行判断，例如，如果目标变量值通常在数千或更高的范围内，那么这个误差可能还能接受；如果范围较小，则说明模型需要改进。

3. **Test R² Score (测试 R² 分数)**:
   - **值：0.4591**  
   R² 分数是衡量模型拟合优度的指标，取值范围为 -∞ 到 1：
     - **1** 表示模型完美拟合数据。
     - **0** 表示模型预测能力与简单的均值预测一样差。
     - **负值** 表示模型甚至比均值预测更差。
     
   当前 R² 值为 0.4591，表明模型能够解释 45.91% 的数据方差，说明模型有一定的预测能力，但性能尚未达到理想水平。

## 使用简单的PINN

将扭矩方程和速度方程加入损失函数

![image](https://github.com/user-attachments/assets/89719b3c-bea5-4195-bd33-e2df692be7a5)

```python
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
```

```
Epoch [950/1000], Data Loss: 0.077357, Physical Loss: 1.089441, Total Loss: 0.078446
Epoch [960/1000], Data Loss: 0.077187, Physical Loss: 1.090424, Total Loss: 0.078278
Epoch [970/1000], Data Loss: 0.076989, Physical Loss: 1.090475, Total Loss: 0.078080
Epoch [980/1000], Data Loss: 0.076832, Physical Loss: 1.087985, Total Loss: 0.077920
Epoch [990/1000], Data Loss: 0.076586, Physical Loss: 1.089882, Total Loss: 0.077676
Epoch [1000/1000], Data Loss: 0.076360, Physical Loss: 1.089078, Total Loss: 0.077449

Test Loss: 0.215963
Test Mean Squared Error: 3609.480957
Test R^2 Score: 0.451778

```

训练效果不好，没有原来的好
分析：根据公式，扭矩应该与iq正相关，而iq一直上下波动，扭矩变化不大，导致物理损失有问题，考虑将前一段时间的数据纳入统计

## 分析时间相关神经网络
如果神经网络的输出和时间有关，说明你的问题涉及时间序列或动态变化的特性。为了捕捉时间相关性，需要设计能够处理时间依赖的网络结构，例如：


### 1. **循环神经网络（RNN, LSTM, GRU）**
   - **RNN（Recurrent Neural Networks）**: 用于捕获时间序列中的依赖关系，其输出会基于当前输入和之前的隐藏状态。适用于短期依赖问题。
   - **LSTM（Long Short-Term Memory）**: 是一种改进的 RNN，能够捕获长期依赖关系，适用于更复杂的时间序列任务。
   - **GRU（Gated Recurrent Unit）**: 是 LSTM 的一种简化版本，计算效率更高，效果和 LSTM 接近。

   **优点**: 适合处理连续时间序列数据。
   **缺点**: 随着序列长度增加，可能出现梯度消失或爆炸问题。


### 2. **一维卷积神经网络（1D-CNN）**
   - 通过卷积操作处理时间序列数据，提取局部的时间相关模式。
   - 通常适合用于固定长度的时间窗口分析，计算效率高。

   **优点**: 比 RNN 的训练速度快，适合平行化处理。
   **缺点**: 对长时间依赖建模能力较弱。

### 3. **Transformer**
   - **基于注意力机制**，能够捕捉序列中的全局依赖关系，无需像 RNN 那样逐步处理。
   - **时间序列 Transformer（Time-Series Transformer）**: 特别适用于长时间序列预测。
   - **优点**: 计算效率高，尤其适合长序列。
   - **缺点**: 对小规模数据可能表现不佳。


### 4. **物理信息嵌入（Physics-Informed Neural Networks, PINNs）**
   - 如果你的时间序列数据具有物理背景，可以通过 PINNs 将物理微分方程（如时间相关的动力学方程）融入神经网络。
   - PINNs 会直接在网络中嵌入时间变量 \( t \)，并约束输出满足物理规律。
   - **优点**: 能够提高模型的泛化性能，特别是对未见数据或边界条件的预测。
   - **缺点**: 需要明确的物理规律作为先验信息。


### 时间相关性建模的输入设计
1. **显式加入时间变量**:
   - 直接将时间 \( t \) 作为网络的一个输入维度，模型的输出 \( y \) 依赖于 \( t \)：  
     \[
     y = f(x, t)
     \]  

2. **时间窗口滑动**:
   - 提取固定长度的时间窗口数据作为输入，例如 ( [x_{t-3}, x_{t-2}, x_{t-1}, x_t] ） 预测 ( x_{t+1} )。

3. **时间序列嵌入**:
   - 将时间序列数据通过特征提取网络（如 LSTM 或 Transformer 编码器）转换为隐含特征，再进行后续预测。


## 使用CNN1D（一维卷积神经网络）

输入时间窗口滑动，试着优化模型

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv('data/merged.csv')

# Step 2: Generate time step feature (assuming data is sorted by 'unique_time')
df['time_step'] = np.arange(len(df)) * 0.1  # 每一步 0.1s 的固定时间间隔

# Features and targets
features = ['iq', 'id', 'vd', 'vq', 'time_step']
target = ['torque', 'speed']

# Step 3: Normalize the features and targets
X = df[features].values
y = df[target].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y)

# Step 4: Sliding window for time series data
def create_sliding_window(data, target, window_size):
    """
    Creates sliding window data for time series modeling.

    Args:
        data (np.array): Feature data.
        target (np.array): Target data.
        window_size (int): Number of time steps in the window.

    Returns:
        X, y: Features and targets for the model.
    """
    X_window, y_window = [], []
    for i in range(len(data) - window_size):
        X_window.append(data[i:i + window_size])  # Use the past `window_size` steps
        y_window.append(target[i + window_size])  # Predict the next time step's target
    return np.array(X_window), np.array(y_window)

# Parameters
window_size = 10  # Use the past 10 time steps

# Prepare data
X_window, y_window = create_sliding_window(X, y, window_size)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.2, random_state=42)

# Reshape for 1D-CNN (samples, channels, sequence_length)
X_train = X_train.transpose(0, 2, 1)  # Shape: (samples, channels, sequence_length)
X_test = X_test.transpose(0, 2, 1)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 5: Define the 1D-CNN model
class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=len(features), out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * window_size, 64)
        self.fc2 = nn.Linear(64, len(target))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Step 6: Initialize model, loss, and optimizer
model = CNN1D()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train the model
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# Step 8: Final evaluation
#model.eval()
#with torch.no_grad():
#    predictions = model(X_test)
#    mse = criterion(predictions, y_test).item()
#    print(f"Final Test MSE: {mse:.4f}")
#
## Step 9: Inverse transform the predictions (optional)
#predictions = scaler_y.inverse_transform(predictions.numpy())
#y_test = scaler_y.inverse_transform(y_test.numpy())

from sklearn.metrics import r2_score


# Step 8: Final evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test).item()

    # Convert to NumPy arrays for R^2 calculation
    predictions_np = predictions.numpy()
    y_test_np = y_test.numpy()

    # Calculate R^2
    r2 = r2_score(y_test_np, predictions_np)

print(f"Final Test MSE: {mse:.4f}")
print(f"Final Test R^2 Score: {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(y_train.numpy().flatten(), bins=30, alpha=0.7, label='Train')
plt.hist(y_test.numpy().flatten(), bins=30, alpha=0.7, label='Test')
plt.legend()
plt.title("Distribution of Target Values")
plt.subplot(1, 2, 2)
plt.boxplot([y_train.numpy().flatten(), y_test.numpy().flatten()], labels=["Train", "Test"])
plt.title("Boxplot of Target Values")
plt.show()
```

```
Epoch 1/200, Train Loss: 0.9145, Test Loss: 0.4756
Epoch 2/200, Train Loss: 0.8898, Test Loss: 0.4527
Epoch 3/200, Train Loss: 0.8612, Test Loss: 0.4292
Epoch 4/200, Train Loss: 0.8357, Test Loss: 0.4053
...
Epoch 199/200, Train Loss: 0.0170, Test Loss: 0.0071
Epoch 200/200, Train Loss: 0.0214, Test Loss: 0.0072
Final Test MSE: 0.0072
Final Test R^2 Score: -16.4196
```

可以看出，损失直接小了很多，MSE也很小，但是R2分数负数，分析问题

![image](https://github.com/user-attachments/assets/b2239a3d-d951-41d6-860c-d329b2ed195b)

可能原因分析
测试集数据分布问题：

如果测试集与训练集的分布差异过大（例如，数据偏移或不均衡），模型可能无法在测试集上生成合理的预测结果。
数据泄漏或标签问题：

检查数据预处理和滑动窗口的实现，确保测试集的目标变量没有被训练数据提前泄露。
检查数据中的目标值（torque 和 speed），确认是否存在异常值或错误标注。
模型过拟合：

尽管 Train Loss 和 Test Loss 都很低，模型可能过度拟合训练集，而未学到测试集的通用模式。
检查模型的复杂度和正则化（如 Dropout 的使用）。

评价指标问题：
𝑅2计算中，目标值的分布非常集中或有极端值时，SS_tot会变得非常小，导致 𝑅2异常。

应该是训练和测试数据集划分有问题，训练数据集取了80%,测试20%，训练集包含启动的部分，而测试集只包含了稳定运行的部分（下一步准备增加数据，将启动和停止部分加上去）





