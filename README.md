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


在电机控制中，\(i_{uvw}\)、\(u_{uvw}\) 是指定子三相电流和电压，\(i_{d}\)、\(i_{q}\)、\(u_{d}\)、\(u_{q}\) 是在旋转坐标系（dq坐标系）中的电流和电压分量。将三相坐标系 (\(uvw\)) 转换到两相旋转坐标系 (\(dq\))，需要进行坐标变换，这包括 **Clark变换** 和 **Park变换**。

以下是主要步骤：

---

1. **从三相坐标系到两相静止坐标系（\(\alpha\beta\)） - Clark变换**
Clark变换将三相信号投影到一个静止的两相平面上。假设三相电流为 \(i_u\), \(i_v\), \(i_w\)，我们有：

\[
i_\alpha = i_u
\]
\[
i_\beta = \frac{1}{\sqrt{3}} (i_u + 2i_v)
\]

若三相电流满足平衡（\(i_u + i_v + i_w = 0\)），上述公式可以简化为：

\[
\begin{bmatrix}
i_\alpha \\
i_\beta
\end{bmatrix}
=
\frac{2}{3}
\begin{bmatrix}
1 & -\frac{1}{2} & -\frac{1}{2} \\
0 & \frac{\sqrt{3}}{2} & -\frac{\sqrt{3}}{2}
\end{bmatrix}
\begin{bmatrix}
i_u \\
i_v \\
i_w
\end{bmatrix}
\]

同理适用于电压 (\(u_u, u_v, u_w\))。

---

2. **从静止坐标系到旋转坐标系（\(dq\)） - Park变换**
Park变换将 \(\alpha\beta\) 坐标系中的量投影到旋转坐标系 \(dq\)。旋转坐标系与电机转子的旋转位置同步，使用转子磁链位置角 \(\theta\) 进行变换：

\[
\begin{bmatrix}
i_d \\
i_q
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
i_\alpha \\
i_\beta
\end{bmatrix}
\]

\[
\begin{bmatrix}
u_d \\
u_q
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
u_\alpha \\
u_\beta
\end{bmatrix}
\]

其中，\(\theta\) 通常由电机控制器中的位置传感器（如编码器）或观测器提供。

---

3. **总结**
- Clark变换：将三相信号投影到静止的 \(\alpha\beta\) 平面；
- Park变换：将静止的 \(\alpha\beta\) 信号投影到旋转的 \(dq\) 坐标系；
- \(i_d\)、\(i_q\) 分别对应直轴（d轴）和交轴（q轴）的电流分量，通常 \(i_d\) 用于控制磁链，\(i_q\) 用于控制电磁转矩。

完整流程为：
\[
i_{uvw} \xrightarrow{\text{Clark}} i_{\alpha\beta} \xrightarrow{\text{Park}} i_{dq}
\]
