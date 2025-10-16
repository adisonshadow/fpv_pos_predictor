# FPV 拦截预测算法

----------
应对FPV这种**高度机动**的目标的**非线性**轨迹预测，使用**无损卡尔曼滤波 (Unscented Kalman Filter, UKF)** 算法，来捕捉到FPV的复杂运动状态。

本算法提供 **2D** 和 **3D** 两种模式，适配不同应用场景。

### 🎯 核心特性

- ✅ **双模式支持**: 2D CTRV模型 + 3D恒定加速度模型
- ✅ **标准化接口**: 基于OperatorIO协议，支持热插拔和跨语言调用
- ✅ **实时预测**: 支持多时间点预测（50ms-1000ms）
- ✅ **射击建议**: 自动评估目标可打击性
- ✅ **微服务就绪**: JSON/Protobuf接口，易于集成

## 模型架构

### 1. 2D CTRV 模型 (FlyPredictor)

使用 **恒定转弯率和速度 (Constant Turn Rate and Velocity, CTRV)** 模型，适用于平面运动场景。

**状态向量 $\mathbf{x}_{2D}$：**
$$\mathbf{x}_{2D} = [x, y, v, \phi, \dot{\phi}]^T$$
其中：
  * $x, y$：平面位置坐标
  * $v$：速度 (标量)
  * $\phi$：航向角 (Heading)
  * $\dot{\phi}$：转弯率 (Turn Rate, 角速度)

**特点：**
- 适合平面运动跟踪
- 能够捕捉直线飞行和转弯两种关键运动状态
- 计算效率高

### 2. 3D 恒定加速度模型 (FlyPredictor3D) ✨

使用恒定加速度模型，适用于真实的3D空间运动场景。

**状态向量 $\mathbf{x}_{3D}$：**
$$\mathbf{x}_{3D} = [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T$$
其中：
  * $x, y, z$：3D空间位置坐标
  * $v_x, v_y, v_z$：3D速度分量
  * $a_x, a_y, a_z$：3D加速度分量

**运动学方程：**
$$\mathbf{p}_{t+\Delta t} = \mathbf{p}_t + \mathbf{v}_t \Delta t + \frac{1}{2}\mathbf{a}_t \Delta t^2$$
$$\mathbf{v}_{t+\Delta t} = \mathbf{v}_t + \mathbf{a}_t \Delta t$$
$$\mathbf{a}_{t+\Delta t} = \mathbf{a}_t \quad \text{(恒定加速度假设)}$$

**特点：**
- 支持完整的3D空间运动
- 能够跟踪速度和加速度的三维分量
- 更精确地预测高度机动目标
- 适合实际FPV拦截应用

## 快速开始

### 方式1: 直接调用（适合Python项目）

#### 2D 模型
```python
from operator import FlyPredictor

# 初始化预测器
predictor = FlyPredictor(initial_pos=[0, 0], measurement_std=0.1, process_std=0.5)

# 更新测量
predictor.update(current_pos=[1.2, 0.5], current_time=0.05)

# 预测未来位置
predictions = predictor.predict_and_evaluate([50, 200, 500, 1000])

# 获取当前状态
state = predictor.get_current_state()
```

#### 3D 模型
```python
from operator import FlyPredictor3D

# 初始化3D预测器
predictor = FlyPredictor3D(initial_pos=[0, 0, 10], measurement_std=0.1, process_std=0.5)

# 更新测量
predictor.update(current_pos=[1.2, 0.5, 10.3], current_time=0.05)

# 预测未来位置
predictions = predictor.predict_and_evaluate([50, 200, 500, 1000])

# 获取当前状态（包含速度和加速度）
state = predictor.get_current_state()
print(f"速度: {state['velocity']}, 加速度: {state['acceleration']}")
```

### 方式2: 标准化算子接口（适合微服务/跨语言）

使用 **OperatorIO** 标准协议，支持热插拔和跨语言调用：

```python
from operator_wrapper import FPVPredictorOperator, create_position_input

# 1. 初始化算子
config = {
    "type": "3D",
    "initial_position": [0.0, 0.0, 10.0],
    "measurement_std": 0.15,
    "process_std": 0.6,
    "prediction_delays": [50, 200, 500, 1000]
}
operator = FPVPredictorOperator(config)

# 2. 创建标准输入（OperatorIO格式）
input_io = create_position_input(
    position=[1.5, 2.0, 10.5],
    timestamp=0.05,
    source="camera_01",
    dimension="3D"
)

# 3. 处理并获取结果
output_io = operator.process_input(input_io)

# 4. 解析预测结果
predictions = output_io['data_bodies'][0]['prediction']['predictions']
for pred in predictions:
    print(f"{pred['delay_ms']}ms: {pred['predicted_position']} "
          f"射击建议: {pred['fire_feasibility']:.3f}")
```

**标准接口优势：**
- 🔄 支持JSON/Protobuf跨语言调用
- 🔌 热插拔，无需重启系统
- 🌐 易于集成REST API、gRPC、Kafka等
- 📊 完善的错误码和监控指标

详见: [算子接口文档](OPERATOR_INTERFACE.md)

### 方式3: IMM多模型融合（最高精度） ⭐

**IMM** (交互式多模型) 是跟踪高度机动目标的黄金标准，能自动识别运动模式：

```python
from operator_imm import IMMPredictor2D, IMMPredictor3D

# 初始化2D IMM预测器 (包含4个模型: CV/CA/CT/Hover)
predictor = IMMPredictor2D(initial_pos=[0, 0], measurement_std=0.1)

# 更新测量
predictor.update(current_pos=[1.2, 0.5], current_time=0.05)

# 预测 - 自动选择最佳模型组合
predictions = predictor.predict_and_evaluate([50, 200, 500, 1000])

# 查看当前活跃模型
state = predictor.get_current_state()
print(f"活跃模型: {state['active_model']}")
print(f"模型概率: CV={state['model_probabilities'][0]:.2f}, "
      f"CA={state['model_probabilities'][1]:.2f}, "
      f"CT={state['model_probabilities'][2]:.2f}, "
      f"Hover={state['model_probabilities'][3]:.2f}")
```

**IMM优势：**
- 🎯 自动识别运动模式（匀速/加速/转弯/悬停）
- 🔄 在机动变化时快速适应
- 📈 预测精度优于单模型方法
- 🎨 提供模型概率，可用于决策分析

## 结果解读和射击建议

### 1\. 预测位置及准确率 (Accuracy Score)

  * **预测位置:** `predicted_position` 是 UKF 根据当前估计状态和运动模型推算出来的 $50\text{ms}$ / $200\text{ms}$ / $500\text{ms}$ / $1\text{s}$ 后的位置坐标。
      * 2D 模型: $(x, y)$
      * 3D 模型: $(x, y, z)$
  * **准确率 (Accuracy Score):**
      * 这是一个**0 到 1**的浮点数，**越接近 1 越准确**。
      * 它基于预测位置的协方差矩阵 $\mathbf{P}$。$\mathbf{P}$ 越大，说明预测的**不确定性**越高，因此准确率得分越低。
      * **$50\text{ms}$** 的预测通常会比 **$1\text{s}$** 的预测准确率高得多，因为不确定性会随时间累积。
      * **工程意义：** 这个值可以用来调整激光器的**瞄准区域大小**（协方差大的时候，可以扩大射击范围）。

### 2\. 射击建议 (Fire Feasibility)

  * **核心逻辑:** FPV运动越复杂、越剧烈，其轨迹越难以预测，因此越不适合射击。
  * **计算方式:** 
      * 2D 模型: 结合**速度 $v$** 和**转弯率 $\dot{\phi}$** 两个关键指标
      * 3D 模型: 结合**速度模 $|\mathbf{v}|$** 和**加速度模 $|\mathbf{a}|$** 两个关键指标
  * **$0 \sim 1$ 浮点数:**
      * **接近 1.0 (例如 $>0.8$):** FPV运动平稳（低速、低加速度），预测精度高，**适合射击**。
      * **接近 0.0 (例如 $<0.3$):** FPV正在高速或剧烈机动，预测精度低，**不适合射击**。

## API 参考

### 通用方法（2D 和 3D 都支持）

| 方法 | 说明 | 参数 | 返回值 |
|------|------|------|--------|
| `__init__(initial_pos, measurement_std, process_std)` | 初始化预测器 | 初始位置、测量噪声、过程噪声 | - |
| `update(current_pos, current_time)` | 更新当前测量 | 当前位置、当前时间 | - |
| `predict_and_evaluate(delay_ms_list)` | 预测未来位置并评估 | 延迟时间列表(ms) | 预测字典 |
| `get_current_state()` | 获取当前估计状态 | - | 状态字典 |
| `reset(initial_pos, initial_time)` | 重置预测器 | 新初始位置、新初始时间 | - |
| `adjust_noise_parameters(measurement_std, process_std)` | 动态调整噪声参数 | 可选：测量噪声、过程噪声 | - |

## 技术特点

### 优势
✅ **非线性处理能力强**: UKF 能够处理高度非线性的运动模型  
✅ **实时性能好**: 计算复杂度适中，适合实时应用  
✅ **自适应能力**: 通过协方差矩阵自动调整不确定性  
✅ **双模式支持**: 2D 和 3D 模型满足不同场景需求  
✅ **完整的状态估计**: 不仅预测位置，还估计速度和加速度  

### 限制
⚠️ **单一模型假设**: 每个预测器只使用一种运动模型  
⚠️ **参数敏感**: 噪声参数需要根据实际传感器调优  
⚠️ **恒定假设**: 假设速度/加速度在短时间内保持相对恒定  


## TODO 与未来改进

1.  ✅ ~~**3D 状态**~~: 已实现 `FlyPredictor3D` 类，支持完整的3D空间运动跟踪。
2.  ✅ ~~**标准化接口**~~: 已实现基于OperatorIO的标准化算子接口。
3.  ✅ ~~**多模型 (IMM)**~~: 已实现**交互式多模型 (IMM)** 算法！IMM能够：
    - 同时运行4个运动模型 (2D: CV/CA/CT/Hover, 3D: CV/CA/Hover)
    - 根据观测数据自动调整各模型权重
    - 实时识别目标运动模式变化 (直线、加速、转弯、悬停)
    - 通过加权融合提供更准确的预测结果
    - 详见 `operator_imm.py` 和 `test_imm.py`
4.  **实际参数调优:** 代码中的 `measurement_std` (传感器噪声) 和 `process_std` (模型噪声) 必须根据实际的视觉识别系统（摄像头、相控阵雷达）的性能进行反复测试和调优。
5.  **自适应噪声估计:** 实现自适应卡尔曼滤波 (AKF)，根据观测残差自动调整 Q 和 R 矩阵。
6.  **目标跟踪丢失恢复:** 添加跟踪丢失检测和重新初始化机制。
7.  **gRPC服务:** 基于Protobuf定义实现gRPC服务端。


## 许可证

MIT License

