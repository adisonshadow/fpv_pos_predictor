# 模拟数据说明

这个目录包含用于测试FPV预测算法的模拟运动轨迹数据。

## 数据格式

所有数据文件都是JSON格式，包含时间序列的轨迹点。每个数据点包含:

```json
{
  "time": 0.5,                           // 时间戳 (秒)
  "true_position": [x, y] 或 [x, y, z],  // 真实位置 (无噪声)
  "measured_position": [x, y] 或 [x, y, z], // 测量位置 (含传感器噪声)
  "velocity": [vx, vy] 或 [vx, vy, vz], // 速度向量
  "acceleration": [ax, ay, az],          // 加速度向量 (仅3D)
  "scenario": "scenario_name"            // 场景名称
}
```

## 场景描述

### 2D 场景

#### 1. `2d_straight_line.json`
- **描述**: 匀速直线运动
- **参数**: 速度 2.5 m/s, 方向 45°
- **持续时间**: 2.0秒
- **数据点**: 7个
- **用途**: 测试基本的直线运动预测

### 3D 场景

#### 2. `3d_helical_climb.json`
- **描述**: 螺旋爬升运动
- **参数**: 
  - 螺旋半径: 2.5米
  - 角速度: 0.8 rad/s
  - 爬升速率: 1.5 m/s
- **持续时间**: 4.0秒
- **数据点**: 10个
- **用途**: 测试复杂3D机动预测

#### 3. `3d_aggressive_maneuver.json`
- **描述**: 高机动性运动（模拟FPV快速机动）
- **特点**: 
  - 高加速度变化
  - 不规则轨迹
  - 速度变化剧烈
- **持续时间**: 3.0秒
- **数据点**: 7个
- **用途**: 测试算法在极端情况下的表现

## 使用方法

### 方式1: 使用测试脚本

```bash
python test_with_mockdata.py
```

这个脚本会自动读取所有mock数据并运行预测器测试。

### 方式2: 手动加载数据

```python
import json
from operator import FlyPredictor, FlyPredictor3D

# 加载2D数据
with open('mockdata/2d_straight_line.json', 'r') as f:
    data_2d = json.load(f)

# 初始化2D预测器
predictor_2d = FlyPredictor(
    initial_pos=data_2d[0]['measured_position'],
    measurement_std=0.12,
    process_std=0.5
)

# 处理数据
for point in data_2d:
    predictor_2d.update(
        current_pos=point['measured_position'],
        current_time=point['time']
    )
    
    # 预测未来位置
    predictions = predictor_2d.predict_and_evaluate([50, 200, 500])
    print(predictions)
```

### 方式3: 生成自定义数据

修改并运行 `generate_mockdata.py` 来生成新的模拟数据:

```python
from generate_mockdata import MockDataGenerator

generator = MockDataGenerator(dt=0.05, noise_std=0.15)

# 生成自定义场景
data = generator.generate_3d_helical_climb(
    duration=5.0,
    radius=3.0,
    angular_velocity=1.2,
    climb_rate=2.0
)
```

## 数据特点

- **采样率**: 20 Hz (dt = 0.05秒)
- **测量噪声**: 
  - 2D数据: σ = 0.12米
  - 3D数据: σ = 0.12-0.15米
- **噪声模型**: 高斯白噪声

## 评估指标

使用这些数据可以评估:

1. **预测精度**: 比较预测位置与真实位置的误差
2. **准确率评分**: 算法输出的 `accuracy_score`
3. **射击建议**: 算法输出的 `fire_feasibility`
4. **状态估计**: 速度和加速度的估计准确性

## 扩展数据集

需要更多场景? 可以添加:
- 悬停运动
- S型机动
- 俯冲运动
- 随机轨迹
- 多目标场景

修改 `generate_mockdata.py` 中的生成器方法即可。

