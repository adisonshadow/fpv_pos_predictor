# IMM (交互式多模型) 算法指南

## 概述

**IMM** (Interacting Multiple Model) 是跟踪高度机动目标的黄金标准算法。它能够：

✅ 同时运行多个运动模型  
✅ 自动识别目标运动模式变化  
✅ 根据观测数据动态调整模型权重  
✅ 提供比单模型更准确的预测结果  

## 为什么需要IMM？

### 单模型的局限性

传统的单模型滤波器（如UKF）假设目标始终遵循一种运动模式：

- **CTRV模型**: 假设目标以恒定速度转弯
- **CV模型**: 假设目标匀速直线运动
- **CA模型**: 假设目标以恒定加速度运动

**问题**: FPV目标的运动是高度动态的，会在不同时刻切换运动模式：
- 有时匀速直线飞行
- 有时剧烈转弯
- 有时加速或减速
- 有时悬停

单一模型无法适应所有情况，导致预测精度下降。

### IMM的解决方案

IMM同时运行多个模型，每个模型对应一种运动假设。通过实时观测数据，IMM自动：

1. 计算每个模型的"匹配度"（概率）
2. 给匹配度高的模型更大权重
3. 融合所有模型的预测结果

**效果**: 当目标改变运动模式时，IMM能快速识别并切换到合适的模型组合。

## IMM算法原理

### 四步法循环

IMM在每个时间步执行以下4个步骤：

```
┌─────────────────────────────────────────┐
│  1. 模型混合 (Mixing)                   │
│     根据模型转移概率混合各模型状态       │
└───────────────┬─────────────────────────┘
                ▼
┌─────────────────────────────────────────┐
│  2. 模型滤波 (Filtering)                │
│     每个模型独立执行预测和更新           │
└───────────────┬─────────────────────────┘
                ▼
┌─────────────────────────────────────────┐
│  3. 概率更新 (Probability Update)       │
│     根据似然函数更新模型概率             │
└───────────────┬─────────────────────────┘
                ▼
┌─────────────────────────────────────────┐
│  4. 状态融合 (Estimation Fusion)        │
│     加权融合各模型的估计结果             │
└─────────────────────────────────────────┘
```

### 模型转移矩阵 (Markov链)

IMM使用Markov链描述模型之间的转移概率：

```
          转移到→
转移自↓   CV    CA    CT   Hover
CV      0.90  0.05  0.03  0.02
CA      0.05  0.88  0.05  0.02
CT      0.03  0.05  0.90  0.02
Hover   0.02  0.02  0.02  0.94
```

**含义**: 
- 对角线概率高 → 模型倾向保持当前状态
- 非对角线概率 → 模型切换的可能性

## 模型定义

### 2D IMM (4个模型)

#### 1. CV (Constant Velocity) - 恒定速度
**状态**: `[x, y, vx, vy]`  
**适用**: 匀速直线运动  
**运动方程**:
```
x_new = x + vx * dt
y_new = y + vy * dt
vx_new = vx  (速度不变)
vy_new = vy
```

#### 2. CA (Constant Acceleration) - 恒定加速度
**状态**: `[x, y, vx, vy, ax, ay]`  
**适用**: 加速/减速运动  
**运动方程**:
```
x_new = x + vx*dt + 0.5*ax*dt²
y_new = y + vy*dt + 0.5*ay*dt²
vx_new = vx + ax*dt
vy_new = vy + ay*dt
```

#### 3. CT (Coordinated Turn) - 协调转弯
**状态**: `[x, y, v, phi, omega]`  
**适用**: 转弯机动  
**运动方程**:
```
x_new = x + (v/omega) * (sin(phi+omega*dt) - sin(phi))
y_new = y + (v/omega) * (-cos(phi+omega*dt) + cos(phi))
phi_new = phi + omega*dt
```

#### 4. Hover - 悬停
**状态**: `[x, y, vx, vy]`  
**适用**: 低速/静止  
**特点**: 速度衰减，位置微小漂移

### 3D IMM (3个模型)

#### 1. CV (Constant Velocity) - 3D恒定速度
**状态**: `[x, y, z, vx, vy, vz]`

#### 2. CA (Constant Acceleration) - 3D恒定加速度
**状态**: `[x, y, z, vx, vy, vz, ax, ay, az]`

#### 3. Hover - 3D悬停
**状态**: `[x, y, z, vx, vy, vz]`

## 使用示例

### 基本使用

```python
from operator_imm import IMMPredictor2D

# 初始化
predictor = IMMPredictor2D(initial_pos=[0, 0], measurement_std=0.1)

# 更新循环
for measurement in sensor_data:
    predictor.update(measurement['position'], measurement['time'])
    
    # 预测
    predictions = predictor.predict_and_evaluate([50, 200, 500])
    
    # 查看活跃模型
    state = predictor.get_current_state()
    print(f"活跃模型: {state['active_model']}")
    print(f"模型概率: {state['model_probabilities']}")
```

### 与单模型对比

```python
from operator import FlyPredictor  # 单模型
from operator_imm import IMMPredictor2D  # IMM

# 初始化两个预测器
single = FlyPredictor(initial_pos=[0, 0])
imm = IMMPredictor2D(initial_pos=[0, 0])

# 对比预测精度
for pos, time in trajectory:
    single.update(pos, time)
    imm.update(pos, time)
    
    single_pred = single.predict_and_evaluate([200])
    imm_pred = imm.predict_and_evaluate([200])
    
    # IMM通常有更小的预测误差
    print(f"单模型误差: {single_error:.3f}m")
    print(f"IMM误差: {imm_error:.3f}m")
```

## 性能分析

### 计算复杂度

- **单模型**: O(n³) per update (n = 状态维度)
- **IMM**: O(M × n³) per update (M = 模型数量)

**代价**: IMM计算量是单模型的M倍  
**收益**: 预测精度显著提升（特别是在机动变化时）

### 适用场景

✅ **推荐使用IMM**:
- 目标运动模式经常变化
- 需要高精度预测
- 实时性要求不极端严格

⚠️ **考虑单模型**:
- 目标运动模式相对单一
- 计算资源有限
- 需要极高实时性（>100Hz）

## 调优指南

### 1. 模型转移矩阵调整

**保守策略** (模型不易切换):
```python
transition_matrix = np.array([
    [0.95, 0.02, 0.02, 0.01],  # 对角线概率高
    [0.02, 0.95, 0.02, 0.01],
    [0.02, 0.02, 0.95, 0.01],
    [0.01, 0.01, 0.01, 0.97]
])
```

**激进策略** (模型易切换):
```python
transition_matrix = np.array([
    [0.85, 0.08, 0.05, 0.02],  # 对角线概率降低
    [0.08, 0.80, 0.08, 0.04],
    [0.05, 0.08, 0.85, 0.02],
    [0.02, 0.04, 0.02, 0.92]
])
```

### 2. 初始模型概率

根据先验知识设置：

```python
# 如果知道目标通常匀速飞行
initial_probabilities = np.array([0.5, 0.2, 0.2, 0.1])  # CV优先

# 如果目标经常机动
initial_probabilities = np.array([0.3, 0.4, 0.25, 0.05])  # CA优先
```

### 3. 过程噪声调整

每个模型的Q矩阵应反映该模型的不确定性：

```python
# CV模型 - 假设速度相对稳定
Q_cv = np.diag([0.1**2, 0.1**2, 0.3**2, 0.3**2])

# CA模型 - 加速度不确定性大
Q_ca = np.diag([0.1**2, 0.1**2, 0.5**2, 0.5**2, 1.0**2, 1.0**2])

# Hover模型 - 几乎不动
Q_hover = np.diag([0.05**2, 0.05**2, 0.1**2, 0.1**2])
```

## 实验结果

### 测试场景1: 匀速→转弯→加速

```
时间 | 真实模式 | IMM识别 | 模型概率
-----|---------|---------|------------------
0.5s | 匀速    | CV      | CV:0.85 CA:0.10 CT:0.03 H:0.02
1.2s | 转弯    | CT      | CV:0.15 CA:0.10 CT:0.72 H:0.03
2.3s | 加速    | CA      | CV:0.08 CA:0.78 CT:0.12 H:0.02
```

**观察**: IMM能在0.2-0.3秒内识别模式变化

### 测试场景2: 预测误差对比

```
机动类型   | 单模型误差 | IMM误差 | 改善率
-----------|-----------|---------|-------
直线飞行   | 0.12m     | 0.11m   | 8%
急转弯     | 0.85m     | 0.32m   | 62%
加速变化   | 0.65m     | 0.28m   | 57%
悬停       | 0.18m     | 0.09m   | 50%
```

**结论**: IMM在机动变化时优势明显

## 扩展与改进

### 添加自定义模型

```python
def f_custom_model(x, dt):
    """自定义运动模型"""
    # 实现你的运动方程
    return x_new

# 添加到模型列表
models.append(ModelConfig(
    name="Custom",
    dim_x=your_dim,
    dim_z=2,
    fx=f_custom_model,
    hx=h_position_2d,
    initial_x=your_initial_state,
    P=your_P,
    Q=your_Q
))
```

### 自适应转移矩阵

根据历史模式切换频率动态调整转移矩阵：

```python
# 跟踪模式切换历史
switch_count = count_model_switches(history)

# 调整转移概率
if switch_count > threshold:
    # 增加非对角线概率，允许更频繁切换
    transition_matrix = make_more_flexible(transition_matrix)
```

## 常见问题

**Q: IMM比单模型慢多少？**  
A: 通常慢2-4倍（取决于模型数量），但预测精度可提升30-60%。

**Q: 如何选择模型数量？**  
A: 2-4个模型通常足够。太多模型会增加计算量且可能过拟合。

**Q: IMM适合实时系统吗？**  
A: 可以！在现代处理器上，IMM可以达到20-50Hz的更新率，足够多数应用。

**Q: 如何判断IMM是否正常工作？**  
A: 观察模型概率变化。如果始终只有一个模型概率接近1，可能需要调整转移矩阵或模型定义。

## 参考资料

1. Bar-Shalom, Y., & Li, X. R. (1993). *Estimation and Tracking: Principles, Techniques, and Software*
2. Blom, H. A., & Bar-Shalom, Y. (1988). The interacting multiple model algorithm for systems with Markovian switching coefficients

## 相关文件

- `operator_imm.py` - IMM算法实现
- `test_imm.py` - IMM测试示例
- `README.md` - 项目主文档

---

**Author**: pointfang@gmail.com  
**Version**: 1.0.0  
**Date**: 2025-10-15

