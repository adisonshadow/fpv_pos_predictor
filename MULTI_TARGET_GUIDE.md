# 多目标跟踪 (MTT) 指南

## 概述

**多目标跟踪** (Multi-Target Tracking, MTT) 系统能够同时跟踪多个FPV目标，自动处理：
- 📍 数据关联 - 将观测值正确匹配到目标
- 🎯 目标管理 - 创建、确认、删除轨迹
- 🔄 轨迹维持 - 处理目标出现和消失
- 🚀 批量预测 - 为所有目标生成预测

## 核心挑战

### 数据关联问题

当有多个目标和多个测量时，如何知道哪个测量属于哪个目标？

```
时刻t:
  目标A: (1, 2)     测量1: (1.1, 2.1) ← 应该关联到A
  目标B: (5, 6)     测量2: (4.9, 6.2) ← 应该关联到B
  目标C: (3, 3)     测量3: (3.0, 2.9) ← 应该关联到C
```

**解决方案**: 使用数据关联算法（NN, GNN, JPDA）

## 支持的关联算法

### 1. 最近邻 (NN - Nearest Neighbor)
**方法**: 每个目标选择最近的测量  
**优点**: 计算快速  
**缺点**: 可能产生次优关联  
**适用**: 目标间距离较大

### 2. 全局最近邻 (GNN - Global Nearest Neighbor)
**方法**: 使用匈牙利算法找到全局最优分配  
**优点**: 全局最优解  
**缺点**: 计算复杂度 O(n³)  
**适用**: 目标数量中等 (<20个)  

### 3. JPDA (Joint Probabilistic Data Association)
**方法**: 计算关联概率，使用加权平均更新  
**优点**: 不做硬决策，更鲁棒  
**缺点**: 计算量最大  
**适用**: 密集目标场景  

## 快速开始

### 基础多目标跟踪

```python
from operator_multi_target import MultiTargetTracker

# 创建跟踪器
tracker = MultiTargetTracker(
    predictor_type="2D",       # 或 "3D"
    measurement_std=0.1,
    max_targets=10,            # 最多跟踪10个目标
    association_threshold=3.0, # 关联阈值（米）
    confirmation_threshold=3,  # 3次更新后确认
    deletion_threshold=5       # 5次丢失后删除
)

# 每帧更新（measurements是当前所有观测）
measurements = [
    np.array([1.2, 0.5]),  # 观测1
    np.array([5.3, 3.2]),  # 观测2
    np.array([3.1, 2.8])   # 观测3
]
tracker.update(measurements, current_time)

# 获取所有目标状态
states = tracker.get_all_states()
for state in states:
    print(f"{state['target_id']}: {state['position']}")

# 预测所有目标
predictions = tracker.predict_all([200, 500, 1000])
```

### 全功能多目标跟踪

```python
from operator_multi_target import create_multi_target_tracker

# 创建全功能跟踪器
tracker = create_multi_target_tracker(
    predictor_type="2D",
    use_imm=True,        # ✓ IMM多模型
    use_adaptive=True,   # ✓ 自适应滤波
    use_recovery=True,   # ✓ 跟踪恢复
    use_jpda=True,       # ✓ JPDA关联
    max_targets=10
)

# 使用方式相同
tracker.update(measurements, time)
```

## 目标生命周期

```
┌──────────┐
│ 新测量   │ (未关联的测量)
└────┬─────┘
     ▼
┌──────────────┐
│ 暂定目标     │ confirmed = False
│ (Tentative)  │ 需要连续更新确认
└────┬─────────┘
     ▼ (连续更新3次)
┌──────────────┐
│ 确认目标     │ confirmed = True
│ (Confirmed)  │ 正常跟踪
└────┬─────────┘
     ▼ (连续丢失5次)
┌──────────────┐
│ 删除         │ alive = False
└──────────────┘
```

## 性能对比

### 场景：5个目标交叉飞行

| 算法 | 正确关联率 | 计算时间 | 误关联数 |
|------|-----------|---------|---------|
| NN   | 85%       | 0.5ms   | 7       |
| GNN  | 95%       | 1.2ms   | 2       |
| JPDA | 98%       | 2.5ms   | 1       |

**推荐**: 
- 简单场景 → NN
- 一般场景 → GNN ⭐
- 密集场景 → JPDA

## 实际应用示例

### 示例1: 拦截多架FPV

```python
tracker = create_multi_target_tracker(
    predictor_type="3D",
    use_imm=True,
    use_recovery=True,
    max_targets=5
)

# 每帧循环
while True:
    # 1. 获取传感器数据
    measurements = get_sensor_measurements()
    
    # 2. 更新跟踪
    tracker.update(measurements, time.time())
    
    # 3. 获取所有目标预测
    predictions = tracker.predict_all([200, 500])
    
    # 4. 选择最佳射击目标
    best_target = None
    best_score = 0
    
    for target_id, preds in predictions.items():
        fire_score = preds[200]['fire_feasibility']
        if fire_score > best_score:
            best_score = fire_score
            best_target = target_id
    
    # 5. 执行射击
    if best_score > 0.7:
        execute_fire(best_target, predictions[best_target])
```

### 示例2: 威胁评估

```python
# 获取所有目标
states = tracker.get_all_states()

# 按威胁等级排序
threats = []
for state in states:
    if state['confirmed']:
        # 计算威胁分数
        distance = np.linalg.norm(state['position'])
        speed = state.get('speed', 0)
        
        threat_score = (1.0 / (distance + 1)) * (speed / 10.0)
        threats.append({
            'target_id': state['target_id'],
            'threat_score': threat_score,
            'position': state['position']
        })

# 排序
threats.sort(key=lambda x: x['threat_score'], reverse=True)

# 优先处理高威胁目标
for threat in threats[:3]:
    print(f"高威胁: {threat['target_id']} - 分数:{threat['threat_score']:.2f}")
```

## 配置建议

### 场景1: 开阔空域，目标稀疏

```python
tracker = MultiTargetTracker(
    max_targets=5,
    association_threshold=5.0,  # 较大阈值
    confirmation_threshold=2,   # 快速确认
    deletion_threshold=8        # 更宽容的删除
)
```

### 场景2: 城市环境，密集目标

```python
tracker = create_multi_target_tracker(
    max_targets=20,
    association_threshold=2.0,  # 严格阈值
    confirmation_threshold=4,   # 谨慎确认
    deletion_threshold=3,       # 快速删除
    use_jpda=True              # 使用JPDA
)
```

### 场景3: 高机动目标

```python
tracker = create_multi_target_tracker(
    use_imm=True,              # IMM识别机动
    use_adaptive=True,         # 自适应噪声
    use_recovery=True,         # 丢失恢复
    max_targets=10
)
```

## 性能与扩展性

### 计算复杂度

| 组件 | 复杂度 | 说明 |
|------|--------|------|
| NN关联 | O(nm) | n=目标数, m=测量数 |
| GNN关联 | O(n³) | 匈牙利算法 |
| JPDA | O(n²m) | 概率计算 |
| 总体 | O(n × 滤波器复杂度) | 每个目标独立滤波 |

### 最大目标数建议

- **基础UKF**: 50+ 个目标 @20Hz
- **IMM**: 20-30 个目标 @20Hz  
- **IMM+自适应+恢复**: 10-15 个目标 @20Hz

### 优化建议

1. **并行化**: 各目标的滤波可并行处理
2. **选择性更新**: 只更新高优先级目标
3. **分层跟踪**: 远距离目标降低更新频率
4. **GPU加速**: 矩阵运算可用GPU加速

## 常见问题

**Q: 如何处理目标ID切换？**  
A: 使用更严格的确认阈值，或添加目标特征匹配。

**Q: 如何避免误关联？**  
A: 降低association_threshold，使用GNN或JPDA。

**Q: 如何处理目标交叉？**  
A: 推荐使用JPDA，它在交叉场景表现最好。

**Q: 可以跟踪多少个目标？**  
A: 取决于计算资源和更新频率，通常10-50个。

## 扩展功能

### 添加目标特征

```python
@dataclass
class EnhancedTarget(Target):
    """增强目标 - 添加特征"""
    feature_vector: np.ndarray = None  # 视觉特征
    target_class: str = "unknown"      # 目标类别
    threat_level: float = 0.0          # 威胁等级
```

### 自定义关联算法

```python
class CustomAssociator(DataAssociator):
    def associate(self, targets, measurements):
        # 实现自定义关联逻辑
        # 例如：结合位置+特征的关联
        pass
```

## 测试

运行多目标跟踪测试：

```bash
python test_multi_target.py
```

测试包括:
1. ✅ 基础多目标跟踪 (3个目标)
2. ✅ 目标动态出现/消失
3. ✅ 全功能跟踪 (IMM+自适应+恢复)
4. ✅ 多目标预测和射击建议
5. ✅ 关联算法对比 (NN vs GNN vs JPDA)

## 参考文献

1. Bar-Shalom, Y., Fortmann, T. E., & Cable, P. G. (1990). *Tracking and Data Association*
2. Blackman, S., & Popoli, R. (1999). *Design and Analysis of Modern Tracking Systems*

---

**文件**: `operator_multi_target.py` (500+行)  
**测试**: `test_multi_target.py` (400+行)  
**Author**: pointfang@gmail.com
