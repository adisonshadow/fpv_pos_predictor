# FPV预测算子 - 标准化接口文档

## 概述

本项目实现了基于 **OperatorIO** 标准协议的FPV预测算子，支持热插拔、跨语言调用和微服务集成。

### 核心特性

- ✅ **标准化接口**: 基于Protobuf定义的OperatorIO协议
- ✅ **JSON兼容**: 支持JSON格式数据交换，便于REST API集成
- ✅ **跨语言**: 协议可用Python、Go、Java、C++等语言实现
- ✅ **热插拔**: 算子可动态加载/卸载，无需重启系统
- ✅ **错误处理**: 完善的错误码和错误信息反馈
- ✅ **多维度**: 同时支持2D和3D预测模式

## 架构设计

```
┌─────────────┐
│   调用方     │ (微服务/流式计算引擎/Web API)
└──────┬──────┘
       │ JSON/Protobuf
       ▼
┌─────────────────────────────────────┐
│     OperatorIO 标准接口层            │
│  ┌───────────────────────────────┐  │
│  │  FPVPredictorOperator         │  │
│  │  - process_input()            │  │
│  │  - handle_update()            │  │
│  │  - handle_predict()           │  │
│  └───────────────────────────────┘  │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│      核心预测算法                     │
│  ┌────────────┐  ┌─────────────┐   │
│  │ FlyPredictor│  │FlyPredictor3D│  │
│  │   (2D UKF) │  │   (3D UKF)  │   │
│  └────────────┘  └─────────────┘   │
└──────────────────────────────────────┘
```

## OperatorIO 协议详解

### 1. 协议结构

```json
{
  "metadata": {
    "io_id": "唯一ID",
    "data_type": "数据类型",
    "shape": [维度],
    "source": "数据来源",
    "timestamp": 时间戳毫秒,
    "ext": {}
  },
  "data_bodies": [
    {
      "position": {...},  // 位置数据
      "prediction": {...}, // 预测结果
      "state": {...}      // 状态估计
    }
  ],
  "control_info": {
    "op_action": "动作类型",
    "priority": 优先级,
    "params": {}
  },
  "error": {
    "code": 错误码,
    "msg": "错误信息",
    "detail": "详细描述"
  }
}
```

### 2. 支持的动作 (op_action)

| 动作 | 说明 | 输入数据 | 输出数据 |
|------|------|---------|---------|
| `compute` | 更新并预测 | 位置数据 | 预测结果 |
| `update` | 仅更新测量 | 位置数据 | 更新确认 |
| `predict` | 仅执行预测 | 预测参数 | 预测结果 |
| `get_state` | 获取状态估计 | - | 状态数据 |
| `reset` | 重置预测器 | 新初始位置 | 重置确认 |

### 3. 数据类型

#### 位置数据 (PositionData)

```json
{
  "position": {
    "dimension": "2D" | "3D",
    "coordinates": [x, y] | [x, y, z],
    "timestamp": 时间戳秒,
    "confidence": 0.0-1.0,
    "sensor_id": "传感器ID"
  }
}
```

#### 预测结果 (PredictionResult)

```json
{
  "prediction": {
    "predictions": [
      {
        "delay_ms": 延迟毫秒,
        "predicted_position": [x, y] | [x, y, z],
        "accuracy_score": 准确率 (0-1),
        "fire_feasibility": 射击可行性 (0-1)
      }
    ],
    "prediction_time": 预测时刻时间戳,
    "predictor_type": "2D" | "3D"
  }
}
```

#### 状态估计 (StateEstimate)

```json
{
  "state": {
    "position": [x, y] | [x, y, z],
    "velocity": [vx, vy] | [vx, vy, vz],
    "acceleration": [ax, ay, az],  // 仅3D
    "speed": 速度模,
    "acceleration_magnitude": 加速度模,  // 仅3D
    "uncertainty": 位置不确定性,
    "timestamp": 时间戳
  }
}
```

### 4. 错误码定义

| 错误码 | 错误信息 | 说明 |
|--------|---------|------|
| 1000 | 数据解析失败 | JSON格式错误 |
| 1001 | 数据类型不兼容 | 数据类型不匹配 |
| 1002 | 维度不匹配 | 位置维度错误 |
| 1003 | 时间戳异常 | 时间戳倒序或无效 |
| 1004 | 参数缺失 | 必需参数未提供 |
| 1005 | 算子状态异常 | 算子内部错误 |
| 2000 | 预测器未初始化 | 算子未正确初始化 |
| 2001 | 位置数据无效 | 位置数据格式错误 |

## 使用示例

### Python

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

# 2. 发送位置数据
input_io = create_position_input(
    position=[1.5, 2.0, 10.5],
    timestamp=0.05,
    source="camera_01",
    dimension="3D"
)

# 3. 处理并获取结果
output_io = operator.process_input(input_io)

# 4. 解析预测结果
if output_io['data_bodies']:
    predictions = output_io['data_bodies'][0]['prediction']['predictions']
    for pred in predictions:
        print(f"{pred['delay_ms']}ms: {pred['predicted_position']}")
```

### REST API示例

```bash
# 更新位置
curl -X POST http://localhost:8080/fpv_predictor/process \
  -H "Content-Type: application/json" \
  -d '{
    "metadata": {
      "io_id": "req_001",
      "data_type": "position_3d",
      "shape": [3],
      "source": "radar_01",
      "timestamp": 1697356800000,
      "ext": {}
    },
    "data_bodies": [{
      "position": {
        "dimension": "3D",
        "coordinates": [1.5, 2.0, 10.5],
        "timestamp": 0.05,
        "confidence": 1.0,
        "sensor_id": "radar_01"
      }
    }],
    "control_info": {
      "op_action": "compute",
      "priority": 1,
      "params": {}
    },
    "error": null
  }'
```

### gRPC示例 (Go)

```go
// 使用生成的Protobuf代码
client := fpv_predictor.NewFPVPredictorClient(conn)

request := &fpv_predictor.OperatorIO{
    Metadata: &fpv_predictor.OperatorIO_Metadata{
        IoId:      "req_001",
        DataType:  "position_3d",
        Shape:     []int32{3},
        Source:    "radar_01",
        Timestamp: time.Now().UnixMilli(),
    },
    DataBodies: []*fpv_predictor.OperatorIO_DataBody{
        {
            DataType: &fpv_predictor.OperatorIO_DataBody_Position{
                Position: &fpv_predictor.PositionData{
                    Dimension:   fpv_predictor.PositionData_DIM_3D,
                    Coordinates: []float32{1.5, 2.0, 10.5},
                    Timestamp:   0.05,
                    Confidence:  1.0,
                    SensorId:    "radar_01",
                },
            },
        },
    },
    ControlInfo: &fpv_predictor.OperatorIO_ControlInfo{
        OpAction: "compute",
        Priority: 1,
    },
}

response, err := client.Process(ctx, request)
```

## 集成场景

### 1. 微服务架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 传感器服务   │────▶│ FPV预测算子  │────▶│ 拦截控制服务 │
└─────────────┘     └─────────────┘     └─────────────┘
     Kafka              REST/gRPC           WebSocket
```

### 2. 流式计算 (Flink/Spark)

```python
# Flink DataStream API
positions_stream
    .map(lambda x: create_position_input(x))
    .process(FPVPredictorOperator(config))
    .filter(lambda x: x['data_bodies'][0]['prediction']['fire_feasibility'] > 0.8)
    .sink_to(interception_system)
```

### 3. 消息队列集成

```python
# Kafka Consumer
for message in kafka_consumer:
    operator_io = json.loads(message.value)
    result = operator.process_input(operator_io)
    
    if result['error'] is None:
        kafka_producer.send('prediction_results', json.dumps(result))
```

## 性能优化

### 批处理模式

```python
# 批量处理历史数据
batch = []
for position in historical_data:
    input_io = create_position_input(position['coords'], position['time'])
    result = operator.process_input(input_io)
    batch.append(result)

# 批量写入数据库
db.bulk_insert(batch)
```

### 异步处理

```python
import asyncio

async def process_async(operator, input_io):
    # 在线程池中执行
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, operator.process_input, input_io)

# 并发处理多个请求
results = await asyncio.gather(
    process_async(operator, io1),
    process_async(operator, io2),
    process_async(operator, io3)
)
```

## 部署建议

### Docker化

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY operator.py operator_wrapper.py ./
EXPOSE 8080

CMD ["python", "operator_service.py"]
```

### Kubernetes配置

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fpv-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fpv-predictor
  template:
    metadata:
      labels:
        app: fpv-predictor
    spec:
      containers:
      - name: predictor
        image: fpv-predictor:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## 监控指标

建议收集以下指标：

- **吞吐量**: 每秒处理的请求数
- **延迟**: P50/P95/P99延迟
- **错误率**: 各类错误码的出现频率
- **预测准确率**: accuracy_score的统计分布
- **内存使用**: 算子内存占用

## 扩展接口

### 自定义动作

```python
class CustomFPVOperator(FPVPredictorOperator):
    def process_input(self, operator_io):
        action = operator_io['control_info'].get('op_action')
        
        if action == 'custom_action':
            return self._handle_custom_action(operator_io)
        
        return super().process_input(operator_io)
    
    def _handle_custom_action(self, operator_io):
        # 自定义逻辑
        pass
```

## 常见问题

**Q: 如何提高预测精度？**  
A: 调整 `measurement_std` 和 `process_std` 参数，根据实际传感器性能调优。

**Q: 支持多目标跟踪吗？**  
A: 当前版本为单目标预测器，多目标需要实例化多个算子。

**Q: 如何处理传感器数据丢失？**  
A: 检查返回的 `error` 字段，根据错误码进行重试或告警。

**Q: 可以动态切换2D/3D模式吗？**  
A: 需要重新初始化算子，使用 `reset` 动作无法切换维度。

## 相关文件

- `operator_io.proto` - Protobuf协议定义
- `operator_wrapper.py` - Python实现
- `example_operator_usage.py` - 使用示例
- `operator.py` - 核心算法实现

## 许可证

MIT License

