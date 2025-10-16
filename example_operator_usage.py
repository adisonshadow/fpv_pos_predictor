"""
FPV预测算子使用示例 - 标准化接口

展示如何使用OperatorIO标准接口与FPV预测算子交互
"""

import json
import time
from operator_wrapper import (
    FPVPredictorOperator,
    create_position_input,
    create_update_only_input,
    create_predict_input
)


def print_separator(title=""):
    """打印分隔线"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)


def print_operator_io(io_data, title=""):
    """美化打印OperatorIO"""
    if title:
        print(f"\n【{title}】")
    print(json.dumps(io_data, ensure_ascii=False, indent=2))


def example_1_basic_usage():
    """示例1: 基本使用 - 2D预测器"""
    print_separator("示例1: 基本使用 - 2D预测器")
    
    # 1. 初始化算子
    config = {
        "type": "2D",
        "initial_position": [0.0, 0.0],
        "measurement_std": 0.1,
        "process_std": 0.5,
        "prediction_delays": [50, 200, 500]
    }
    
    operator = FPVPredictorOperator(config)
    print(f"✓ 算子已初始化: {operator.operator_id}")
    print(f"  类型: {operator.predictor_type}")
    print(f"  预测延迟: {operator.prediction_delays}")
    
    # 2. 发送位置更新（使用标准OperatorIO格式）
    print("\n--- 步骤1: 更新位置测量 ---")
    input_io = create_position_input(
        position=[1.5, 2.0],
        timestamp=0.05,
        source="camera_01",
        dimension="2D"
    )
    print_operator_io(input_io, "输入 OperatorIO")
    
    output_io = operator.process_input(input_io)
    print_operator_io(output_io, "输出 OperatorIO")
    
    # 3. 再次更新
    print("\n--- 步骤2: 继续更新位置 ---")
    input_io = create_position_input([3.0, 4.0], 0.10, "camera_01", "2D")
    output_io = operator.process_input(input_io)
    print(f"✓ 位置已更新: {output_io['control_info'].get('status')}")
    
    # 4. 执行预测
    print("\n--- 步骤3: 请求预测 ---")
    predict_io = create_predict_input(prediction_delays=[100, 500, 1000])
    result_io = operator.process_input(predict_io)
    print_operator_io(result_io, "预测结果 OperatorIO")
    
    # 解析预测结果
    if result_io['data_bodies']:
        predictions = result_io['data_bodies'][0].get('prediction', {}).get('predictions', [])
        print("\n预测结果汇总:")
        for pred in predictions:
            pos = pred['predicted_position']
            print(f"  {pred['delay_ms']:4}ms: 位置({pos[0]:.2f}, {pos[1]:.2f}) | "
                  f"准确率:{pred['accuracy_score']:.3f} | "
                  f"射击建议:{pred['fire_feasibility']:.3f}")
    
    # 5. 获取当前状态
    print("\n--- 步骤4: 获取状态估计 ---")
    state_io = {
        "metadata": {"io_id": "state_request", "data_type": "state_request", "shape": [], "source": "user", "timestamp": int(time.time()*1000), "ext": {}},
        "data_bodies": [],
        "control_info": {"op_action": "get_state"},
        "error": None
    }
    state_result = operator.process_input(state_io)
    
    if state_result['data_bodies']:
        state = state_result['data_bodies'][0].get('state', {})
        print(f"  当前位置: {state['position']}")
        print(f"  当前速度: {state['velocity']}")
        print(f"  速度模: {state['speed']:.2f} m/s")
        print(f"  不确定性: {state['uncertainty']:.6f}")


def example_2_3d_predictor():
    """示例2: 3D预测器使用"""
    print_separator("示例2: 3D预测器使用")
    
    # 初始化3D算子
    config = {
        "type": "3D",
        "initial_position": [0.0, 0.0, 10.0],
        "measurement_std": 0.15,
        "process_std": 0.6,
        "prediction_delays": [50, 200, 500, 1000]
    }
    
    operator = FPVPredictorOperator(config)
    print(f"✓ 3D算子已初始化: {operator.operator_id}")
    
    # 模拟一系列3D位置更新
    positions = [
        ([0.5, 0.2, 10.5], 0.05),
        ([1.2, 0.8, 11.0], 0.10),
        ([2.0, 1.5, 11.8], 0.15),
        ([2.8, 2.3, 12.5], 0.20),
    ]
    
    print("\n--- 处理3D位置序列 ---")
    for pos, ts in positions:
        input_io = create_position_input(pos, ts, "radar_01", "3D")
        output_io = operator.process_input(input_io)
        print(f"✓ t={ts:.2f}s: 位置 {pos} | 状态: {output_io['control_info'].get('status')}")
    
    # 获取最终状态和预测
    print("\n--- 最终状态估计 ---")
    state_io = {
        "metadata": {"io_id": "state_req", "data_type": "state_request", "shape": [], "source": "user", "timestamp": int(time.time()*1000), "ext": {}},
        "data_bodies": [],
        "control_info": {"op_action": "get_state"},
        "error": None
    }
    state_result = operator.process_input(state_io)
    
    if state_result['data_bodies']:
        state = state_result['data_bodies'][0].get('state', {})
        print(f"  位置: {state['position']}")
        print(f"  速度: {state['velocity']} (模: {state['speed']:.2f} m/s)")
        print(f"  加速度: {state['acceleration']} (模: {state['acceleration_magnitude']:.2f} m/s²)")
    
    # 预测
    print("\n--- 未来轨迹预测 ---")
    predict_io = create_predict_input([100, 500, 1000])
    result_io = operator.process_input(predict_io)
    
    if result_io['data_bodies']:
        predictions = result_io['data_bodies'][0].get('prediction', {}).get('predictions', [])
        for pred in predictions:
            pos = pred['predicted_position']
            fire_status = "✓ 适合" if pred['fire_feasibility'] > 0.6 else "✗ 不适合" if pred['fire_feasibility'] < 0.3 else "△ 一般"
            print(f"  {pred['delay_ms']:4}ms: ({pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}) | "
                  f"射击: {pred['fire_feasibility']:.3f} {fire_status}")


def example_3_error_handling():
    """示例3: 错误处理"""
    print_separator("示例3: 错误处理示例")
    
    # 初始化算子
    config = {
        "type": "2D",
        "initial_position": [0.0, 0.0],
        "measurement_std": 0.1,
        "process_std": 0.5
    }
    operator = FPVPredictorOperator(config)
    
    # 1. 维度不匹配错误
    print("\n--- 测试1: 维度不匹配 ---")
    input_io = create_position_input([1.0], 0.1, "sensor", "2D")  # 只有1个坐标
    result = operator.process_input(input_io)
    
    if result.get('error'):
        error = result['error']
        print(f"✗ 错误码: {error['code']}")
        print(f"  错误信息: {error['msg']}")
        print(f"  详细信息: {error['detail']}")
    
    # 2. 不支持的动作
    print("\n--- 测试2: 不支持的动作 ---")
    invalid_io = {
        "metadata": {"io_id": "test", "data_type": "test", "shape": [], "source": "test", "timestamp": int(time.time()*1000), "ext": {}},
        "data_bodies": [],
        "control_info": {"op_action": "invalid_action"},
        "error": None
    }
    result = operator.process_input(invalid_io)
    
    if result.get('error'):
        error = result['error']
        print(f"✗ 错误码: {error['code']}")
        print(f"  错误信息: {error['msg']}")
        print(f"  详细信息: {error['detail']}")


def example_4_json_serialization():
    """示例4: JSON序列化 - 跨语言交互"""
    print_separator("示例4: JSON序列化（跨语言交互）")
    
    config = {
        "type": "2D",
        "initial_position": [0.0, 0.0],
        "measurement_std": 0.1,
        "process_std": 0.5
    }
    operator = FPVPredictorOperator(config)
    
    # 创建输入
    input_io = create_position_input([1.0, 2.0], 0.05, "sensor_01", "2D")
    
    # 序列化为JSON（可以通过网络/消息队列传输）
    json_input = operator.to_json(input_io)
    print("\n--- JSON输入（可通过HTTP/gRPC/Kafka传输）---")
    print(json_input)
    
    # 模拟远程传输：反序列化
    print("\n--- 远程接收并处理 ---")
    received_io = operator.from_json(json_input)
    result_io = operator.process_input(received_io)
    
    # 序列化结果
    json_output = operator.to_json(result_io)
    print("\n--- JSON输出（返回给调用方）---")
    print(json_output)
    
    print("\n✓ 数据可以通过JSON在不同语言/服务之间传递")
    print("  支持的传输方式: HTTP REST API, gRPC, Kafka, RabbitMQ等")


def example_5_batch_processing():
    """示例5: 批处理模式"""
    print_separator("示例5: 批处理模式 - 处理历史数据")
    
    config = {
        "type": "2D",
        "initial_position": [0.0, 0.0],
        "measurement_std": 0.12,
        "process_std": 0.5
    }
    operator = FPVPredictorOperator(config)
    
    # 模拟从文件/数据库加载的历史轨迹数据
    trajectory_data = [
        {"position": [0.5, 0.3], "time": 0.05},
        {"position": [1.0, 0.8], "time": 0.10},
        {"position": [1.5, 1.5], "time": 0.15},
        {"position": [2.0, 2.3], "time": 0.20},
        {"position": [2.5, 3.2], "time": 0.25},
    ]
    
    print(f"\n处理 {len(trajectory_data)} 个历史数据点...")
    
    results = []
    for data_point in trajectory_data:
        # 更新
        update_io = create_update_only_input(data_point["position"], data_point["time"])
        operator.process_input(update_io)
        
        # 预测
        predict_io = create_predict_input([200, 500])
        result = operator.process_input(predict_io)
        results.append({
            "time": data_point["time"],
            "position": data_point["position"],
            "prediction": result['data_bodies'][0].get('prediction') if result['data_bodies'] else None
        })
    
    # 输出批处理结果
    print("\n批处理结果汇总:")
    for i, res in enumerate(results, 1):
        print(f"\n数据点 {i}: t={res['time']:.2f}s, 位置={res['position']}")
        if res['prediction']:
            preds = res['prediction']['predictions']
            for pred in preds:
                print(f"  → {pred['delay_ms']}ms预测: {pred['predicted_position']} "
                      f"(射击建议: {pred['fire_feasibility']:.3f})")


def main():
    """主函数"""
    print("="*80)
    print("FPV预测算子 - 标准化接口使用示例")
    print("="*80)
    print("\n本示例展示如何使用OperatorIO标准协议与FPV预测算子交互")
    print("适用场景: 微服务架构、算子热插拔、跨语言调用")
    
    try:
        example_1_basic_usage()
        example_2_3d_predictor()
        example_3_error_handling()
        example_4_json_serialization()
        example_5_batch_processing()
        
        print_separator("所有示例执行完成")
        print("\n✓ 算子接口已标准化")
        print("✓ 支持JSON/Protobuf跨语言调用")
        print("✓ 完善的错误处理机制")
        print("✓ 适配微服务/流式计算架构")
        
    except Exception as e:
        print(f"\n✗ 执行失败: {e}")
        print("提示: 请确保已安装依赖: pip install numpy filterpy scipy")


if __name__ == '__main__':
    main()

