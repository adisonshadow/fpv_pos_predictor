"""
使用模拟数据测试预测器

从mockdata目录加载JSON格式的模拟数据，测试2D和3D预测器的性能
"""

import json
import sys
from pathlib import Path

# 动态导入（处理numpy未安装的情况）
try:
    import numpy as np
    from operator import FlyPredictor, FlyPredictor3D
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("警告: numpy或filterpy未安装，无法运行预测器")
    print("请运行: pip install numpy filterpy scipy")


def load_mockdata(filepath):
    """加载模拟数据"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_2d_predictor(data, scenario_name):
    """测试2D预测器"""
    print(f"\n{'='*80}")
    print(f"测试场景: {scenario_name} (2D)")
    print(f"{'='*80}")
    
    # 初始化预测器
    first_point = data[0]
    predictor = FlyPredictor(
        initial_pos=first_point['measured_position'],
        measurement_std=0.12,
        process_std=0.5
    )
    
    prediction_delays = [50, 200, 500, 1000]
    
    # 处理数据
    for i, point in enumerate(data):
        current_pos = point['measured_position']
        current_time = point['time']
        true_pos = point['true_position']
        
        # 更新预测器
        predictor.update(current_pos, current_time)
        
        # 每隔几个点输出一次
        if i % 2 == 0 or i == len(data) - 1:
            # 预测
            results = predictor.predict_and_evaluate(prediction_delays)
            
            # 当前状态
            state = predictor.get_current_state()
            
            print(f"\n[t={current_time:.2f}s] 真实: ({true_pos[0]:.2f}, {true_pos[1]:.2f}) "
                  f"| 测量: ({current_pos[0]:.2f}, {current_pos[1]:.2f})")
            print(f"  估计位置: ({state['position'][0]:.2f}, {state['position'][1]:.2f})")
            print(f"  估计速度: {state['velocity'][0]:.2f}, {state['velocity'][1]:.2f} m/s "
                  f"(模: {np.sqrt(state['velocity'][0]**2 + state['velocity'][1]**2):.2f})")
            
            print(f"  预测结果:")
            for delay_ms, res in results.items():
                pred_pos = res['predicted_position']
                acc_score = res['accuracy_score']
                fire = res['fire_feasibility']
                status = "✓" if fire > 0.6 else "✗" if fire < 0.3 else "△"
                
                print(f"    [{delay_ms:4}ms] ({pred_pos[0]:6.2f}, {pred_pos[1]:6.2f}) "
                      f"| 准确率:{acc_score:.3f} | 射击:{fire:.3f} {status}")


def test_3d_predictor(data, scenario_name):
    """测试3D预测器"""
    print(f"\n{'='*80}")
    print(f"测试场景: {scenario_name} (3D)")
    print(f"{'='*80}")
    
    # 初始化预测器
    first_point = data[0]
    predictor = FlyPredictor3D(
        initial_pos=first_point['measured_position'],
        measurement_std=0.15,
        process_std=0.6
    )
    
    prediction_delays = [50, 200, 500, 1000]
    
    # 处理数据
    for i, point in enumerate(data):
        current_pos = point['measured_position']
        current_time = point['time']
        true_pos = point['true_position']
        
        # 更新预测器
        predictor.update(current_pos, current_time)
        
        # 每隔几个点输出一次
        if i % 2 == 0 or i == len(data) - 1:
            # 预测
            results = predictor.predict_and_evaluate(prediction_delays)
            
            # 当前状态
            state = predictor.get_current_state()
            
            print(f"\n[t={current_time:.2f}s] 真实: ({true_pos[0]:.2f}, {true_pos[1]:.2f}, {true_pos[2]:.2f}) "
                  f"| 测量: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f})")
            print(f"  估计位置: ({state['position'][0]:.2f}, {state['position'][1]:.2f}, {state['position'][2]:.2f})")
            print(f"  估计速度: ({state['velocity'][0]:.2f}, {state['velocity'][1]:.2f}, {state['velocity'][2]:.2f}) "
                  f"| 速度模: {state['speed']:.2f} m/s")
            print(f"  估计加速度: ({state['acceleration'][0]:.2f}, {state['acceleration'][1]:.2f}, {state['acceleration'][2]:.2f}) "
                  f"| 加速度模: {state['acceleration_magnitude']:.2f} m/s²")
            
            print(f"  预测结果:")
            for delay_ms, res in results.items():
                pred_pos = res['predicted_position']
                acc_score = res['accuracy_score']
                fire = res['fire_feasibility']
                status = "✓ 适合" if fire > 0.6 else "✗ 不适合" if fire < 0.3 else "△ 一般"
                
                print(f"    [{delay_ms:4}ms] ({pred_pos[0]:6.2f}, {pred_pos[1]:6.2f}, {pred_pos[2]:6.2f}) "
                      f"| 准确率:{acc_score:.3f} | 射击:{fire:.3f} {status}")


def main():
    """主函数"""
    if not NUMPY_AVAILABLE:
        sys.exit(1)
    
    print("="*80)
    print("FPV 预测器 - 模拟数据测试")
    print("="*80)
    
    # 加载索引
    index_file = Path('mockdata/index.json')
    if not index_file.exists():
        print(f"错误: 找不到 {index_file}")
        print("请先运行 generate_mockdata.py 生成模拟数据")
        sys.exit(1)
    
    with open(index_file, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    print(f"\n找到 {index['total_scenarios']} 个测试场景")
    print(f"采样率: {index['sampling_rate']}")
    print(f"测量噪声: {index['measurement_noise_std']}")
    
    # 测试每个场景
    for scenario in index['scenarios']:
        filepath = Path(scenario['file'])
        if not filepath.exists():
            print(f"\n跳过: {scenario['name']} (文件不存在: {filepath})")
            continue
        
        # 加载数据
        data = load_mockdata(filepath)
        
        # 根据维度选择测试函数
        if scenario['dimension'] == '2D':
            test_2d_predictor(data, scenario['name'])
        elif scenario['dimension'] == '3D':
            test_3d_predictor(data, scenario['name'])
    
    print("\n" + "="*80)
    print("测试完成!")
    print("="*80)


if __name__ == '__main__':
    main()

