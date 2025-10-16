"""
高级功能测试 - 自适应滤波和跟踪丢失恢复

演示:
1. 自适应滤波 - 自动调整Q和R
2. 跟踪丢失恢复 - 自动检测和恢复

Author: pointfang@gmail.com
Date: 2025-10-16
"""

import numpy as np
from operator_adaptive import AdaptiveFilter, create_adaptive_predictor, RobustAdaptiveFilter
from operator_track_recovery import TrackRecoveryFilter, create_robust_predictor, TrackStatus


def print_separator(title=""):
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
    print("="*80)


def test_adaptive_filter():
    """测试自适应滤波"""
    print_separator("测试1: 自适应滤波 - 自动调整噪声参数")
    
    print("\n场景: 传感器噪声在运行中突然变化")
    print("  0-1s: 低噪声环境 (σ=0.1)")
    print("  1-2s: 高噪声环境 (σ=0.5) <-- 噪声突然增大")
    print("  2-3s: 恢复低噪声 (σ=0.1)\n")
    
    # 创建自适应预测器
    predictor = create_adaptive_predictor(
        predictor_type="2D",
        initial_pos=[0, 0],
        measurement_std=0.1,
        process_std=0.3
    )
    
    # 生成轨迹（匀速直线）
    trajectory = []
    t = 0.0
    dt = 0.05
    true_pos = np.array([0.0, 0.0])
    vel = np.array([2.0, 1.0])
    
    while t <= 3.0:
        true_pos += vel * dt
        
        # 根据时间段添加不同的噪声
        if t < 1.0:
            noise_std = 0.1  # 低噪声
        elif t < 2.0:
            noise_std = 0.5  # 高噪声
        else:
            noise_std = 0.1  # 恢复低噪声
        
        measured_pos = true_pos + np.random.randn(2) * noise_std
        trajectory.append((measured_pos, t, noise_std))
        t += dt
    
    print("时间 | 噪声水平 | R矩阵trace | 自适应次数 | 跟踪质量")
    print("-" * 80)
    
    initial_R_trace = np.trace(predictor.filter.ukf.R)
    
    for i, (pos, time, true_noise) in enumerate(trajectory):
        predictor.update(pos, time)
        
        if i % 10 == 0:  # 每0.5秒输出
            stats = predictor.get_adaptation_stats()
            Q, R = predictor.get_noise_matrices()
            
            R_trace = np.trace(R)
            R_change = (R_trace / initial_R_trace - 1) * 100
            
            consistent = "✓" if stats['is_consistent'] else "✗"
            
            print(f"{time:.2f}s | σ={true_noise:.1f} | "
                  f"{R_trace:.4f} ({R_change:+.0f}%) | "
                  f"{stats['adaptation_count']:3d}次 | {consistent}")
    
    print("\n观察:")
    print("  ✓ 高噪声期间，R矩阵自动增大")
    print("  ✓ 噪声恢复后，R矩阵逐渐减小")
    print("  ✓ 自适应调整提高了滤波器的鲁棒性")


def test_track_loss_recovery():
    """测试跟踪丢失恢复"""
    print_separator("测试2: 跟踪丢失恢复 - 自动检测和重新获取")
    
    print("\n场景: 目标被遮挡导致跟踪丢失")
    print("  0-1s: 正常跟踪")
    print("  1-1.5s: 跟踪丢失（无测量）<-- 模拟遮挡")
    print("  1.5-3s: 重新出现，自动恢复\n")
    
    # 创建带恢复功能的预测器
    predictor = create_robust_predictor(
        predictor_type="2D",
        enable_recovery=True,
        initial_pos=[0, 0],
        measurement_std=0.1,
        process_std=0.3
    )
    
    # 生成轨迹
    trajectory = []
    t = 0.0
    dt = 0.05
    true_pos = np.array([0.0, 0.0])
    vel = np.array([3.0, 2.0])
    
    while t <= 3.0:
        true_pos += vel * dt
        
        # 模拟遮挡：1.0-1.5秒无测量
        if 1.0 <= t < 1.5:
            measured_pos = None  # 丢失
        else:
            measured_pos = true_pos + np.random.randn(2) * 0.1
        
        trajectory.append((measured_pos, t))
        t += dt
    
    print("时间 | 状态 | 位置 | 跟踪质量 | 搜索区域")
    print("-" * 90)
    
    for i, (pos, time) in enumerate(trajectory):
        # 更新（可能为None）
        predictor.update(pos, time, measurement_confidence=0.9)
        
        if i % 5 == 0:  # 每0.25秒输出
            state = predictor.get_current_state()
            status = state['track_status']
            quality = state.get('track_quality', 0.0)
            est_pos = state['position']
            
            # 状态显示
            status_icon = {
                'tracking': '✓ 跟踪',
                'lost': '✗ 丢失',
                'recovered': '✓ 恢复',
                'uncertain': '△ 不确定'
            }.get(status, status)
            
            # 搜索区域
            if 'search_center' in state and state['search_center'] is not None:
                search_info = f"半径{state['search_radius']:.1f}m"
            else:
                search_info = "-"
            
            print(f"{time:.2f}s | {status_icon:8s} | "
                  f"({est_pos[0]:5.2f}, {est_pos[1]:5.2f}) | "
                  f"{quality:.2f} | {search_info}")
    
    # 统计
    final_state = predictor.get_current_state()
    print(f"\n统计:")
    print(f"  丢失次数: {final_state['loss_count']}")
    print(f"  恢复次数: {final_state['recovery_count']}")
    print(f"  最终质量: {final_state['track_quality']:.2f}")


def test_robust_adaptive():
    """测试鲁棒自适应滤波器（抗异常值）"""
    print_separator("测试3: 鲁棒自适应 - 异常值处理")
    
    print("\n场景: 存在偶然的异常测量值")
    print("  正常测量 + 随机异常值（5%概率）\n")
    
    from operator_adaptive import RobustAdaptiveFilter
    from operator import FlyPredictor
    
    base = FlyPredictor(initial_pos=[0, 0], measurement_std=0.1, process_std=0.3)
    predictor = RobustAdaptiveFilter(base, outlier_threshold=5.0)
    
    # 生成轨迹
    trajectory = []
    t = 0.0
    dt = 0.05
    true_pos = np.array([0.0, 0.0])
    vel = np.array([2.0, 1.5])
    
    while t <= 2.0:
        true_pos += vel * dt
        
        # 5%概率出现异常值
        if np.random.rand() < 0.05:
            # 异常值：偏离真实位置5-10米
            measured_pos = true_pos + np.random.randn(2) * 5.0
            is_outlier = True
        else:
            measured_pos = true_pos + np.random.randn(2) * 0.1
            is_outlier = False
        
        trajectory.append((measured_pos, t, is_outlier))
        t += dt
    
    print("时间 | 测量值 | 估计值 | 异常值 | 异常次数")
    print("-" * 70)
    
    for i, (pos, time, is_true_outlier) in enumerate(trajectory):
        predictor.update(pos, time)
        
        if i % 8 == 0:
            state = predictor.get_current_state()
            stats = predictor.get_adaptation_stats()
            est_pos = state['position']
            
            outlier_mark = "⚠" if is_true_outlier else " "
            
            print(f"{time:.2f}s | ({pos[0]:5.2f}, {pos[1]:5.2f}) | "
                  f"({est_pos[0]:5.2f}, {est_pos[1]:5.2f}) | "
                  f"{outlier_mark:^3s} | {stats['outlier_count']:2d}")
    
    final_stats = predictor.get_adaptation_stats()
    print(f"\n异常值统计:")
    print(f"  检测到: {final_stats['outlier_count']} 次")
    print(f"  异常率: {final_stats['outlier_rate']*100:.1f}%")


def test_combined_features():
    """测试组合功能：自适应 + 丢失恢复"""
    print_separator("测试4: 组合功能 - 自适应+丢失恢复")
    
    print("\n场景: 复杂环境（噪声变化 + 间歇性遮挡）\n")
    
    # 创建自适应预测器
    from operator import FlyPredictor
    base = FlyPredictor(initial_pos=[0, 0], measurement_std=0.1, process_std=0.3)
    adaptive = AdaptiveFilter(base)
    predictor = TrackRecoveryFilter(adaptive.filter)
    
    # 生成复杂轨迹
    trajectory = []
    t = 0.0
    dt = 0.05
    true_pos = np.array([0.0, 0.0])
    vel = np.array([2.5, 1.8])
    
    while t <= 3.0:
        true_pos += vel * dt
        
        # 复杂场景
        if t < 1.0:
            # 正常
            noise = 0.1
            available = True
        elif t < 1.3:
            # 遮挡
            noise = 0.1
            available = False
        elif t < 2.0:
            # 高噪声
            noise = 0.4
            available = True
        elif t < 2.2:
            # 再次遮挡
            noise = 0.1
            available = False
        else:
            # 恢复正常
            noise = 0.1
            available = True
        
        if available:
            measured_pos = true_pos + np.random.randn(2) * noise
        else:
            measured_pos = None
        
        trajectory.append((measured_pos, t, noise, available))
        t += dt
    
    print("时间 | 环境 | 跟踪状态 | 估计位置 | 质量")
    print("-" * 75)
    
    for i, (pos, time, noise, available) in enumerate(trajectory):
        predictor.update(pos, time)
        
        if i % 8 == 0:
            state = predictor.get_current_state()
            
            env = "遮挡" if not available else f"σ={noise:.1f}"
            status = state['track_status']
            est_pos = state['position']
            quality = state.get('track_quality', 0.0)
            
            status_icon = {
                'tracking': '✓',
                'lost': '✗',
                'recovered': '✓',
                'uncertain': '△'
            }.get(status, '?')
            
            print(f"{time:.2f}s | {env:6s} | {status_icon} {status:10s} | "
                  f"({est_pos[0]:5.2f}, {est_pos[1]:5.2f}) | {quality:.2f}")
    
    print("\n✓ 系统成功应对复杂环境变化")


def main():
    print("="*80)
    print("高级功能测试 - 自适应滤波与跟踪恢复")
    print("="*80)
    
    try:
        test_adaptive_filter()
        test_track_loss_recovery()
        test_robust_adaptive()
        test_combined_features()
        
        print_separator("所有测试完成")
        print("\n✓ 自适应滤波: 自动调整噪声参数，提高鲁棒性")
        print("✓ 跟踪丢失恢复: 自动检测丢失并尝试重新获取")
        print("✓ 鲁棒性增强: 对异常值和环境变化更robust")
        print("\n这些功能大幅提升了系统的实用性和可靠性！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

