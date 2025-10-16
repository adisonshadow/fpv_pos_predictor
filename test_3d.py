"""
3D FPV 预测器测试脚本

模拟一个FPV在3D空间中的螺旋上升运动
"""

import numpy as np
from operator import FlyPredictor3D, f_3d_constant_acceleration

if __name__ == '__main__':
    print("=== 3D FPV 轨迹预测模拟 (基于 UKF 恒定加速度模型) ===\n")
    
    # 初始状态: [x, y, z, vx, vy, vz, ax, ay, az]
    x_init = np.array([
        0., 0., 10.,      # 初始位置 (x, y, z) - 从10米高度开始
        3.0, 0., 1.0,     # 初始速度 (vx, vy, vz) - 水平3m/s，垂直1m/s
        0., 0., 0.        # 初始加速度
    ])
    
    # 模拟3D螺旋上升轨迹
    sim_data = []
    T_end = 3.0  # 模拟 3 秒
    dt_sim = 0.05  # 50ms
    time = 0.0
    
    omega = 1.0  # 角速度 (rad/s) - 用于模拟螺旋运动
    
    for i in range(int(T_end / dt_sim)):
        # 模拟螺旋运动: 在 XY 平面旋转，同时沿 Z 轴上升
        if time > 0.5 and time < 2.5:
            # 在 XY 平面上做圆周运动，同时加速上升
            radius = 2.0
            x_init[3] = -radius * omega * np.sin(omega * time)  # vx
            x_init[4] = radius * omega * np.cos(omega * time)   # vy
            x_init[5] = 1.5  # vz - 向上速度
            x_init[8] = 0.5  # az - 向上微弱加速
        else:
            # 匀速直线运动
            x_init[8] = 0.0
        
        # 使用运动模型更新状态
        x_init = f_3d_constant_acceleration(x_init, dt_sim)
        
        # 模拟传感器噪声 (3D高斯噪声)
        measurement = x_init[:3] + np.random.randn(3) * 0.15
        sim_data.append((measurement, time))
        time += dt_sim
    
    # 初始化3D预测器
    predictor = FlyPredictor3D(
        initial_pos=sim_data[0][0], 
        measurement_std=0.15, 
        process_std=0.6
    )
    
    prediction_delays = [50, 200, 500, 1000]
    
    print(f"模拟场景: FPV螺旋上升运动")
    print(f"初始位置: ({sim_data[0][0][0]:.2f}, {sim_data[0][0][1]:.2f}, {sim_data[0][0][2]:.2f}) m")
    print(f"模拟时长: {T_end}s\n")
    
    # 循环处理传感器数据
    for i, (current_pos, current_time) in enumerate(sim_data):
        
        # 1. 更新 UKF 状态
        predictor.update(current_pos, current_time)
        
        # 2. 预测和评估
        results = predictor.predict_and_evaluate(prediction_delays)
        
        if i % 10 == 0:  # 每 500ms 打印一次结果
            print(f"\n{'='*80}")
            print(f"[时间: {current_time:.2f}s | 测量位置: ({current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}) m]")
            
            # 获取当前估计状态
            state = predictor.get_current_state()
            pos = state['position']
            vel = state['velocity']
            acc = state['acceleration']
            speed = state['speed']
            acc_mag = state['acceleration_magnitude']
            
            print(f"-> 估计位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m")
            print(f"-> 估计速度: ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) m/s | 速度模: {speed:.2f} m/s")
            print(f"-> 估计加速度: ({acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}) m/s² | 加速度模: {acc_mag:.2f} m/s²")
            print(f"-> 位置不确定性: {state['uncertainty']:.6f}")
            
            print(f"\n预测结果:")
            for delay_ms, res in results.items():
                pred_pos = res['predicted_position']
                acc_score = res['accuracy_score']
                fire = res['fire_feasibility']
                
                fire_status = "✓ 适合" if fire > 0.6 else "✗ 不适合" if fire < 0.3 else "△ 一般"
                
                print(f"  [{delay_ms:4}ms] 预测位置: ({pred_pos[0]:6.2f}, {pred_pos[1]:6.2f}, {pred_pos[2]:6.2f}) m | "
                      f"准确率: {acc_score:.3f} | 射击建议: {fire:.3f} {fire_status}")
    
    print("\n" + "="*80)
    print("--- 预测结束 ---")
    
    # 最终状态总结
    final_state = predictor.get_current_state()
    print(f"\n最终估计状态:")
    print(f"  位置: {final_state['position']}")
    print(f"  速度: {final_state['velocity']} (模: {final_state['speed']:.2f} m/s)")
    print(f"  加速度: {final_state['acceleration']} (模: {final_state['acceleration_magnitude']:.2f} m/s²)")

