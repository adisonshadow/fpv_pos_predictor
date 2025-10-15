if __name__ == '__main__':
    # 模拟传感器数据 (假设以 50ms 间隔接收数据)
    
    # 场景 1: 苍蝇做圆周转弯
    # 状态: [x, y, v, phi, ddot_phi]
    
    # 初始状态
    x_init = np.array([0., 0., 2.5, 0.0, 0.0]) 
    
    # 运动轨迹 (转弯)
    sim_data = []
    T_end = 2.0 # 模拟 2 秒
    dt_sim = 0.05
    time = 0.0
    
    for i in range(int(T_end / dt_sim)):
        # 模拟剧烈转弯 (3 rad/s)
        if time > 0.5 and time < 1.5:
            x_init[4] = 3.0 # ddot_phi = 3.0 rad/s
        else:
            x_init[4] = 0.0
        
        x_init = f_non_linear(x_init, dt_sim)
        # 模拟传感器噪声 (高斯分布)
        measurement = x_init[:2] + np.random.randn(2) * 0.1 
        sim_data.append((measurement, time))
        time += dt_sim

    # 初始化预测器
    predictor = FlyPredictor(initial_pos=sim_data[0][0], measurement_std=0.1, process_std=0.5)

    prediction_delays = [50, 200, 500, 1000]

    print("--- 苍蝇轨迹预测模拟 (基于 UKF) ---")
    print(f"运动模型: CTRV (恒定转弯率和速度)\n")

    # 循环处理传感器数据
    for i, (current_pos, current_time) in enumerate(sim_data):
        
        # 1. 更新 UKF 状态
        predictor.update(current_pos, current_time)
        
        # 2. 预测和评估
        results = predictor.predict_and_evaluate(prediction_delays)
        
        if i % 5 == 0: # 每 250ms 打印一次结果
            print(f"\n[时间: {current_time:.2f}s | 当前位置: {current_pos[0]:.2f}, {current_pos[1]:.2f}]")
            
            v_curr = predictor.ukf.x[2]
            ddot_phi_curr = predictor.ukf.x[4]
            print(f"-> 估计状态: 速度={v_curr:.2f} m/s, 转弯率={np.rad2deg(ddot_phi_curr):.1f} °/s")

            for delay_ms, res in results.items():
                pos = res['predicted_position']
                acc = res['accuracy_score']
                fire = res['fire_feasibility']
                
                print(f"  > {delay_ms:4}ms 预测: Pos=({pos[0]:.2f}, {pos[1]:.2f}) | 准确率={acc:.2f} | 射击建议={fire:.2f} ({'YES' if fire > 0.5 else 'NO '})")
