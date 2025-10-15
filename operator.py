import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.spatial.distance import mahalanobis

# --- 1. 定义非线性运动模型 (CTRV - Constant Turn Rate and Velocity) ---

def f_non_linear(x, dt):
    """
    状态转移函数 (非线性) - Constant Turn Rate and Velocity (CTRV) 模型
    x = [x, y, v, phi, ddot_phi]T
    """
    x_new = x.copy()
    
    # 提取当前状态
    x, y, v, phi, ddot_phi = x[0], x[1], x[2], x[3], x[4]
    
    # 预测下一状态
    if abs(ddot_phi) > 1.0e-5: # 正在转弯
        x_new[0] = x + (v / ddot_phi) * (np.sin(phi + ddot_phi * dt) - np.sin(phi)) # x
        x_new[1] = y + (v / ddot_phi) * (-np.cos(phi + ddot_phi * dt) + np.cos(phi)) # y
        x_new[3] = phi + ddot_phi * dt # phi
    else: # 直线飞行 (ddot_phi ≈ 0)
        x_new[0] = x + v * dt * np.cos(phi) # x
        x_new[1] = y + v * dt * np.sin(phi) # y
        # x_new[3] = phi (航向不变)

    # v 和 ddot_phi 保持不变 (CTRV 假设)
    # x_new[2] = v 
    # x_new[4] = ddot_phi
    
    return x_new

def h_non_linear(x):
    """
    测量函数 (线性) - 假设我们只测量 (x, y) 坐标
    """
    return np.array([x[0], x[1]])


# --- 2. 预测器类定义 ---

class FlyPredictor:
    def __init__(self, initial_pos, measurement_std=0.1, process_std=0.5):
        
        dim_x = 5  # 状态维度: [x, y, v, phi, ddot_phi]
        dim_z = 2  # 测量维度: [x, y]
        dt = 0.05  # 初始时间步长 (50ms)

        # 1. Sigma Point 选择 (Merwe Scaled) - 适用于非线性
        # n: 状态维度, alpha: 分布扩散, beta: 偏离中心点的权重, kappa: 缩放因子
        points = MerweScaledSigmaPoints(dim_x, alpha=.1, beta=2., kappa=1.)

        # 2. 初始化 UKF
        self.ukf = UKF(dim_x=dim_x, dim_z=dim_z, dt=dt, 
                       fx=f_non_linear, hx=h_non_linear, 
                       points=points)
        
        # 3. 初始化状态和协方差
        # 初始状态: x, y, v, phi, ddot_phi (假设 v=1.0, phi=0.0, ddot_phi=0.0)
        self.ukf.x = np.array([initial_pos[0], initial_pos[1], 1.0, 0.0, 0.0]) 

        # 初始协方差 P (不确定性)
        # 对位置 (x, y) 信任度高 (小值)，对速度、航向、转弯率信任度低 (大值)
        self.ukf.P = np.diag([0.1**2, 0.1**2, 1.**2, np.deg2rad(10)**2, np.deg2rad(5)**2])

        # 过程噪声 Q (模型不确定性，高值适应剧烈机动)
        q_pos = (process_std * dt)**2  # 随着时间步长缩放
        self.ukf.Q = np.diag([q_pos, q_pos, 0.5**2, np.deg2rad(5)**2, np.deg2rad(10)**2])

        # 测量噪声 R (传感器精度)
        self.ukf.R = np.diag([measurement_std**2, measurement_std**2])

        self.last_pos = initial_pos
        self.last_time = 0.0
        
    def update(self, current_pos, current_time):
        """用新的测量值更新滤波器状态"""
        dt = current_time - self.last_time
        if dt > 0.001:
            self.ukf.dt = dt
            self.ukf.predict()
            self.ukf.update(current_pos)
            self.last_time = current_time
            self.last_pos = current_pos
        
    def predict_and_evaluate(self, delay_ms_list):
        """预测给定时间后的位置、准确率和射击建议"""
        
        predictions = {}
        
        # 使用当前 UKF 状态作为起点进行预测
        initial_x = self.ukf.x.copy()
        initial_P = self.ukf.P.copy()

        # 计算当前的运动指标
        v = initial_x[2] # 速度
        ddot_phi = initial_x[4] # 转弯率 (角速度)
        
        # 1. 预测准确率 (基于协方差矩阵 P)
        for delay_ms in delay_ms_list:
            dt_pred = delay_ms / 1000.0
            
            # 复制滤波器进行预测（不影响主滤波器的当前状态）
            ukf_temp = self.ukf.copy()
            ukf_temp.dt = dt_pred

            # 预测步骤
            ukf_temp.predict()
            
            # 预测位置 (x, y) 和其协方差 (不确定性)
            predicted_pos = ukf_temp.x[:2]
            predicted_P = ukf_temp.P[:2, :2] # 只关注 x, y 的协方差
            
            # 准确率/不确定性：使用位置协方差矩阵的行列式作为不确定性指标
            # 行列式越大，不确定性越大，准确率越低。
            # 准确率 score (0~1): 1 / (1 + det(P_xy))
            det_P = np.linalg.det(predicted_P)
            accuracy_score = 1.0 / (1.0 + 100 * det_P) # 100 是一个经验缩放因子

            # 2. 射击建议 (Fire Feasibility)
            # 射击建议基于两个核心指标：速度 V 和转弯率 ddot_phi
            # - 高速 (V > 2.0 m/s) -> 准确性下降 -> 不适合射击
            # - 剧烈转弯 (|ddot_phi| > 1.0 rad/s) -> 轨迹难以预测 -> 不适合射击
            
            # 归一化 V 和 ddot_phi (假设苍蝇最大速度 4 m/s，最大转弯率 5 rad/s)
            v_norm = np.clip(v / 4.0, 0, 1)
            ddot_phi_norm = np.clip(abs(ddot_phi) / 5.0, 0, 1)
            
            # 射击建议 score: 1 - 运动复杂性
            # 运动复杂性 = 0.5 * V_norm + 0.5 * ddot_phi_norm
            # 当运动复杂性接近 1 时，score 接近 0
            motion_complexity = 0.5 * v_norm + 0.5 * ddot_phi_norm
            fire_feasibility = np.clip(1.0 - motion_complexity, 0.0, 1.0)
            
            predictions[delay_ms] = {
                'predicted_position': predicted_pos.tolist(),
                'accuracy_score': accuracy_score,
                'fire_feasibility': fire_feasibility
            }

        return predictions



    print("\n--- 预测结束 ---")
