"""
FPV 拦截预测算法 - 基于无损卡尔曼滤波 (UKF) 和运动模型

使用无损卡尔曼滤波器 (Unscented Kalman Filter, UKF) 和
恒定转弯率和速度 (Constant Turn Rate and Velocity, CTRV) 模型
来预测高度机动目标的运动轨迹。

支持 2D 和 3D 两种模式:
- 2D: CTRV 模型, 状态向量 [x, y, v, phi, dphi]
- 3D: 恒定加速度模型, 状态向量 [x, y, z, vx, vy, vz, ax, ay, az]

Author: pointfang@gmail.com
Date: 2025-10-15
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ========================== 2D CTRV 模型 ==========================
# --- 1. 定义非线性运动模型 (CTRV - Constant Turn Rate and Velocity) ---

def f_non_linear(x, dt):
    """
    状态转移函数 (非线性) - Constant Turn Rate and Velocity (CTRV) 模型
    
    参数:
        x: 状态向量 [x, y, v, phi, ddot_phi]T
            - x, y: 位置坐标
            - v: 速度
            - phi: 航向角
            - ddot_phi: 转弯率 (角速度)
        dt: 时间步长 (秒)
    
    返回:
        x_new: 预测的下一时刻状态向量
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
    测量函数 (线性) - 从状态向量提取可观测量
    
    参数:
        x: 状态向量 [x, y, v, phi, ddot_phi]T
    
    返回:
        测量向量 [x, y] (只测量位置坐标)
    """
    return np.array([x[0], x[1]])


# --- 2. 预测器类定义 ---

class FlyPredictor:
    def __init__(self, initial_pos, measurement_std=0.1, process_std=0.5):
        """
        初始化FPV预测器
        
        参数:
            initial_pos: 初始位置 [x, y] 或类似数组
            measurement_std: 测量噪声标准差 (默认 0.1)
            process_std: 过程噪声标准差 (默认 0.5)
        """
        # 输入验证
        initial_pos = np.array(initial_pos)
        if initial_pos.shape[0] < 2:
            raise ValueError("initial_pos 必须至少包含 [x, y] 两个坐标")
        if measurement_std <= 0 or process_std <= 0:
            raise ValueError("measurement_std 和 process_std 必须大于 0")
        
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
        """
        用新的测量值更新滤波器状态
        
        参数:
            current_pos: 当前测量位置 [x, y]
            current_time: 当前时间戳 (秒)
        """
        # 输入验证
        current_pos = np.array(current_pos)
        if current_pos.shape[0] < 2:
            raise ValueError("current_pos 必须至少包含 [x, y] 两个坐标")
        if current_time < self.last_time:
            raise ValueError(f"current_time ({current_time}) 不能小于 last_time ({self.last_time})")
        
        dt = current_time - self.last_time
        if dt > 0.001:  # 避免时间步长过小
            self.ukf.dt = dt
            self.ukf.predict()
            self.ukf.update(current_pos[:2])  # 只使用 x, y 坐标
            self.last_time = current_time
            self.last_pos = current_pos[:2]
        
    def predict_and_evaluate(self, delay_ms_list):
        """
        预测给定时间后的位置、准确率和射击建议
        
        参数:
            delay_ms_list: 延迟时间列表 (毫秒)，例如 [50, 200, 500, 1000]
        
        返回:
            字典，键为延迟时间(ms)，值包含:
                - predicted_position: 预测位置 [x, y]
                - accuracy_score: 准确率得分 (0-1, 越大越准确)
                - fire_feasibility: 射击可行性 (0-1, 越大越适合射击)
        """
        # 输入验证
        if not delay_ms_list or len(delay_ms_list) == 0:
            raise ValueError("delay_ms_list 不能为空")
        
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
    
    def get_current_state(self):
        """
        获取当前估计的状态信息
        
        返回:
            字典包含:
                - position: 当前估计位置 [x, y]
                - velocity: 当前估计速度
                - heading: 当前估计航向角 (弧度)
                - turn_rate: 当前估计转弯率 (弧度/秒)
                - uncertainty: 位置不确定性 (协方差矩阵行列式)
        """
        return {
            'position': self.ukf.x[:2].tolist(),
            'velocity': float(self.ukf.x[2]),
            'heading': float(self.ukf.x[3]),
            'turn_rate': float(self.ukf.x[4]),
            'uncertainty': float(np.linalg.det(self.ukf.P[:2, :2]))
        }
    
    def reset(self, initial_pos, initial_time=0.0):
        """
        重置预测器到新的初始状态
        
        参数:
            initial_pos: 新的初始位置 [x, y]
            initial_time: 新的初始时间 (默认 0.0)
        """
        initial_pos = np.array(initial_pos)
        if initial_pos.shape[0] < 2:
            raise ValueError("initial_pos 必须至少包含 [x, y] 两个坐标")
        
        # 重置状态向量
        self.ukf.x = np.array([initial_pos[0], initial_pos[1], 1.0, 0.0, 0.0])
        
        # 重置协方差矩阵
        self.ukf.P = np.diag([0.1**2, 0.1**2, 1.**2, np.deg2rad(10)**2, np.deg2rad(5)**2])
        
        # 重置时间和位置
        self.last_pos = initial_pos[:2]
        self.last_time = initial_time
    
    def adjust_noise_parameters(self, measurement_std=None, process_std=None):
        """
        动态调整噪声参数 (用于运行时优化)
        
        参数:
            measurement_std: 新的测量噪声标准差 (None 表示不改变)
            process_std: 新的过程噪声标准差 (None 表示不改变)
        """
        if measurement_std is not None:
            if measurement_std <= 0:
                raise ValueError("measurement_std 必须大于 0")
            self.ukf.R = np.diag([measurement_std**2, measurement_std**2])
        
        if process_std is not None:
            if process_std <= 0:
                raise ValueError("process_std 必须大于 0")
            dt = self.ukf.dt
            q_pos = (process_std * dt)**2
            self.ukf.Q = np.diag([q_pos, q_pos, 0.5**2, np.deg2rad(5)**2, np.deg2rad(10)**2])


# ========================== 3D 恒定加速度模型 ==========================

def f_3d_constant_acceleration(x, dt):
    """
    3D 状态转移函数 - 恒定加速度模型
    
    参数:
        x: 状态向量 [x, y, z, vx, vy, vz, ax, ay, az]T
            - x, y, z: 3D位置坐标
            - vx, vy, vz: 3D速度分量
            - ax, ay, az: 3D加速度分量
        dt: 时间步长 (秒)
    
    返回:
        x_new: 预测的下一时刻状态向量
        
    运动学方程:
        x_new = x + vx*dt + 0.5*ax*dt^2
        vx_new = vx + ax*dt
        ax_new = ax (假设加速度恒定)
    """
    x_new = x.copy()
    
    # 提取当前状态
    pos = x[0:3]      # [x, y, z]
    vel = x[3:6]      # [vx, vy, vz]
    acc = x[6:9]      # [ax, ay, az]
    
    # 更新位置: x_new = x + v*dt + 0.5*a*dt^2
    x_new[0:3] = pos + vel * dt + 0.5 * acc * dt**2
    
    # 更新速度: v_new = v + a*dt
    x_new[3:6] = vel + acc * dt
    
    # 加速度保持不变 (恒定加速度假设)
    # x_new[6:9] = acc
    
    return x_new


def h_3d(x):
    """
    3D 测量函数 - 从状态向量提取可观测量
    
    参数:
        x: 状态向量 [x, y, z, vx, vy, vz, ax, ay, az]T
    
    返回:
        测量向量 [x, y, z] (只测量3D位置坐标)
    """
    return np.array([x[0], x[1], x[2]])


class FlyPredictor3D:
    """3D FPV 预测器 - 使用恒定加速度模型"""
    
    def __init__(self, initial_pos, measurement_std=0.1, process_std=0.5):
        """
        初始化 3D FPV预测器
        
        参数:
            initial_pos: 初始位置 [x, y, z] 或类似数组
            measurement_std: 测量噪声标准差 (默认 0.1)
            process_std: 过程噪声标准差 (默认 0.5)
        """
        # 输入验证
        initial_pos = np.array(initial_pos)
        if initial_pos.shape[0] < 3:
            raise ValueError("initial_pos 必须包含 [x, y, z] 三个坐标")
        if measurement_std <= 0 or process_std <= 0:
            raise ValueError("measurement_std 和 process_std 必须大于 0")
        
        dim_x = 9  # 状态维度: [x, y, z, vx, vy, vz, ax, ay, az]
        dim_z = 3  # 测量维度: [x, y, z]
        dt = 0.05  # 初始时间步长 (50ms)
        
        # 1. Sigma Point 选择
        points = MerweScaledSigmaPoints(dim_x, alpha=.1, beta=2., kappa=1.)
        
        # 2. 初始化 UKF
        self.ukf = UKF(dim_x=dim_x, dim_z=dim_z, dt=dt,
                       fx=f_3d_constant_acceleration, hx=h_3d,
                       points=points)
        
        # 3. 初始化状态和协方差
        # 初始状态: x, y, z, vx, vy, vz, ax, ay, az
        # 假设初始速度和加速度都为0
        self.ukf.x = np.array([
            initial_pos[0], initial_pos[1], initial_pos[2],  # 位置
            0.0, 0.0, 0.0,  # 速度
            0.0, 0.0, 0.0   # 加速度
        ])
        
        # 初始协方差 P
        # 对位置信任度高，对速度和加速度信任度低
        self.ukf.P = np.diag([
            0.1**2, 0.1**2, 0.1**2,  # x, y, z 位置不确定性
            1.0**2, 1.0**2, 1.0**2,  # vx, vy, vz 速度不确定性
            2.0**2, 2.0**2, 2.0**2   # ax, ay, az 加速度不确定性
        ])
        
        # 过程噪声 Q
        q_pos = (process_std * dt)**2
        self.ukf.Q = np.diag([
            q_pos, q_pos, q_pos,     # 位置过程噪声
            0.5**2, 0.5**2, 0.5**2,  # 速度过程噪声
            1.0**2, 1.0**2, 1.0**2   # 加速度过程噪声
        ])
        
        # 测量噪声 R
        self.ukf.R = np.diag([measurement_std**2, measurement_std**2, measurement_std**2])
        
        self.last_pos = initial_pos[:3]
        self.last_time = 0.0
    
    def update(self, current_pos, current_time):
        """
        用新的测量值更新滤波器状态
        
        参数:
            current_pos: 当前测量位置 [x, y, z]
            current_time: 当前时间戳 (秒)
        """
        # 输入验证
        current_pos = np.array(current_pos)
        if current_pos.shape[0] < 3:
            raise ValueError("current_pos 必须包含 [x, y, z] 三个坐标")
        if current_time < self.last_time:
            raise ValueError(f"current_time ({current_time}) 不能小于 last_time ({self.last_time})")
        
        dt = current_time - self.last_time
        if dt > 0.001:  # 避免时间步长过小
            self.ukf.dt = dt
            self.ukf.predict()
            self.ukf.update(current_pos[:3])  # 只使用 x, y, z 坐标
            self.last_time = current_time
            self.last_pos = current_pos[:3]
    
    def predict_and_evaluate(self, delay_ms_list):
        """
        预测给定时间后的位置、准确率和射击建议
        
        参数:
            delay_ms_list: 延迟时间列表 (毫秒)，例如 [50, 200, 500, 1000]
        
        返回:
            字典，键为延迟时间(ms)，值包含:
                - predicted_position: 预测位置 [x, y, z]
                - accuracy_score: 准确率得分 (0-1, 越大越准确)
                - fire_feasibility: 射击可行性 (0-1, 越大越适合射击)
        """
        # 输入验证
        if not delay_ms_list or len(delay_ms_list) == 0:
            raise ValueError("delay_ms_list 不能为空")
        
        predictions = {}
        
        # 使用当前 UKF 状态作为起点进行预测
        initial_x = self.ukf.x.copy()
        
        # 计算当前的运动指标
        vel = initial_x[3:6]  # 速度向量 [vx, vy, vz]
        acc = initial_x[6:9]  # 加速度向量 [ax, ay, az]
        
        # 计算速度和加速度的模
        v_magnitude = np.linalg.norm(vel)
        a_magnitude = np.linalg.norm(acc)
        
        for delay_ms in delay_ms_list:
            dt_pred = delay_ms / 1000.0
            
            # 复制滤波器进行预测
            ukf_temp = self.ukf.copy()
            ukf_temp.dt = dt_pred
            ukf_temp.predict()
            
            # 预测位置 (x, y, z) 和其协方差
            predicted_pos = ukf_temp.x[:3]
            predicted_P = ukf_temp.P[:3, :3]  # 只关注 x, y, z 的协方差
            
            # 准确率评估
            det_P = np.linalg.det(predicted_P)
            accuracy_score = 1.0 / (1.0 + 100 * det_P)
            
            # 射击建议评估
            # 基于速度和加速度的运动复杂性
            # 假设最大速度 10 m/s，最大加速度 20 m/s^2 (考虑重力加速度)
            v_norm = np.clip(v_magnitude / 10.0, 0, 1)
            a_norm = np.clip(a_magnitude / 20.0, 0, 1)
            
            # 运动复杂性 = 0.6 * 速度 + 0.4 * 加速度
            motion_complexity = 0.6 * v_norm + 0.4 * a_norm
            fire_feasibility = np.clip(1.0 - motion_complexity, 0.0, 1.0)
            
            predictions[delay_ms] = {
                'predicted_position': predicted_pos.tolist(),
                'accuracy_score': accuracy_score,
                'fire_feasibility': fire_feasibility
            }
        
        return predictions
    
    def get_current_state(self):
        """
        获取当前估计的状态信息
        
        返回:
            字典包含:
                - position: 当前估计位置 [x, y, z]
                - velocity: 当前估计速度向量 [vx, vy, vz]
                - acceleration: 当前估计加速度向量 [ax, ay, az]
                - speed: 速度的模 (标量)
                - acceleration_magnitude: 加速度的模 (标量)
                - uncertainty: 位置不确定性 (协方差矩阵行列式)
        """
        vel = self.ukf.x[3:6]
        acc = self.ukf.x[6:9]
        
        return {
            'position': self.ukf.x[:3].tolist(),
            'velocity': vel.tolist(),
            'acceleration': acc.tolist(),
            'speed': float(np.linalg.norm(vel)),
            'acceleration_magnitude': float(np.linalg.norm(acc)),
            'uncertainty': float(np.linalg.det(self.ukf.P[:3, :3]))
        }
    
    def reset(self, initial_pos, initial_time=0.0):
        """
        重置预测器到新的初始状态
        
        参数:
            initial_pos: 新的初始位置 [x, y, z]
            initial_time: 新的初始时间 (默认 0.0)
        """
        initial_pos = np.array(initial_pos)
        if initial_pos.shape[0] < 3:
            raise ValueError("initial_pos 必须包含 [x, y, z] 三个坐标")
        
        # 重置状态向量
        self.ukf.x = np.array([
            initial_pos[0], initial_pos[1], initial_pos[2],
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        
        # 重置协方差矩阵
        self.ukf.P = np.diag([
            0.1**2, 0.1**2, 0.1**2,
            1.0**2, 1.0**2, 1.0**2,
            2.0**2, 2.0**2, 2.0**2
        ])
        
        # 重置时间和位置
        self.last_pos = initial_pos[:3]
        self.last_time = initial_time
    
    def adjust_noise_parameters(self, measurement_std=None, process_std=None):
        """
        动态调整噪声参数 (用于运行时优化)
        
        参数:
            measurement_std: 新的测量噪声标准差 (None 表示不改变)
            process_std: 新的过程噪声标准差 (None 表示不改变)
        """
        if measurement_std is not None:
            if measurement_std <= 0:
                raise ValueError("measurement_std 必须大于 0")
            self.ukf.R = np.diag([measurement_std**2, measurement_std**2, measurement_std**2])
        
        if process_std is not None:
            if process_std <= 0:
                raise ValueError("process_std 必须大于 0")
            dt = self.ukf.dt
            q_pos = (process_std * dt)**2
            self.ukf.Q = np.diag([
                q_pos, q_pos, q_pos,
                0.5**2, 0.5**2, 0.5**2,
                1.0**2, 1.0**2, 1.0**2
            ])
