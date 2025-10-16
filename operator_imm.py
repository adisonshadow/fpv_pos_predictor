"""
交互式多模型 (IMM - Interacting Multiple Model) 算法实现

IMM是跟踪高度机动目标的黄金标准，能够：
1. 同时运行多个运动模型（匀速、加速、转弯、悬停等）
2. 根据实时观测数据自动调整各模型的权重
3. 通过加权融合所有模型的预测结果，获得更高的准确率

Date: 2025-10-15
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from typing import List, Dict, Tuple, Optional


# ==================== 多种运动模型定义 ====================

def f_constant_velocity_2d(x, dt):
    """
    恒定速度模型 (CV - Constant Velocity) - 2D
    状态: [x, y, vx, vy]
    假设目标匀速直线运动
    """
    x_new = x.copy()
    x_new[0] = x[0] + x[2] * dt  # x += vx * dt
    x_new[1] = x[1] + x[3] * dt  # y += vy * dt
    # vx, vy 保持不变
    return x_new


def f_constant_acceleration_2d(x, dt):
    """
    恒定加速度模型 (CA - Constant Acceleration) - 2D
    状态: [x, y, vx, vy, ax, ay]
    假设目标以恒定加速度运动
    """
    x_new = x.copy()
    x_new[0] = x[0] + x[2] * dt + 0.5 * x[4] * dt**2  # x
    x_new[1] = x[1] + x[3] * dt + 0.5 * x[5] * dt**2  # y
    x_new[2] = x[2] + x[4] * dt  # vx
    x_new[3] = x[3] + x[5] * dt  # vy
    # ax, ay 保持不变
    return x_new


def f_coordinated_turn_2d(x, dt):
    """
    协调转弯模型 (CT - Coordinated Turn) - 2D
    状态: [x, y, v, phi, omega]
    假设目标以恒定角速度转弯
    """
    x_new = x.copy()
    
    x_pos, y_pos, v, phi, omega = x[0], x[1], x[2], x[3], x[4]
    
    if abs(omega) > 1e-5:  # 转弯
        x_new[0] = x_pos + (v / omega) * (np.sin(phi + omega * dt) - np.sin(phi))
        x_new[1] = y_pos + (v / omega) * (-np.cos(phi + omega * dt) + np.cos(phi))
        x_new[3] = phi + omega * dt
    else:  # 直线
        x_new[0] = x_pos + v * dt * np.cos(phi)
        x_new[1] = y_pos + v * dt * np.sin(phi)
    
    # v, omega 保持不变
    return x_new


def f_hovering_2d(x, dt):
    """
    悬停模型 (Hovering) - 2D
    状态: [x, y, vx, vy]
    假设目标基本静止，只有微小漂移
    """
    x_new = x.copy()
    # 位置几乎不变，只有噪声导致的微小移动
    x_new[0] = x[0] + x[2] * dt * 0.1  # 漂移很小
    x_new[1] = x[1] + x[3] * dt * 0.1
    # 速度衰减
    x_new[2] = x[2] * 0.9
    x_new[3] = x[3] * 0.9
    return x_new


def h_position_2d(x):
    """测量函数 - 只观测位置 [x, y]"""
    return np.array([x[0], x[1]])


# ==================== 3D 运动模型 ====================

def f_constant_velocity_3d(x, dt):
    """恒定速度模型 - 3D, 状态: [x, y, z, vx, vy, vz]"""
    x_new = x.copy()
    x_new[0:3] = x[0:3] + x[3:6] * dt
    return x_new


def f_constant_acceleration_3d(x, dt):
    """恒定加速度模型 - 3D, 状态: [x, y, z, vx, vy, vz, ax, ay, az]"""
    x_new = x.copy()
    x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * x[6:9] * dt**2
    x_new[3:6] = x[3:6] + x[6:9] * dt
    return x_new


def f_hovering_3d(x, dt):
    """悬停模型 - 3D, 状态: [x, y, z, vx, vy, vz]"""
    x_new = x.copy()
    x_new[0:3] = x[0:3] + x[3:6] * dt * 0.1
    x_new[3:6] = x[3:6] * 0.9
    return x_new


def h_position_3d(x):
    """测量函数 - 只观测3D位置"""
    return np.array([x[0], x[1], x[2]])


# ==================== IMM 滤波器实现 ====================

class ModelConfig:
    """单个模型的配置"""
    def __init__(self, name: str, dim_x: int, dim_z: int, fx, hx, 
                 initial_x: np.ndarray, P: np.ndarray, Q: np.ndarray):
        self.name = name
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.initial_x = initial_x
        self.P = P
        self.Q = Q


class IMMFilter:
    """交互式多模型滤波器"""
    
    def __init__(self, models: List[ModelConfig], measurement_noise: np.ndarray,
                 transition_matrix: np.ndarray, initial_probabilities: np.ndarray,
                 dt: float = 0.05):
        """
        初始化IMM滤波器
        
        参数:
            models: 模型配置列表
            measurement_noise: 测量噪声协方差矩阵 R
            transition_matrix: 模型转移矩阵 (Markov链)
            initial_probabilities: 各模型初始概率
            dt: 时间步长
        """
        self.n_models = len(models)
        self.models = models
        self.dt = dt
        
        # 模型转移概率矩阵 (Markov链)
        self.transition_matrix = transition_matrix
        
        # 当前模型概率
        self.mu = initial_probabilities.copy()
        
        # 为每个模型创建UKF
        self.filters = []
        for model in models:
            points = MerweScaledSigmaPoints(model.dim_x, alpha=.1, beta=2., kappa=1.)
            ukf = UKF(dim_x=model.dim_x, dim_z=model.dim_z, dt=dt,
                     fx=model.fx, hx=model.hx, points=points)
            ukf.x = model.initial_x.copy()
            ukf.P = model.P.copy()
            ukf.Q = model.Q.copy()
            ukf.R = measurement_noise.copy()
            self.filters.append(ukf)
        
        # 混合后的状态（用于下一次迭代）
        self.x_mixed = [f.x.copy() for f in self.filters]
        self.P_mixed = [f.P.copy() for f in self.filters]
        
        # 似然函数值
        self.likelihood = np.ones(self.n_models)
        
        self.last_time = 0.0
    
    def _mixing(self):
        """
        步骤1: 模型混合
        根据当前模型概率和转移矩阵，计算混合后的初始条件
        """
        # 计算混合概率 mu_ij
        mu_ij = np.zeros((self.n_models, self.n_models))
        c_j = np.zeros(self.n_models)
        
        for j in range(self.n_models):
            for i in range(self.n_models):
                mu_ij[i, j] = self.transition_matrix[i, j] * self.mu[i]
            c_j[j] = np.sum(mu_ij[:, j])
            if c_j[j] > 1e-10:
                mu_ij[:, j] /= c_j[j]
        
        # 计算混合后的状态和协方差
        for j in range(self.n_models):
            # 混合状态
            x_mixed = np.zeros_like(self.filters[j].x)
            for i in range(self.n_models):
                x_mixed += mu_ij[i, j] * self.filters[i].x
            
            # 混合协方差
            P_mixed = np.zeros_like(self.filters[j].P)
            for i in range(self.n_models):
                dx = self.filters[i].x - x_mixed
                P_mixed += mu_ij[i, j] * (self.filters[i].P + np.outer(dx, dx))
            
            self.x_mixed[j] = x_mixed
            self.P_mixed[j] = P_mixed
    
    def _model_filtering(self, z: np.ndarray):
        """
        步骤2: 模型滤波
        用混合后的初始条件对每个模型进行预测和更新
        """
        for j in range(self.n_models):
            # 使用混合后的状态作为初始条件
            self.filters[j].x = self.x_mixed[j].copy()
            self.filters[j].P = self.P_mixed[j].copy()
            
            # 预测
            self.filters[j].predict()
            
            # 更新
            self.filters[j].update(z)
            
            # 计算似然函数（观测概率）
            residual = z - self.filters[j].x[:len(z)]
            S = self.filters[j].P[:len(z), :len(z)] + self.filters[j].R
            
            # 多元高斯分布的概率密度
            try:
                S_inv = np.linalg.inv(S)
                det_S = np.linalg.det(S)
                if det_S > 0:
                    self.likelihood[j] = np.exp(-0.5 * residual @ S_inv @ residual) / \
                                        np.sqrt((2 * np.pi)**len(z) * det_S)
                else:
                    self.likelihood[j] = 1e-10
            except:
                self.likelihood[j] = 1e-10
    
    def _model_probability_update(self):
        """
        步骤3: 模型概率更新
        根据似然函数更新各模型的概率
        """
        # 计算归一化常数
        c = np.zeros(self.n_models)
        for j in range(self.n_models):
            c[j] = self.likelihood[j] * np.sum(self.transition_matrix[:, j] * self.mu)
        
        c_sum = np.sum(c)
        if c_sum > 1e-10:
            self.mu = c / c_sum
        else:
            # 如果所有似然都很小，均匀分布
            self.mu = np.ones(self.n_models) / self.n_models
    
    def _state_estimation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        步骤4: 状态融合
        根据模型概率加权融合各模型的估计
        """
        # 融合状态
        x_fused = np.zeros_like(self.filters[0].x)
        for j in range(self.n_models):
            x_fused += self.mu[j] * self.filters[j].x
        
        # 融合协方差
        P_fused = np.zeros_like(self.filters[0].P)
        for j in range(self.n_models):
            dx = self.filters[j].x - x_fused
            P_fused += self.mu[j] * (self.filters[j].P + np.outer(dx, dx))
        
        return x_fused, P_fused
    
    def update(self, z: np.ndarray, dt: Optional[float] = None):
        """
        IMM更新步骤
        
        参数:
            z: 测量值 (位置)
            dt: 时间步长（可选，使用初始化时的dt）
        """
        if dt is not None:
            self.dt = dt
            for f in self.filters:
                f.dt = dt
        
        # IMM四步法
        self._mixing()                    # 1. 模型混合
        self._model_filtering(z)          # 2. 模型滤波
        self._model_probability_update()  # 3. 概率更新
        # 状态融合在 get_state() 中按需计算
    
    def get_state(self) -> Dict:
        """获取融合后的状态估计"""
        x_fused, P_fused = self._state_estimation()
        
        return {
            'fused_state': x_fused,
            'fused_covariance': P_fused,
            'model_probabilities': self.mu.copy(),
            'active_model': self.models[np.argmax(self.mu)].name,
            'model_states': [f.x.copy() for f in self.filters]
        }
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        预测未来状态
        
        参数:
            steps: 预测步数
        
        返回:
            融合后的预测位置
        """
        # 对每个模型进行预测
        predictions = []
        for j in range(self.n_models):
            ukf_temp = self.filters[j].copy()
            for _ in range(steps):
                ukf_temp.predict()
            predictions.append(ukf_temp.x.copy())
        
        # 加权融合
        x_pred = np.zeros_like(predictions[0])
        for j in range(self.n_models):
            x_pred += self.mu[j] * predictions[j]
        
        return x_pred


# ==================== IMM 预测器封装 ====================

class IMMPredictor2D:
    """2D IMM预测器 - 多模型融合"""
    
    def __init__(self, initial_pos: np.ndarray, measurement_std: float = 0.1):
        """
        初始化2D IMM预测器
        
        模型:
            1. 恒定速度 (CV) - 匀速直线
            2. 恒定加速度 (CA) - 加速/减速
            3. 协调转弯 (CT) - 转弯机动
            4. 悬停 (Hover) - 低速/静止
        """
        # 定义4个模型
        models = []
        
        # 模型1: 恒定速度 (CV)
        models.append(ModelConfig(
            name="CV",
            dim_x=4, dim_z=2,
            fx=f_constant_velocity_2d,
            hx=h_position_2d,
            initial_x=np.array([initial_pos[0], initial_pos[1], 0.0, 0.0]),
            P=np.diag([0.1**2, 0.1**2, 1.0**2, 1.0**2]),
            Q=np.diag([0.1**2, 0.1**2, 0.5**2, 0.5**2])
        ))
        
        # 模型2: 恒定加速度 (CA)
        models.append(ModelConfig(
            name="CA",
            dim_x=6, dim_z=2,
            fx=f_constant_acceleration_2d,
            hx=h_position_2d,
            initial_x=np.array([initial_pos[0], initial_pos[1], 0.0, 0.0, 0.0, 0.0]),
            P=np.diag([0.1**2, 0.1**2, 1.0**2, 1.0**2, 2.0**2, 2.0**2]),
            Q=np.diag([0.1**2, 0.1**2, 0.5**2, 0.5**2, 1.0**2, 1.0**2])
        ))
        
        # 模型3: 协调转弯 (CT)
        models.append(ModelConfig(
            name="CT",
            dim_x=5, dim_z=2,
            fx=f_coordinated_turn_2d,
            hx=h_position_2d,
            initial_x=np.array([initial_pos[0], initial_pos[1], 1.0, 0.0, 0.0]),
            P=np.diag([0.1**2, 0.1**2, 1.0**2, np.deg2rad(10)**2, np.deg2rad(5)**2]),
            Q=np.diag([0.1**2, 0.1**2, 0.5**2, np.deg2rad(5)**2, np.deg2rad(10)**2])
        ))
        
        # 模型4: 悬停 (Hover)
        models.append(ModelConfig(
            name="Hover",
            dim_x=4, dim_z=2,
            fx=f_hovering_2d,
            hx=h_position_2d,
            initial_x=np.array([initial_pos[0], initial_pos[1], 0.0, 0.0]),
            P=np.diag([0.1**2, 0.1**2, 0.2**2, 0.2**2]),
            Q=np.diag([0.05**2, 0.05**2, 0.1**2, 0.1**2])
        ))
        
        # 模型转移矩阵 (Markov链) - 每行和为1
        # [CV, CA, CT, Hover]
        transition_matrix = np.array([
            [0.90, 0.05, 0.03, 0.02],  # 从CV转移
            [0.05, 0.88, 0.05, 0.02],  # 从CA转移
            [0.03, 0.05, 0.90, 0.02],  # 从CT转移
            [0.02, 0.02, 0.02, 0.94]   # 从Hover转移
        ])
        
        # 初始模型概率 - CV概率最高
        initial_probabilities = np.array([0.4, 0.3, 0.2, 0.1])
        
        # 测量噪声
        R = np.diag([measurement_std**2, measurement_std**2])
        
        # 创建IMM滤波器
        self.imm = IMMFilter(models, R, transition_matrix, initial_probabilities)
        self.last_time = 0.0
    
    def update(self, current_pos: np.ndarray, current_time: float):
        """更新IMM滤波器"""
        dt = current_time - self.last_time
        if dt > 0.001:
            self.imm.update(current_pos[:2], dt)
            self.last_time = current_time
    
    def predict_and_evaluate(self, delay_ms_list: List[int]) -> Dict:
        """预测并评估"""
        state_info = self.imm.get_state()
        x_fused = state_info['fused_state']
        P_fused = state_info['fused_covariance']
        
        predictions = {}
        for delay_ms in delay_ms_list:
            dt_pred = delay_ms / 1000.0
            steps = max(1, int(dt_pred / self.imm.dt))
            
            # 预测
            x_pred = self.imm.predict(steps)
            pred_pos = x_pred[:2]
            
            # 准确率评估
            P_pred = P_fused[:2, :2] * (1 + steps * 0.1)  # 简化：不确定性随时间增长
            det_P = np.linalg.det(P_pred)
            accuracy_score = 1.0 / (1.0 + 100 * det_P)
            
            # 射击建议 - 基于活跃模型
            active_model = state_info['active_model']
            model_probs = state_info['model_probabilities']
            
            # 悬停模型概率高 -> 适合射击
            # 加速/转弯模型概率高 -> 不适合射击
            fire_score = model_probs[0] * 0.8 + model_probs[3] * 0.9 - \
                        model_probs[1] * 0.3 - model_probs[2] * 0.5
            fire_feasibility = np.clip(fire_score, 0.0, 1.0)
            
            predictions[delay_ms] = {
                'predicted_position': pred_pos.tolist(),
                'accuracy_score': accuracy_score,
                'fire_feasibility': fire_feasibility,
                'active_model': active_model,
                'model_probabilities': {
                    'CV': float(model_probs[0]),
                    'CA': float(model_probs[1]),
                    'CT': float(model_probs[2]),
                    'Hover': float(model_probs[3])
                }
            }
        
        return predictions
    
    def get_current_state(self) -> Dict:
        """获取当前状态"""
        state_info = self.imm.get_state()
        x_fused = state_info['fused_state']
        
        return {
            'position': x_fused[:2].tolist(),
            'velocity': x_fused[2:4].tolist() if len(x_fused) >= 4 else [0.0, 0.0],
            'active_model': state_info['active_model'],
            'model_probabilities': state_info['model_probabilities'].tolist(),
            'uncertainty': float(np.linalg.det(state_info['fused_covariance'][:2, :2]))
        }


class IMMPredictor3D:
    """3D IMM预测器"""
    
    def __init__(self, initial_pos: np.ndarray, measurement_std: float = 0.15):
        """
        初始化3D IMM预测器
        
        模型:
            1. 恒定速度 (CV)
            2. 恒定加速度 (CA)
            3. 悬停 (Hover)
        """
        models = []
        
        # 模型1: 恒定速度
        models.append(ModelConfig(
            name="CV",
            dim_x=6, dim_z=3,
            fx=f_constant_velocity_3d,
            hx=h_position_3d,
            initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 
                               0.0, 0.0, 0.0]),
            P=np.diag([0.1**2, 0.1**2, 0.1**2, 1.0**2, 1.0**2, 1.0**2]),
            Q=np.diag([0.1**2, 0.1**2, 0.1**2, 0.5**2, 0.5**2, 0.5**2])
        ))
        
        # 模型2: 恒定加速度
        models.append(ModelConfig(
            name="CA",
            dim_x=9, dim_z=3,
            fx=f_constant_acceleration_3d,
            hx=h_position_3d,
            initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            P=np.diag([0.1**2, 0.1**2, 0.1**2, 1.0**2, 1.0**2, 1.0**2,
                      2.0**2, 2.0**2, 2.0**2]),
            Q=np.diag([0.1**2, 0.1**2, 0.1**2, 0.5**2, 0.5**2, 0.5**2,
                      1.0**2, 1.0**2, 1.0**2])
        ))
        
        # 模型3: 悬停
        models.append(ModelConfig(
            name="Hover",
            dim_x=6, dim_z=3,
            fx=f_hovering_3d,
            hx=h_position_3d,
            initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                               0.0, 0.0, 0.0]),
            P=np.diag([0.1**2, 0.1**2, 0.1**2, 0.2**2, 0.2**2, 0.2**2]),
            Q=np.diag([0.05**2, 0.05**2, 0.05**2, 0.1**2, 0.1**2, 0.1**2])
        ))
        
        # 模型转移矩阵
        transition_matrix = np.array([
            [0.92, 0.05, 0.03],
            [0.05, 0.90, 0.05],
            [0.03, 0.05, 0.92]
        ])
        
        initial_probabilities = np.array([0.5, 0.3, 0.2])
        
        R = np.diag([measurement_std**2, measurement_std**2, measurement_std**2])
        
        self.imm = IMMFilter(models, R, transition_matrix, initial_probabilities)
        self.last_time = 0.0
    
    def update(self, current_pos: np.ndarray, current_time: float):
        """更新IMM滤波器"""
        dt = current_time - self.last_time
        if dt > 0.001:
            self.imm.update(current_pos[:3], dt)
            self.last_time = current_time
    
    def predict_and_evaluate(self, delay_ms_list: List[int]) -> Dict:
        """预测并评估"""
        state_info = self.imm.get_state()
        x_fused = state_info['fused_state']
        P_fused = state_info['fused_covariance']
        
        predictions = {}
        for delay_ms in delay_ms_list:
            dt_pred = delay_ms / 1000.0
            steps = max(1, int(dt_pred / self.imm.dt))
            
            x_pred = self.imm.predict(steps)
            pred_pos = x_pred[:3]
            
            P_pred = P_fused[:3, :3] * (1 + steps * 0.1)
            det_P = np.linalg.det(P_pred)
            accuracy_score = 1.0 / (1.0 + 100 * det_P)
            
            model_probs = state_info['model_probabilities']
            fire_score = model_probs[0] * 0.8 + model_probs[2] * 0.9 - model_probs[1] * 0.3
            fire_feasibility = np.clip(fire_score, 0.0, 1.0)
            
            predictions[delay_ms] = {
                'predicted_position': pred_pos.tolist(),
                'accuracy_score': accuracy_score,
                'fire_feasibility': fire_feasibility,
                'active_model': state_info['active_model'],
                'model_probabilities': {
                    'CV': float(model_probs[0]),
                    'CA': float(model_probs[1]),
                    'Hover': float(model_probs[2])
                }
            }
        
        return predictions
    
    def get_current_state(self) -> Dict:
        """获取当前状态"""
        state_info = self.imm.get_state()
        x_fused = state_info['fused_state']
        
        vel = x_fused[3:6] if len(x_fused) >= 6 else np.array([0.0, 0.0, 0.0])
        
        return {
            'position': x_fused[:3].tolist(),
            'velocity': vel.tolist(),
            'speed': float(np.linalg.norm(vel)),
            'active_model': state_info['active_model'],
            'model_probabilities': state_info['model_probabilities'].tolist(),
            'uncertainty': float(np.linalg.det(state_info['fused_covariance'][:3, :3]))
        }

