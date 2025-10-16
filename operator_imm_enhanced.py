"""
增强版IMM - 按运动维度分类的FPV专用多模型

基于FPV飞行特征的运动维度分类：
1. 垂直方向 - 俯冲、爬升、悬停
2. 水平方向 - 匀速/加速、协调转弯、侧飞
3. 姿态旋转 - 盘旋、横滚机动、自旋、半滚倒转
4. 速度突变 - 急刹

状态向量扩展: [x, y, z, vx, vy, vz, ax, ay, az, roll, pitch, yaw, ωx, ωy, ωz]

Author: pointfang@gmail.com
Date: 2025-10-16
"""

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from typing import List, Dict, Tuple
from operator_imm import ModelConfig, IMMFilter


# ==================== 运动维度1: 垂直方向 ====================

def f_vertical_dive(x, dt):
    """
    垂直俯冲模型
    状态: [x, y, z, vx, vy, vz, ax, ay, az]
    特征: 高垂直加速度，vz < 0（向下）
    """
    x_new = x.copy()
    x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * x[6:9] * dt**2
    x_new[3:6] = x[3:6] + x[6:9] * dt
    
    # 俯冲特征：az向下加速
    # 实际应用中可添加约束: az应该 < -2.0 m/s²
    return x_new


def f_vertical_climb(x, dt):
    """
    垂直爬升模型
    状态: [x, y, z, vx, vy, vz, ax, ay, az]
    特征: 正垂直速度和加速度，vz > 0（向上）
    """
    x_new = x.copy()
    x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * x[6:9] * dt**2
    x_new[3:6] = x[3:6] + x[6:9] * dt
    return x_new


def f_hover_vertical(x, dt):
    """
    垂直悬停模型
    状态: [x, y, z, vx, vy, vz]
    特征: vz ≈ 0，微小垂直漂移
    """
    x_new = x.copy()
    x_new[0:3] = x[0:3] + x[3:6] * dt * 0.05  # 极小漂移
    x_new[3:6] = x[3:6] * 0.85  # 速度快速衰减
    return x_new


# ==================== 运动维度2: 水平方向 ====================

def f_horizontal_acceleration(x, dt):
    """
    水平加速模型
    状态: [x, y, z, vx, vy, vz, ax, ay, az]
    特征: 水平面(xy)加速度显著，az ≈ 0
    """
    x_new = x.copy()
    x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * x[6:9] * dt**2
    x_new[3:6] = x[3:6] + x[6:9] * dt
    
    # 水平加速特征：保持高度相对稳定
    x_new[8] = x[8] * 0.9  # az衰减
    return x_new


def f_coordinated_turn_3d(x, dt):
    """
    3D协调转弯模型
    状态: [x, y, z, v, phi, theta, omega]
    - v: 速度模
    - phi: 航向角(yaw)
    - theta: 俯仰角(pitch)  
    - omega: 角速度
    """
    x_new = x.copy()
    
    v, phi, theta, omega = x[3], x[4], x[5], x[6]
    
    # 水平分量
    if abs(omega) > 1e-5:
        # 转弯
        x_new[0] = x[0] + (v / omega) * np.cos(theta) * (np.sin(phi + omega * dt) - np.sin(phi))
        x_new[1] = x[1] + (v / omega) * np.cos(theta) * (-np.cos(phi + omega * dt) + np.cos(phi))
        x_new[4] = phi + omega * dt
    else:
        # 直线
        x_new[0] = x[0] + v * np.cos(theta) * np.cos(phi) * dt
        x_new[1] = x[1] + v * np.cos(theta) * np.sin(phi) * dt
    
    # 垂直分量
    x_new[2] = x[2] + v * np.sin(theta) * dt
    
    return x_new


def f_sideways_flight(x, dt):
    """
    侧飞模型
    状态: [x, y, z, vx, vy, vz, heading]
    特征: 速度方向与heading不一致（侧向速度分量显著）
    """
    x_new = x.copy()
    
    # 位置更新（允许侧向速度）
    x_new[0:3] = x[0:3] + x[3:6] * dt
    
    # 速度保持（侧飞时速度方向相对heading有偏移）
    # heading保持或缓慢变化
    
    return x_new


# ==================== 运动维度3: 姿态旋转 ====================

def f_roll_maneuver(x, dt):
    """
    横滚机动模型
    状态: [x, y, z, vx, vy, vz, roll, ωx]
    特征: 快速roll角变化，ωx显著
    """
    x_new = x.copy()
    
    # 位置更新
    x_new[0:3] = x[0:3] + x[3:6] * dt
    
    # roll角更新
    roll, omega_x = x[6], x[7]
    x_new[6] = roll + omega_x * dt
    
    # 横滚机动会影响速度方向（重力分量）
    # 简化处理：速度在横滚时保持相对机体坐标系不变
    
    return x_new


def f_spin_on_spot(x, dt):
    """
    原地自旋模型
    状态: [x, y, z, yaw, ωz]
    特征: 位置几乎不变，yaw快速变化，ωz显著
    """
    x_new = x.copy()
    
    # 位置微小漂移
    x_new[0:3] = x[0:3] + np.random.randn(3) * 0.01 * dt
    
    # yaw角快速旋转
    yaw, omega_z = x[3], x[4]
    x_new[3] = yaw + omega_z * dt
    
    # 自旋时速度很小
    if len(x) > 5:
        x_new[5:8] = x[5:8] * 0.8  # 速度衰减
    
    return x_new


def f_half_roll_inversion(x, dt):
    """
    半滚倒转模型
    状态: [x, y, z, vx, vy, vz, roll, pitch]
    特征: roll从0→180°或180°→0°，同时pitch反转
    """
    x_new = x.copy()
    
    # 位置更新
    x_new[0:3] = x[0:3] + x[3:6] * dt
    
    # roll角快速变化（典型：π rad/s）
    roll_rate = np.pi  # rad/s
    x_new[6] = x[6] + roll_rate * dt
    
    # pitch同时调整
    x_new[7] = -x[7] * (1 - dt)  # 逐渐反转
    
    return x_new


# ==================== 运动维度4: 速度突变 ====================

def f_emergency_brake(x, dt):
    """
    急刹模型
    状态: [x, y, z, vx, vy, vz, ax, ay, az]
    特征: 高负加速度（与速度方向相反），速度快速降为0
    """
    x_new = x.copy()
    
    vel = x[3:6]
    speed = np.linalg.norm(vel)
    
    if speed > 0.1:
        # 制动加速度：方向与速度相反
        brake_magnitude = 15.0  # m/s² (很大的制动)
        brake_acc = -(vel / speed) * brake_magnitude
        
        x_new[3:6] = vel + brake_acc * dt
        x_new[0:3] = x[0:3] + vel * dt + 0.5 * brake_acc * dt**2
        
        # 如果速度已经很小，直接设为0
        if np.linalg.norm(x_new[3:6]) < 0.1:
            x_new[3:6] = 0
    else:
        # 已经停止
        x_new[3:6] = 0
        x_new[0:3] = x[0:3]
    
    return x_new


def f_burst_acceleration(x, dt):
    """
    爆发加速模型
    状态: [x, y, z, vx, vy, vz, ax, ay, az]
    特征: 瞬时大加速度（冲刺）
    """
    x_new = x.copy()
    
    # 高加速度
    acc = x[6:9]
    x_new[3:6] = x[3:6] + acc * dt
    x_new[0:3] = x[0:3] + x[3:6] * dt + 0.5 * acc * dt**2
    
    # 加速度在爆发后逐渐衰减
    x_new[6:9] = acc * 0.95
    
    return x_new


# ==================== 测量函数 ====================

def h_3d_full(x):
    """完整3D测量（位置+姿态）"""
    return np.array([x[0], x[1], x[2]])


def h_3d_with_attitude(x):
    """3D位置+姿态测量"""
    # 如果有IMU数据，可以测量roll, pitch, yaw
    if len(x) >= 9:
        return np.array([x[0], x[1], x[2], x[6], x[7]])  # pos + roll + pitch
    return np.array([x[0], x[1], x[2]])


# ==================== 增强版IMM预测器 ====================

class IMMPredictorEnhanced3D:
    """
    增强版3D IMM预测器 - 按运动维度分类
    
    包含11个模型，覆盖FPV所有典型机动：
    
    垂直维度(3个):
      1. 垂直俯冲 (Dive)
      2. 垂直爬升 (Climb)
      3. 垂直悬停 (Hover-V)
    
    水平维度(4个):
      4. 水平匀速 (CV)
      5. 水平加速 (CA)
      6. 协调转弯 (CT)
      7. 侧飞 (Sideways)
    
    姿态旋转(3个):
      8. 横滚机动 (Roll)
      9. 原地自旋 (Spin)
     10. 半滚倒转 (Half-Roll)
    
    速度突变(1个):
     11. 急刹 (Brake)
    """
    
    def __init__(self, initial_pos: np.ndarray, measurement_std: float = 0.15,
                 model_selection: str = "full"):
        """
        参数:
            initial_pos: 初始3D位置 [x, y, z]
            measurement_std: 测量噪声
            model_selection: 模型选择
                - "full": 全部11个模型（最高精度，计算量大）
                - "standard": 标准6个模型（平衡）
                - "lite": 精简3个模型（快速）
        """
        self.model_selection = model_selection
        models = self._create_models(initial_pos, model_selection)
        
        # 构建转移矩阵
        n_models = len(models)
        transition_matrix = self._create_transition_matrix(n_models, model_selection)
        
        # 初始概率
        initial_probs = self._create_initial_probabilities(n_models, model_selection)
        
        # 测量噪声
        R = np.diag([measurement_std**2] * 3)
        
        # 创建IMM
        self.imm = IMMFilter(models, R, transition_matrix, initial_probs, dt=0.05)
        self.last_time = 0.0
        
        # 模型映射（用于解释）
        self.model_names = [m.name for m in models]
    
    def _create_models(self, initial_pos: np.ndarray, selection: str) -> List[ModelConfig]:
        """创建模型列表"""
        models = []
        
        if selection == "full":
            # ===== 垂直维度 =====
            # 1. 垂直俯冲
            models.append(ModelConfig(
                name="Dive",
                dim_x=9, dim_z=3,
                fx=f_vertical_dive,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   0, 0, -2.0, 0, 0, -5.0]),  # 初始向下
                P=np.diag([0.1**2]*3 + [1.0**2]*3 + [3.0**2]*3),
                Q=np.diag([0.1**2]*3 + [0.8**2]*3 + [2.0**2]*3)
            ))
            
            # 2. 垂直爬升
            models.append(ModelConfig(
                name="Climb",
                dim_x=9, dim_z=3,
                fx=f_vertical_climb,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   0, 0, 2.0, 0, 0, 3.0]),  # 初始向上
                P=np.diag([0.1**2]*3 + [1.0**2]*3 + [3.0**2]*3),
                Q=np.diag([0.1**2]*3 + [0.8**2]*3 + [2.0**2]*3)
            ))
            
            # 3. 垂直悬停
            models.append(ModelConfig(
                name="Hover-V",
                dim_x=6, dim_z=3,
                fx=f_hover_vertical,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   0, 0, 0]),
                P=np.diag([0.1**2]*3 + [0.2**2]*3),
                Q=np.diag([0.05**2]*3 + [0.1**2]*3)
            ))
            
            # ===== 水平维度 =====
            # 4. 水平匀速 (CV)
            from operator_imm import f_constant_velocity_3d
            models.append(ModelConfig(
                name="CV-H",
                dim_x=6, dim_z=3,
                fx=f_constant_velocity_3d,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   1.0, 0, 0]),
                P=np.diag([0.1**2]*3 + [1.0**2]*3),
                Q=np.diag([0.1**2]*3 + [0.5**2]*3)
            ))
            
            # 5. 水平加速 (CA)
            models.append(ModelConfig(
                name="CA-H",
                dim_x=9, dim_z=3,
                fx=f_horizontal_acceleration,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   1.0, 0, 0, 2.0, 0, 0]),
                P=np.diag([0.1**2]*3 + [1.0**2]*3 + [2.0**2]*3),
                Q=np.diag([0.1**2]*3 + [0.5**2]*3 + [1.0**2]*3)
            ))
            
            # 6. 3D协调转弯
            models.append(ModelConfig(
                name="CT-3D",
                dim_x=7, dim_z=3,
                fx=f_coordinated_turn_3d,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   2.0, 0, 0, 0.5]),  # v, phi, theta, omega
                P=np.diag([0.1**2]*3 + [1.0**2, np.deg2rad(10)**2, 
                          np.deg2rad(10)**2, np.deg2rad(5)**2]),
                Q=np.diag([0.1**2]*3 + [0.5**2, np.deg2rad(5)**2,
                          np.deg2rad(5)**2, np.deg2rad(10)**2])
            ))
            
            # 7. 侧飞
            models.append(ModelConfig(
                name="Sideways",
                dim_x=7, dim_z=3,
                fx=f_sideways_flight,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   0, 1.0, 0, 0]),  # 侧向速度
                P=np.diag([0.1**2]*3 + [1.0**2]*3 + [np.deg2rad(10)**2]),
                Q=np.diag([0.1**2]*3 + [0.5**2]*3 + [np.deg2rad(5)**2])
            ))
            
            # ===== 姿态旋转 =====
            # 8. 横滚机动
            models.append(ModelConfig(
                name="Roll",
                dim_x=8, dim_z=3,
                fx=f_roll_maneuver,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   1.0, 0, 0, 0, 2.0]),  # vx,vy,vz, roll, ωx
                P=np.diag([0.1**2]*3 + [1.0**2]*3 + [np.deg2rad(20)**2, np.deg2rad(30)**2]),
                Q=np.diag([0.1**2]*3 + [0.5**2]*3 + [np.deg2rad(10)**2, np.deg2rad(20)**2])
            ))
            
            # 9. 原地自旋
            models.append(ModelConfig(
                name="Spin",
                dim_x=5, dim_z=3,
                fx=f_spin_on_spot,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   0, 3.0]),  # yaw, ωz
                P=np.diag([0.1**2]*3 + [np.deg2rad(20)**2, np.deg2rad(50)**2]),
                Q=np.diag([0.05**2]*3 + [np.deg2rad(10)**2, np.deg2rad(30)**2])
            ))
            
            # 10. 半滚倒转
            models.append(ModelConfig(
                name="Half-Roll",
                dim_x=8, dim_z=3,
                fx=f_half_roll_inversion,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   1.5, 0, 0, 0, 0]),  # vx,vy,vz, roll, pitch
                P=np.diag([0.1**2]*3 + [1.5**2]*3 + [np.deg2rad(30)**2]*2),
                Q=np.diag([0.2**2]*3 + [0.8**2]*3 + [np.deg2rad(20)**2]*2)
            ))
            
            # ===== 速度突变 =====
            # 11. 急刹
            models.append(ModelConfig(
                name="Brake",
                dim_x=9, dim_z=3,
                fx=f_emergency_brake,
                hx=h_3d_full,
                initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2],
                                   3.0, 2.0, 0, -10.0, -8.0, 0]),  # 高负加速度
                P=np.diag([0.1**2]*3 + [2.0**2]*3 + [5.0**2]*3),
                Q=np.diag([0.2**2]*3 + [1.0**2]*3 + [3.0**2]*3)
            ))
            
        elif selection == "standard":
            # 标准6个模型：每个维度选代表性模型
            from operator_imm import f_constant_velocity_3d, f_constant_acceleration_3d, f_hovering_3d
            
            models.append(ModelConfig(name="CV", dim_x=6, dim_z=3, 
                                     fx=f_constant_velocity_3d, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0,0,0]),
                                     P=np.diag([0.1**2]*3 + [1.0**2]*3),
                                     Q=np.diag([0.1**2]*3 + [0.5**2]*3)))
            
            models.append(ModelConfig(name="CA", dim_x=9, dim_z=3,
                                     fx=f_constant_acceleration_3d, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0,0,0,0,0,0]),
                                     P=np.diag([0.1**2]*3 + [1.0**2]*3 + [2.0**2]*3),
                                     Q=np.diag([0.1**2]*3 + [0.5**2]*3 + [1.0**2]*3)))
            
            models.append(ModelConfig(name="CT", dim_x=7, dim_z=3,
                                     fx=f_coordinated_turn_3d, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 2,0,0,0.5]),
                                     P=np.diag([0.1**2]*3 + [1.0**2, np.deg2rad(10)**2, np.deg2rad(10)**2, np.deg2rad(5)**2]),
                                     Q=np.diag([0.1**2]*3 + [0.5**2, np.deg2rad(5)**2, np.deg2rad(5)**2, np.deg2rad(10)**2])))
            
            models.append(ModelConfig(name="Dive", dim_x=9, dim_z=3,
                                     fx=f_vertical_dive, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0,0,-2,0,0,-5]),
                                     P=np.diag([0.1**2]*3 + [1.0**2]*3 + [3.0**2]*3),
                                     Q=np.diag([0.1**2]*3 + [0.8**2]*3 + [2.0**2]*3)))
            
            models.append(ModelConfig(name="Hover", dim_x=6, dim_z=3,
                                     fx=f_hovering_3d, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0,0,0]),
                                     P=np.diag([0.1**2]*3 + [0.2**2]*3),
                                     Q=np.diag([0.05**2]*3 + [0.1**2]*3)))
            
            models.append(ModelConfig(name="Brake", dim_x=9, dim_z=3,
                                     fx=f_emergency_brake, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 3,2,0,-10,-8,0]),
                                     P=np.diag([0.1**2]*3 + [2.0**2]*3 + [5.0**2]*3),
                                     Q=np.diag([0.2**2]*3 + [1.0**2]*3 + [3.0**2]*3)))
        
        else:  # lite
            from operator_imm import f_constant_velocity_3d, f_constant_acceleration_3d, f_hovering_3d
            
            models.append(ModelConfig(name="CV", dim_x=6, dim_z=3,
                                     fx=f_constant_velocity_3d, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0,0,0]),
                                     P=np.diag([0.1**2]*3 + [1.0**2]*3),
                                     Q=np.diag([0.1**2]*3 + [0.5**2]*3)))
            
            models.append(ModelConfig(name="CA", dim_x=9, dim_z=3,
                                     fx=f_constant_acceleration_3d, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0,0,0,0,0,0]),
                                     P=np.diag([0.1**2]*3 + [1.0**2]*3 + [2.0**2]*3),
                                     Q=np.diag([0.1**2]*3 + [0.5**2]*3 + [1.0**2]*3)))
            
            models.append(ModelConfig(name="Hover", dim_x=6, dim_z=3,
                                     fx=f_hovering_3d, hx=h_3d_full,
                                     initial_x=np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0,0,0]),
                                     P=np.diag([0.1**2]*3 + [0.2**2]*3),
                                     Q=np.diag([0.05**2]*3 + [0.1**2]*3)))
        
        return models
    
    def _create_transition_matrix(self, n: int, selection: str) -> np.ndarray:
        """创建模型转移矩阵"""
        if selection == "full":
            # 11×11 转移矩阵
            # 按维度分组，组内转移概率高，组间转移概率低
            matrix = np.zeros((n, n))
            
            # 对角线基础概率
            np.fill_diagonal(matrix, 0.88)
            
            # 垂直维度内部转移 (0,1,2)
            matrix[0, [1,2]] = [0.05, 0.05]  # Dive -> Climb/Hover-V
            matrix[1, [0,2]] = [0.05, 0.05]  # Climb -> Dive/Hover-V
            matrix[2, [0,1]] = [0.03, 0.03]  # Hover-V -> Dive/Climb
            
            # 水平维度内部转移 (3,4,5,6)
            matrix[3, [4,5,6]] = [0.05, 0.04, 0.02]  # CV-H转移
            matrix[4, [3,5,6]] = [0.05, 0.04, 0.02]  # CA-H转移
            matrix[5, [3,4,6]] = [0.04, 0.04, 0.03]  # CT转移
            matrix[6, [3,4,5]] = [0.03, 0.04, 0.04]  # Sideways转移
            
            # 姿态旋转内部转移 (7,8,9)
            matrix[7, [8,9]] = [0.04, 0.04]  # Roll -> Spin/Half-Roll
            matrix[8, [7,9]] = [0.04, 0.04]  # Spin -> Roll/Half-Roll
            matrix[9, [7,8]] = [0.05, 0.03]  # Half-Roll -> Roll/Spin
            
            # 速度突变 (10)
            matrix[10, [3,4]] = [0.06, 0.05]  # Brake -> CV/CA
            
            # 跨维度转移（较小概率）
            # 从任何状态都可能转到Brake
            for i in range(n-1):
                if matrix[i, 10] == 0:
                    matrix[i, 10] = 0.01
            
            # 归一化每一行
            for i in range(n):
                row_sum = np.sum(matrix[i, :])
                if row_sum > 0:
                    matrix[i, :] /= row_sum
            
        elif selection == "standard":
            # 6×6 标准矩阵
            matrix = np.array([
                [0.90, 0.04, 0.02, 0.01, 0.02, 0.01],  # CV
                [0.04, 0.88, 0.03, 0.01, 0.03, 0.01],  # CA
                [0.03, 0.03, 0.90, 0.01, 0.02, 0.01],  # CT
                [0.02, 0.02, 0.02, 0.91, 0.02, 0.01],  # Dive
                [0.02, 0.02, 0.02, 0.02, 0.91, 0.01],  # Hover
                [0.05, 0.05, 0.02, 0.01, 0.02, 0.85]   # Brake
            ])
        else:  # lite
            # 3×3 精简矩阵
            matrix = np.array([
                [0.92, 0.05, 0.03],
                [0.05, 0.90, 0.05],
                [0.03, 0.05, 0.92]
            ])
        
        return matrix
    
    def _create_initial_probabilities(self, n: int, selection: str) -> np.ndarray:
        """创建初始概率"""
        if selection == "full":
            # 水平匀速概率最高
            probs = np.ones(n) * 0.05
            probs[3] = 0.25  # CV-H
            probs[4] = 0.20  # CA-H
            probs[0] = 0.10  # Dive
            probs[10] = 0.05  # Brake
        elif selection == "standard":
            probs = np.array([0.3, 0.25, 0.15, 0.12, 0.13, 0.05])
        else:
            probs = np.array([0.5, 0.3, 0.2])
        
        return probs / np.sum(probs)  # 归一化
    
    def update(self, current_pos: np.ndarray, current_time: float):
        """更新IMM"""
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
            
            # 射击建议：基于活跃模型和运动复杂度
            active_idx = np.argmax(model_probs)
            active_model = self.model_names[active_idx]
            
            # 不同模型的射击适宜性
            fire_weights = {
                'CV-H': 0.9, 'Hover-V': 0.95, 'Hover': 0.95,  # 稳定模式
                'CA-H': 0.7, 'Climb': 0.6, 'CT-3D': 0.5,      # 中等
                'Dive': 0.4, 'Sideways': 0.5,                 # 较难
                'Roll': 0.3, 'Spin': 0.2, 'Half-Roll': 0.2, 'Brake': 0.4  # 很难
            }
            
            fire_feasibility = sum(model_probs[i] * fire_weights.get(self.model_names[i], 0.5) 
                                 for i in range(len(model_probs)))
            fire_feasibility = np.clip(fire_feasibility, 0.0, 1.0)
            
            # 构建模型概率字典（按维度分组）
            model_probs_dict = {}
            for i, name in enumerate(self.model_names):
                model_probs_dict[name] = float(model_probs[i])
            
            predictions[delay_ms] = {
                'predicted_position': pred_pos.tolist(),
                'accuracy_score': accuracy_score,
                'fire_feasibility': fire_feasibility,
                'active_model': active_model,
                'model_probabilities': model_probs_dict,
                'motion_dimension': self._classify_motion_dimension(active_model)
            }
        
        return predictions
    
    def _classify_motion_dimension(self, model_name: str) -> str:
        """分类运动维度"""
        dimension_map = {
            'Dive': '垂直方向',
            'Climb': '垂直方向', 
            'Hover-V': '垂直方向',
            'CV-H': '水平方向',
            'CA-H': '水平方向',
            'CT-3D': '水平方向',
            'Sideways': '水平方向',
            'Roll': '姿态旋转',
            'Spin': '姿态旋转',
            'Half-Roll': '姿态旋转',
            'Brake': '速度突变',
            'CV': '基础',
            'CA': '基础',
            'Hover': '基础'
        }
        return dimension_map.get(model_name, '未知')
    
    def get_current_state(self) -> Dict:
        """获取当前状态（增强信息）"""
        state_info = self.imm.get_state()
        x_fused = state_info['fused_state']
        
        vel = x_fused[3:6] if len(x_fused) >= 6 else np.array([0.0, 0.0, 0.0])
        
        active_idx = np.argmax(state_info['model_probabilities'])
        active_model = self.model_names[active_idx]
        
        return {
            'position': x_fused[:3].tolist(),
            'velocity': vel.tolist(),
            'speed': float(np.linalg.norm(vel)),
            'active_model': active_model,
            'motion_dimension': self._classify_motion_dimension(active_model),
            'model_probabilities': {
                name: float(prob) 
                for name, prob in zip(self.model_names, state_info['model_probabilities'])
            },
            'uncertainty': float(np.linalg.det(state_info['fused_covariance'][:3, :3])),
            'model_count': len(self.model_names),
            'model_selection': self.model_selection
        }
    
    def get_dimension_probabilities(self) -> Dict[str, float]:
        """获取各运动维度的概率分布"""
        state_info = self.imm.get_state()
        probs = state_info['model_probabilities']
        
        dimensions = {
            '垂直方向': 0.0,
            '水平方向': 0.0,
            '姿态旋转': 0.0,
            '速度突变': 0.0,
            '基础': 0.0
        }
        
        for i, name in enumerate(self.model_names):
            dim = self._classify_motion_dimension(name)
            dimensions[dim] += probs[i]
        
        return dimensions

