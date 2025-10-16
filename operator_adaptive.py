"""
自适应卡尔曼滤波 (Adaptive Kalman Filter, AKF)

根据观测残差和新息序列自动调整过程噪声Q和测量噪声R矩阵，
提高滤波器在模型不确定或环境变化时的性能。

实现方法:
1. 新息自适应法 (Innovation-based Adaptation)
2. 残差自适应法 (Residual-based Adaptation)
3. 多窗口自适应法 (Multiple Window Adaptation)

Author: pointfang@gmail.com
Date: 2025-10-16
"""

import numpy as np
from typing import Optional, Tuple, Deque
from collections import deque


class InnovationMonitor:
    """新息序列监控器 - 用于检测滤波器性能"""
    
    def __init__(self, window_size: int = 10):
        """
        参数:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        self.innovations = deque(maxlen=window_size)
        self.innovation_covariances = deque(maxlen=window_size)
    
    def update(self, innovation: np.ndarray, S: np.ndarray):
        """
        更新新息序列
        
        参数:
            innovation: 新息 (观测值 - 预测值)
            S: 新息协方差
        """
        self.innovations.append(innovation.copy())
        self.innovation_covariances.append(S.copy())
    
    def get_normalized_innovation_squared(self) -> float:
        """
        计算归一化新息平方 (NIS - Normalized Innovation Squared)
        
        返回:
            NIS值，用于判断滤波器是否一致
        """
        if len(self.innovations) == 0:
            return 0.0
        
        # 使用最近的新息
        innovation = self.innovations[-1]
        S = self.innovation_covariances[-1]
        
        try:
            S_inv = np.linalg.inv(S)
            nis = innovation.T @ S_inv @ innovation
            return float(nis)
        except:
            return 0.0
    
    def get_average_innovation(self) -> np.ndarray:
        """计算平均新息"""
        if len(self.innovations) == 0:
            return np.zeros(2)  # 默认2D
        
        return np.mean(list(self.innovations), axis=0)
    
    def is_consistent(self, threshold: float = 3.0) -> bool:
        """
        检查滤波器是否一致
        
        参数:
            threshold: NIS阈值（通常为卡方分布的95%分位数）
        
        返回:
            True表示滤波器工作正常
        """
        nis = self.get_normalized_innovation_squared()
        # 对于2D观测，自由度=2，95%置信度的卡方值约为5.99
        # 对于3D观测，自由度=3，95%置信度的卡方值约为7.81
        return nis < threshold * 2.0


class AdaptiveNoiseEstimator:
    """自适应噪声估计器"""
    
    def __init__(self, initial_Q: np.ndarray, initial_R: np.ndarray,
                 adaptation_rate: float = 0.1, window_size: int = 10):
        """
        参数:
            initial_Q: 初始过程噪声协方差
            initial_R: 初始测量噪声协方差
            adaptation_rate: 自适应速率 (0-1)
            window_size: 滑动窗口大小
        """
        self.Q = initial_Q.copy()
        self.R = initial_R.copy()
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size
        
        # 历史记录
        self.residual_history = deque(maxlen=window_size)
        self.innovation_history = deque(maxlen=window_size)
    
    def update_R_from_innovations(self, innovations: deque, 
                                   predicted_S: np.ndarray) -> np.ndarray:
        """
        基于新息序列更新测量噪声R
        
        方法: R_new = (1-α)*R_old + α*sample_covariance(innovations)
        
        参数:
            innovations: 新息序列
            predicted_S: 预测的新息协方差
        
        返回:
            更新后的R矩阵
        """
        if len(innovations) < 3:
            return self.R
        
        # 计算新息的样本协方差
        innovations_array = np.array(list(innovations))
        sample_cov = np.cov(innovations_array.T)
        
        # 确保是矩阵形式
        if sample_cov.ndim == 0:
            sample_cov = np.array([[sample_cov]])
        elif sample_cov.ndim == 1:
            sample_cov = np.diag(sample_cov)
        
        # 自适应更新
        alpha = self.adaptation_rate
        R_new = (1 - alpha) * self.R + alpha * sample_cov
        
        # 确保对称性和正定性
        R_new = (R_new + R_new.T) / 2
        R_new = R_new + np.eye(R_new.shape[0]) * 1e-6
        
        self.R = R_new
        return self.R
    
    def update_Q_from_residuals(self, residuals: deque, 
                                H: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        基于残差序列更新过程噪声Q
        
        参数:
            residuals: 残差序列 (观测值 - 滤波后估计值)
            H: 观测矩阵
            P: 状态协方差
        
        返回:
            更新后的Q矩阵
        """
        if len(residuals) < 3:
            return self.Q
        
        # 计算残差协方差
        residuals_array = np.array(list(residuals))
        residual_cov = np.cov(residuals_array.T)
        
        if residual_cov.ndim == 0:
            residual_cov = np.array([[residual_cov]])
        elif residual_cov.ndim == 1:
            residual_cov = np.diag(residual_cov)
        
        # 通过残差估计Q
        # C_residual ≈ H*P*H^T + R
        # 已知R，可以反推P，进而调整Q
        try:
            alpha = self.adaptation_rate
            # 简化方法：直接调整Q的量级
            scale_factor = np.trace(residual_cov) / (np.trace(self.R) + 1e-6)
            
            if scale_factor > 2.0:  # 残差过大，增加Q
                Q_new = self.Q * (1 + alpha * 0.5)
            elif scale_factor < 0.5:  # 残差过小，减小Q
                Q_new = self.Q * (1 - alpha * 0.3)
            else:
                Q_new = self.Q
            
            # 确保正定性
            Q_new = Q_new + np.eye(Q_new.shape[0]) * 1e-6
            self.Q = Q_new
            
        except:
            pass
        
        return self.Q
    
    def adaptive_update(self, innovation: np.ndarray, residual: np.ndarray,
                       S: np.ndarray, H: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        自适应更新Q和R
        
        参数:
            innovation: 当前新息
            residual: 当前残差
            S: 新息协方差
            H: 观测矩阵
            P: 状态协方差
        
        返回:
            (Q_new, R_new)
        """
        # 添加到历史
        self.innovation_history.append(innovation)
        self.residual_history.append(residual)
        
        # 更新R
        R_new = self.update_R_from_innovations(self.innovation_history, S)
        
        # 更新Q
        Q_new = self.update_Q_from_residuals(self.residual_history, H, P)
        
        return Q_new, R_new


class AdaptiveFilter:
    """自适应滤波器包装器 - 为现有滤波器添加自适应能力"""
    
    def __init__(self, base_filter, adaptation_rate: float = 0.1,
                 window_size: int = 10, enable_adaptation: bool = True):
        """
        参数:
            base_filter: 基础滤波器 (FlyPredictor或FlyPredictor3D)
            adaptation_rate: 自适应速率
            window_size: 滑动窗口大小
            enable_adaptation: 是否启用自适应
        """
        self.filter = base_filter
        self.enable_adaptation = enable_adaptation
        
        # 创建监控器和估计器
        self.innovation_monitor = InnovationMonitor(window_size)
        self.noise_estimator = AdaptiveNoiseEstimator(
            base_filter.ukf.Q.copy(),
            base_filter.ukf.R.copy(),
            adaptation_rate,
            window_size
        )
        
        # 统计信息
        self.adaptation_count = 0
        self.total_updates = 0
    
    def update(self, current_pos: np.ndarray, current_time: float):
        """
        更新滤波器（带自适应）
        
        参数:
            current_pos: 当前位置测量
            current_time: 当前时间
        """
        self.total_updates += 1
        
        # 预测步骤
        dt = current_time - self.filter.last_time
        if dt <= 0.001:
            return
        
        self.filter.ukf.dt = dt
        self.filter.ukf.predict()
        
        # 计算新息
        z = current_pos[:len(self.filter.ukf.R)]  # 观测值
        z_pred = self.filter.ukf.x[:len(z)]  # 预测观测值
        innovation = z - z_pred
        
        # 计算新息协方差
        H = np.eye(len(z), len(self.filter.ukf.x))  # 简化的观测矩阵
        S = H @ self.filter.ukf.P @ H.T + self.filter.ukf.R
        
        # 更新监控器
        self.innovation_monitor.update(innovation, S)
        
        # 执行更新步骤
        self.filter.ukf.update(z)
        
        # 计算残差
        residual = z - self.filter.ukf.x[:len(z)]
        
        # 自适应调整（如果启用）
        if self.enable_adaptation and self.total_updates > 5:
            # 检查是否需要自适应
            if not self.innovation_monitor.is_consistent():
                Q_new, R_new = self.noise_estimator.adaptive_update(
                    innovation, residual, S, H, self.filter.ukf.P
                )
                
                # 应用新的噪声矩阵
                self.filter.ukf.Q = Q_new
                self.filter.ukf.R = R_new
                self.adaptation_count += 1
        
        # 更新时间
        self.filter.last_time = current_time
        self.filter.last_pos = current_pos
    
    def predict_and_evaluate(self, delay_ms_list):
        """预测（调用基础滤波器的方法）"""
        return self.filter.predict_and_evaluate(delay_ms_list)
    
    def get_current_state(self):
        """获取当前状态"""
        state = self.filter.get_current_state()
        state['adaptation_enabled'] = self.enable_adaptation
        state['adaptation_count'] = self.adaptation_count
        state['total_updates'] = self.total_updates
        state['adaptation_rate'] = self.adaptation_count / max(1, self.total_updates)
        state['is_consistent'] = self.innovation_monitor.is_consistent()
        return state
    
    def reset(self, initial_pos: np.ndarray, initial_time: float = 0.0):
        """重置滤波器"""
        self.filter.reset(initial_pos, initial_time)
        
        # 重置自适应组件
        self.innovation_monitor = InnovationMonitor(self.innovation_monitor.window_size)
        self.noise_estimator = AdaptiveNoiseEstimator(
            self.filter.ukf.Q.copy(),
            self.filter.ukf.R.copy(),
            self.noise_estimator.adaptation_rate,
            self.noise_estimator.window_size
        )
        self.adaptation_count = 0
        self.total_updates = 0
    
    def get_noise_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取当前的Q和R矩阵"""
        return self.filter.ukf.Q.copy(), self.filter.ukf.R.copy()
    
    def get_adaptation_stats(self) -> dict:
        """获取自适应统计信息"""
        return {
            'total_updates': self.total_updates,
            'adaptation_count': self.adaptation_count,
            'adaptation_rate': self.adaptation_count / max(1, self.total_updates),
            'is_consistent': self.innovation_monitor.is_consistent(),
            'current_nis': self.innovation_monitor.get_normalized_innovation_squared(),
            'Q_trace': np.trace(self.filter.ukf.Q),
            'R_trace': np.trace(self.filter.ukf.R)
        }


def create_adaptive_predictor(predictor_type: str = "2D", **kwargs):
    """
    创建自适应预测器的便捷函数
    
    参数:
        predictor_type: "2D" 或 "3D"
        **kwargs: 传递给基础预测器的参数
    
    返回:
        AdaptiveFilter实例
    """
    if predictor_type == "2D":
        from operator import FlyPredictor
        base = FlyPredictor(**kwargs)
    elif predictor_type == "3D":
        from operator import FlyPredictor3D
        base = FlyPredictor3D(**kwargs)
    else:
        raise ValueError(f"不支持的预测器类型: {predictor_type}")
    
    return AdaptiveFilter(base, adaptation_rate=0.1, window_size=10)


# ==================== 高级自适应方法 ====================

class MultiWindowAdaptiveFilter(AdaptiveFilter):
    """多窗口自适应滤波器 - 使用多个时间尺度"""
    
    def __init__(self, base_filter, short_window: int = 5, 
                 long_window: int = 20, **kwargs):
        """
        参数:
            base_filter: 基础滤波器
            short_window: 短期窗口大小
            long_window: 长期窗口大小
        """
        super().__init__(base_filter, window_size=short_window, **kwargs)
        
        # 长期监控器
        self.long_monitor = InnovationMonitor(long_window)
        
        # 短期和长期估计器
        self.short_estimator = self.noise_estimator
        self.long_estimator = AdaptiveNoiseEstimator(
            base_filter.ukf.Q.copy(),
            base_filter.ukf.R.copy(),
            adaptation_rate=0.05,  # 长期调整更缓慢
            window_size=long_window
        )
    
    def update(self, current_pos: np.ndarray, current_time: float):
        """使用多窗口策略更新"""
        # 执行基础更新
        super().update(current_pos, current_time)
        
        # 额外的长期监控
        if self.total_updates > 1:
            # 更新长期监控器（使用已计算的新息）
            # 这里简化处理，实际可以从父类获取
            pass


class RobustAdaptiveFilter(AdaptiveFilter):
    """鲁棒自适应滤波器 - 对异常值更robust"""
    
    def __init__(self, base_filter, outlier_threshold: float = 5.0, **kwargs):
        """
        参数:
            base_filter: 基础滤波器
            outlier_threshold: 异常值阈值（马氏距离）
        """
        super().__init__(base_filter, **kwargs)
        self.outlier_threshold = outlier_threshold
        self.outlier_count = 0
    
    def is_outlier(self, innovation: np.ndarray, S: np.ndarray) -> bool:
        """
        检测是否为异常值
        
        参数:
            innovation: 新息
            S: 新息协方差
        
        返回:
            True表示异常值
        """
        try:
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = np.sqrt(innovation.T @ S_inv @ innovation)
            return mahalanobis_dist > self.outlier_threshold
        except:
            return False
    
    def update(self, current_pos: np.ndarray, current_time: float):
        """带异常值检测的更新"""
        # 先预测
        dt = current_time - self.filter.last_time
        if dt <= 0.001:
            return
        
        self.filter.ukf.dt = dt
        self.filter.ukf.predict()
        
        # 计算新息
        z = current_pos[:len(self.filter.ukf.R)]
        z_pred = self.filter.ukf.x[:len(z)]
        innovation = z - z_pred
        
        H = np.eye(len(z), len(self.filter.ukf.x))
        S = H @ self.filter.ukf.P @ H.T + self.filter.ukf.R
        
        # 检测异常值
        if self.is_outlier(innovation, S):
            self.outlier_count += 1
            # 异常值处理：降低测量权重或跳过更新
            # 这里选择增大R来降低测量权重
            self.filter.ukf.R = self.filter.ukf.R * 2.0
        
        # 正常更新流程
        super().update(current_pos, current_time)
        
        # 恢复R
        if self.outlier_count > 0 and not self.is_outlier(innovation, S):
            # 逐渐恢复
            self.filter.ukf.R = self.filter.ukf.R * 0.95
    
    def get_adaptation_stats(self) -> dict:
        """获取统计信息（包含异常值计数）"""
        stats = super().get_adaptation_stats()
        stats['outlier_count'] = self.outlier_count
        stats['outlier_rate'] = self.outlier_count / max(1, self.total_updates)
        return stats

