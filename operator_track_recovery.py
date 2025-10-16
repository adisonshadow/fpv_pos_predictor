"""
跟踪丢失恢复 (Track Loss Recovery)

实现目标跟踪丢失检测和自动重新初始化机制，
提高系统在遮挡、丢失信号等情况下的鲁棒性。

功能:
1. 跟踪质量监控
2. 丢失检测
3. 自动重新初始化
4. 预测辅助搜索

Author: pointfang@gmail.com
Date: 2025-10-16
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import deque
from enum import Enum


class TrackStatus(Enum):
    """跟踪状态枚举"""
    INITIALIZING = "initializing"  # 初始化中
    TRACKING = "tracking"          # 正常跟踪
    UNCERTAIN = "uncertain"        # 不确定状态
    LOST = "lost"                  # 跟踪丢失
    RECOVERED = "recovered"        # 已恢复


class TrackQualityMonitor:
    """跟踪质量监控器"""
    
    def __init__(self, position_threshold: float = 2.0,
                 uncertainty_threshold: float = 5.0,
                 history_size: int = 10):
        """
        参数:
            position_threshold: 位置跳变阈值（米）
            uncertainty_threshold: 不确定性阈值
            history_size: 历史记录大小
        """
        self.position_threshold = position_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.history_size = history_size
        
        # 历史记录
        self.position_history = deque(maxlen=history_size)
        self.uncertainty_history = deque(maxlen=history_size)
        self.measurement_gaps = deque(maxlen=history_size)
        
        # 质量分数历史
        self.quality_scores = deque(maxlen=history_size)
        
        self.last_measurement_time = 0.0
    
    def update(self, position: np.ndarray, uncertainty: float, 
               current_time: float) -> float:
        """
        更新质量监控
        
        参数:
            position: 当前位置
            uncertainty: 当前不确定性
            current_time: 当前时间
        
        返回:
            质量分数 (0-1, 越高越好)
        """
        # 计算时间间隔
        if self.last_measurement_time > 0:
            gap = current_time - self.last_measurement_time
            self.measurement_gaps.append(gap)
        
        # 记录历史
        self.position_history.append(position.copy())
        self.uncertainty_history.append(uncertainty)
        self.last_measurement_time = current_time
        
        # 计算质量分数
        quality = self._calculate_quality_score(position, uncertainty)
        self.quality_scores.append(quality)
        
        return quality
    
    def _calculate_quality_score(self, position: np.ndarray, 
                                 uncertainty: float) -> float:
        """计算质量分数"""
        score = 1.0
        
        # 1. 不确定性惩罚
        if uncertainty > self.uncertainty_threshold:
            score *= 0.5
        elif uncertainty > self.uncertainty_threshold * 0.5:
            score *= 0.8
        
        # 2. 位置跳变检测
        if len(self.position_history) >= 2:
            last_pos = self.position_history[-1]
            distance = np.linalg.norm(position - last_pos)
            
            if distance > self.position_threshold * 2:
                score *= 0.3  # 严重跳变
            elif distance > self.position_threshold:
                score *= 0.6  # 轻微跳变
        
        # 3. 测量间隔惩罚
        if len(self.measurement_gaps) > 0:
            avg_gap = np.mean(list(self.measurement_gaps))
            if self.measurement_gaps[-1] > avg_gap * 3:
                score *= 0.7  # 测量间隔过大
        
        return np.clip(score, 0.0, 1.0)
    
    def get_average_quality(self) -> float:
        """获取平均质量分数"""
        if len(self.quality_scores) == 0:
            return 1.0
        return float(np.mean(list(self.quality_scores)))
    
    def is_stable(self) -> bool:
        """判断跟踪是否稳定"""
        avg_quality = self.get_average_quality()
        return avg_quality > 0.7
    
    def is_degraded(self) -> bool:
        """判断跟踪是否退化"""
        avg_quality = self.get_average_quality()
        return avg_quality < 0.5


class TrackLossDetector:
    """跟踪丢失检测器"""
    
    def __init__(self, max_miss_count: int = 5,
                 max_time_gap: float = 0.5,
                 innovation_threshold: float = 3.0):
        """
        参数:
            max_miss_count: 最大连续丢失次数
            max_time_gap: 最大时间间隔（秒）
            innovation_threshold: 新息阈值（标准差倍数）
        """
        self.max_miss_count = max_miss_count
        self.max_time_gap = max_time_gap
        self.innovation_threshold = innovation_threshold
        
        # 状态
        self.consecutive_miss_count = 0
        self.last_update_time = 0.0
        self.total_miss_count = 0
    
    def check_measurement_received(self, current_time: float, 
                                  received: bool) -> bool:
        """
        检查是否收到测量
        
        参数:
            current_time: 当前时间
            received: 是否收到测量
        
        返回:
            True表示跟踪丢失
        """
        if received:
            self.consecutive_miss_count = 0
            self.last_update_time = current_time
            return False
        else:
            self.consecutive_miss_count += 1
            self.total_miss_count += 1
            
            # 检查丢失条件
            time_gap = current_time - self.last_update_time if self.last_update_time > 0 else 0
            
            if self.consecutive_miss_count >= self.max_miss_count:
                return True
            
            if time_gap > self.max_time_gap:
                return True
            
            return False
    
    def check_innovation(self, innovation: np.ndarray, 
                        S: np.ndarray) -> bool:
        """
        检查新息是否异常（可能表示跟踪丢失）
        
        参数:
            innovation: 新息向量
            S: 新息协方差
        
        返回:
            True表示新息异常，可能丢失
        """
        try:
            # 计算马氏距离
            S_inv = np.linalg.inv(S)
            mahalanobis = np.sqrt(innovation.T @ S_inv @ innovation)
            
            # 与阈值比较
            return mahalanobis > self.innovation_threshold
        except:
            return False
    
    def reset(self):
        """重置检测器"""
        self.consecutive_miss_count = 0
        self.last_update_time = 0.0


class TrackRecoveryManager:
    """跟踪恢复管理器"""
    
    def __init__(self, search_radius: float = 5.0,
                 recovery_confidence_threshold: float = 0.8):
        """
        参数:
            search_radius: 搜索半径（米）
            recovery_confidence_threshold: 恢复置信度阈值
        """
        self.search_radius = search_radius
        self.recovery_confidence_threshold = recovery_confidence_threshold
        
        # 恢复状态
        self.is_in_recovery = False
        self.recovery_attempts = 0
        self.max_recovery_attempts = 10
        
        # 预测辅助
        self.last_known_position = None
        self.last_known_velocity = None
        self.lost_time = None
    
    def enter_recovery_mode(self, last_position: np.ndarray,
                           last_velocity: Optional[np.ndarray],
                           lost_time: float):
        """
        进入恢复模式
        
        参数:
            last_position: 最后已知位置
            last_velocity: 最后已知速度
            lost_time: 丢失时间
        """
        self.is_in_recovery = True
        self.recovery_attempts = 0
        self.last_known_position = last_position.copy()
        self.last_known_velocity = last_velocity.copy() if last_velocity is not None else None
        self.lost_time = lost_time
    
    def get_search_region(self, current_time: float) -> Tuple[np.ndarray, float]:
        """
        获取搜索区域
        
        参数:
            current_time: 当前时间
        
        返回:
            (搜索中心, 搜索半径)
        """
        if self.last_known_position is None:
            return None, self.search_radius
        
        # 基于最后位置和速度预测搜索中心
        search_center = self.last_known_position.copy()
        
        if self.last_known_velocity is not None and self.lost_time is not None:
            dt = current_time - self.lost_time
            # 外推位置
            search_center = search_center + self.last_known_velocity * dt
        
        # 搜索半径随时间增大
        if self.lost_time is not None:
            dt = current_time - self.lost_time
            radius = self.search_radius * (1 + dt * 0.5)  # 每秒扩大50%
        else:
            radius = self.search_radius
        
        return search_center, radius
    
    def attempt_recovery(self, candidate_position: np.ndarray,
                        confidence: float, current_time: float) -> bool:
        """
        尝试恢复跟踪
        
        参数:
            candidate_position: 候选位置
            confidence: 置信度
            current_time: 当前时间
        
        返回:
            True表示恢复成功
        """
        self.recovery_attempts += 1
        
        # 检查是否在搜索区域内
        search_center, radius = self.get_search_region(current_time)
        if search_center is not None:
            distance = np.linalg.norm(candidate_position - search_center)
            if distance > radius:
                return False  # 超出搜索范围
        
        # 检查置信度
        if confidence < self.recovery_confidence_threshold:
            return False
        
        # 恢复成功
        self.is_in_recovery = False
        return True
    
    def is_recovery_failed(self) -> bool:
        """判断恢复是否失败"""
        return self.recovery_attempts >= self.max_recovery_attempts
    
    def reset(self):
        """重置恢复管理器"""
        self.is_in_recovery = False
        self.recovery_attempts = 0
        self.last_known_position = None
        self.last_known_velocity = None
        self.lost_time = None


class TrackRecoveryFilter:
    """带跟踪丢失恢复的滤波器"""
    
    def __init__(self, base_filter,
                 position_threshold: float = 2.0,
                 max_miss_count: int = 5,
                 search_radius: float = 5.0):
        """
        参数:
            base_filter: 基础滤波器
            position_threshold: 位置阈值
            max_miss_count: 最大丢失次数
            search_radius: 搜索半径
        """
        self.filter = base_filter
        
        # 创建组件
        self.quality_monitor = TrackQualityMonitor(
            position_threshold=position_threshold
        )
        self.loss_detector = TrackLossDetector(
            max_miss_count=max_miss_count
        )
        self.recovery_manager = TrackRecoveryManager(
            search_radius=search_radius
        )
        
        # 状态
        self.status = TrackStatus.TRACKING
        self.update_count = 0
        self.loss_count = 0
        self.recovery_count = 0
        
        # 上一次成功更新的状态
        self.last_good_position = None
        self.last_good_velocity = None
        self.last_good_time = 0.0
    
    def update(self, measurement: Optional[np.ndarray], 
               current_time: float, measurement_confidence: float = 1.0):
        """
        更新滤波器（带丢失处理）
        
        参数:
            measurement: 测量值（None表示丢失）
            current_time: 当前时间
            measurement_confidence: 测量置信度
        """
        self.update_count += 1
        
        # 检查是否收到测量
        received = measurement is not None
        
        # 跟踪丢失检测
        is_lost = self.loss_detector.check_measurement_received(current_time, received)
        
        if is_lost and self.status == TrackStatus.TRACKING:
            # 进入丢失状态
            self._handle_track_loss(current_time)
            return
        
        if not received:
            # 无测量，只进行预测
            self._predict_only(current_time)
            return
        
        # 收到测量
        if self.status == TrackStatus.LOST:
            # 尝试恢复
            self._attempt_recovery(measurement, measurement_confidence, current_time)
        else:
            # 正常更新
            self._normal_update(measurement, current_time, measurement_confidence)
    
    def _normal_update(self, measurement: np.ndarray, 
                      current_time: float, confidence: float):
        """正常更新"""
        # 调用基础滤波器更新
        self.filter.update(measurement, current_time)
        
        # 更新质量监控
        state = self.filter.get_current_state()
        uncertainty = state.get('uncertainty', 0.0)
        quality = self.quality_monitor.update(
            np.array(state['position']),
            uncertainty,
            current_time
        )
        
        # 更新状态
        if quality > 0.8:
            self.status = TrackStatus.TRACKING
        elif quality > 0.5:
            self.status = TrackStatus.UNCERTAIN
        
        # 记录良好状态
        if quality > 0.7:
            self.last_good_position = np.array(state['position'])
            self.last_good_velocity = np.array(state.get('velocity', [0, 0]))
            self.last_good_time = current_time
    
    def _predict_only(self, current_time: float):
        """仅预测（无测量更新）"""
        dt = current_time - self.filter.last_time
        if dt > 0.001:
            self.filter.ukf.dt = dt
            self.filter.ukf.predict()
            self.filter.last_time = current_time
    
    def _handle_track_loss(self, current_time: float):
        """处理跟踪丢失"""
        print(f"[警告] 跟踪丢失 @ {current_time:.2f}s")
        
        self.status = TrackStatus.LOST
        self.loss_count += 1
        
        # 进入恢复模式
        self.recovery_manager.enter_recovery_mode(
            self.last_good_position if self.last_good_position is not None 
            else np.array(self.filter.get_current_state()['position']),
            self.last_good_velocity,
            current_time
        )
    
    def _attempt_recovery(self, measurement: np.ndarray,
                         confidence: float, current_time: float):
        """尝试恢复跟踪"""
        # 尝试恢复
        success = self.recovery_manager.attempt_recovery(
            measurement, confidence, current_time
        )
        
        if success:
            print(f"[成功] 跟踪已恢复 @ {current_time:.2f}s")
            self.status = TrackStatus.RECOVERED
            self.recovery_count += 1
            
            # 重新初始化滤波器
            self.filter.reset(measurement, current_time)
            self.loss_detector.reset()
            self.recovery_manager.reset()
            
            # 切换回正常跟踪
            self.status = TrackStatus.TRACKING
        elif self.recovery_manager.is_recovery_failed():
            print(f"[失败] 恢复失败，需要手动重新初始化")
            self.status = TrackStatus.LOST
    
    def predict_and_evaluate(self, delay_ms_list: List[int]) -> Dict:
        """预测"""
        predictions = self.filter.predict_and_evaluate(delay_ms_list)
        
        # 添加跟踪状态信息
        for delay_ms in predictions:
            predictions[delay_ms]['track_status'] = self.status.value
            predictions[delay_ms]['track_quality'] = self.quality_monitor.get_average_quality()
        
        return predictions
    
    def get_current_state(self) -> Dict:
        """获取当前状态"""
        state = self.filter.get_current_state()
        
        # 添加跟踪信息
        state['track_status'] = self.status.value
        state['track_quality'] = self.quality_monitor.get_average_quality()
        state['loss_count'] = self.loss_count
        state['recovery_count'] = self.recovery_count
        state['is_stable'] = self.quality_monitor.is_stable()
        
        # 如果在恢复中，添加搜索区域
        if self.recovery_manager.is_in_recovery:
            search_center, search_radius = self.recovery_manager.get_search_region(
                self.filter.last_time
            )
            state['search_center'] = search_center.tolist() if search_center is not None else None
            state['search_radius'] = search_radius
        
        return state
    
    def reset(self, initial_pos: np.ndarray, initial_time: float = 0.0):
        """重置"""
        self.filter.reset(initial_pos, initial_time)
        self.status = TrackStatus.TRACKING
        self.quality_monitor = TrackQualityMonitor()
        self.loss_detector.reset()
        self.recovery_manager.reset()
        self.last_good_position = initial_pos.copy()
        self.last_good_time = initial_time


def create_robust_predictor(predictor_type: str = "2D", 
                           enable_recovery: bool = True, **kwargs):
    """
    创建带跟踪恢复的鲁棒预测器
    
    参数:
        predictor_type: "2D" 或 "3D"
        enable_recovery: 是否启用恢复功能
        **kwargs: 传递给基础预测器的参数
    
    返回:
        TrackRecoveryFilter实例
    """
    if predictor_type == "2D":
        from operator import FlyPredictor
        base = FlyPredictor(**kwargs)
    elif predictor_type == "3D":
        from operator import FlyPredictor3D
        base = FlyPredictor3D(**kwargs)
    else:
        raise ValueError(f"不支持的预测器类型: {predictor_type}")
    
    if enable_recovery:
        return TrackRecoveryFilter(
            base,
            position_threshold=2.0,
            max_miss_count=5,
            search_radius=5.0
        )
    else:
        return base

