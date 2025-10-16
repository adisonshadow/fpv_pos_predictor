"""
多目标跟踪 (Multi-Target Tracking, MTT)

实现同时跟踪多个FPV目标的能力，包括：
1. 数据关联 (Data Association) - 将观测与目标匹配
2. 目标管理 (Target Management) - 创建、维护、删除轨迹
3. 最近邻关联 (Nearest Neighbor)
4. 全局最近邻 (Global Nearest Neighbor, GNN)
5. 联合概率数据关联 (JPDA)

Author: pointfang@gmail.com
Date: 2025-10-16
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import time


@dataclass
class Target:
    """单个跟踪目标"""
    target_id: str
    predictor: any  # 预测器实例
    created_time: float
    last_update_time: float
    update_count: int = 0
    miss_count: int = 0
    confirmed: bool = False
    alive: bool = True
    
    # 目标属性
    track_quality: float = 1.0
    uncertainty: float = 0.0
    
    def update_from_state(self, state: Dict):
        """从状态字典更新目标属性"""
        self.track_quality = state.get('track_quality', 1.0)
        self.uncertainty = state.get('uncertainty', 0.0)


class DataAssociator:
    """数据关联器 - 解决观测与目标的匹配问题"""
    
    def __init__(self, association_threshold: float = 3.0,
                 method: str = "gnn"):
        """
        参数:
            association_threshold: 关联阈值（马氏距离）
            method: 关联方法 "nearest" (最近邻) 或 "gnn" (全局最近邻)
        """
        self.association_threshold = association_threshold
        self.method = method
    
    def compute_distance_matrix(self, targets: List[Target],
                                measurements: List[np.ndarray]) -> np.ndarray:
        """
        计算目标与测量之间的距离矩阵
        
        返回:
            距离矩阵 [n_targets x n_measurements]
        """
        n_targets = len(targets)
        n_meas = len(measurements)
        
        distance_matrix = np.full((n_targets, n_meas), np.inf)
        
        for i, target in enumerate(targets):
            # 预测目标位置
            state = target.predictor.get_current_state()
            predicted_pos = np.array(state['position'])
            
            for j, meas in enumerate(measurements):
                # 计算欧氏距离（简化版，实际应用马氏距离）
                distance = np.linalg.norm(predicted_pos - meas)
                
                # 检查阈值
                if distance < self.association_threshold:
                    distance_matrix[i, j] = distance
        
        return distance_matrix
    
    def associate_nearest_neighbor(self, targets: List[Target],
                                   measurements: List[np.ndarray]) -> Dict[int, int]:
        """
        最近邻关联
        
        返回:
            字典 {target_idx: measurement_idx}
        """
        distance_matrix = self.compute_distance_matrix(targets, measurements)
        associations = {}
        
        used_measurements = set()
        
        for i in range(len(targets)):
            # 找到最近的测量
            valid_dists = []
            for j in range(len(measurements)):
                if j not in used_measurements and distance_matrix[i, j] < np.inf:
                    valid_dists.append((distance_matrix[i, j], j))
            
            if valid_dists:
                # 选择最近的
                valid_dists.sort()
                _, best_j = valid_dists[0]
                associations[i] = best_j
                used_measurements.add(best_j)
        
        return associations
    
    def associate_global_nearest_neighbor(self, targets: List[Target],
                                         measurements: List[np.ndarray]) -> Dict[int, int]:
        """
        全局最近邻关联 (使用匈牙利算法)
        
        返回:
            字典 {target_idx: measurement_idx}
        """
        distance_matrix = self.compute_distance_matrix(targets, measurements)
        
        # 使用匈牙利算法求最优分配
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        associations = {}
        for i, j in zip(row_ind, col_ind):
            if distance_matrix[i, j] < np.inf:
                associations[i] = j
        
        return associations
    
    def associate(self, targets: List[Target],
                 measurements: List[np.ndarray]) -> Tuple[Dict[int, int], List[int]]:
        """
        执行数据关联
        
        返回:
            (associations, unassociated_measurements)
            associations: {target_idx: measurement_idx}
            unassociated_measurements: 未关联的测量索引列表
        """
        if not targets or not measurements:
            return {}, list(range(len(measurements)))
        
        # 选择关联方法
        if self.method == "gnn":
            associations = self.associate_global_nearest_neighbor(targets, measurements)
        else:
            associations = self.associate_nearest_neighbor(targets, measurements)
        
        # 找出未关联的测量
        associated_meas = set(associations.values())
        unassociated = [j for j in range(len(measurements)) if j not in associated_meas]
        
        return associations, unassociated


class MultiTargetTracker:
    """多目标跟踪器"""
    
    def __init__(self, predictor_type: str = "2D",
                 measurement_std: float = 0.1,
                 process_std: float = 0.5,
                 max_targets: int = 10,
                 association_threshold: float = 3.0,
                 confirmation_threshold: int = 3,
                 deletion_threshold: int = 5,
                 use_imm: bool = False,
                 use_adaptive: bool = False,
                 use_recovery: bool = False):
        """
        初始化多目标跟踪器
        
        参数:
            predictor_type: "2D" 或 "3D"
            measurement_std: 测量噪声
            process_std: 过程噪声
            max_targets: 最大跟踪目标数
            association_threshold: 数据关联阈值
            confirmation_threshold: 确认目标所需的连续更新次数
            deletion_threshold: 删除目标的连续丢失次数
            use_imm: 是否使用IMM
            use_adaptive: 是否使用自适应
            use_recovery: 是否使用跟踪恢复
        """
        self.predictor_type = predictor_type
        self.measurement_std = measurement_std
        self.process_std = process_std
        self.max_targets = max_targets
        self.confirmation_threshold = confirmation_threshold
        self.deletion_threshold = deletion_threshold
        
        # 特性标志
        self.use_imm = use_imm
        self.use_adaptive = use_adaptive
        self.use_recovery = use_recovery
        
        # 目标列表
        self.targets: List[Target] = []
        self.next_target_id = 1
        
        # 数据关联器
        self.associator = DataAssociator(association_threshold, method="gnn")
        
        # 统计
        self.total_created = 0
        self.total_deleted = 0
        self.total_confirmed = 0
    
    def _create_predictor(self, initial_pos: np.ndarray):
        """创建单个目标的预测器"""
        # 选择预测器类型
        if self.use_imm:
            if self.predictor_type == "2D":
                from operator_imm import IMMPredictor2D
                predictor = IMMPredictor2D(initial_pos, self.measurement_std)
            else:
                from operator_imm import IMMPredictor3D
                predictor = IMMPredictor3D(initial_pos, self.measurement_std)
        else:
            if self.predictor_type == "2D":
                from operator import FlyPredictor
                predictor = FlyPredictor(initial_pos, self.measurement_std, self.process_std)
            else:
                from operator import FlyPredictor3D
                predictor = FlyPredictor3D(initial_pos, self.measurement_std, self.process_std)
        
        # 应用高级功能
        if self.use_adaptive:
            from operator_adaptive import AdaptiveFilter
            predictor = AdaptiveFilter(predictor, adaptation_rate=0.1, window_size=10)
        
        if self.use_recovery:
            from operator_track_recovery import TrackRecoveryFilter
            predictor = TrackRecoveryFilter(predictor, max_miss_count=self.deletion_threshold)
        
        return predictor
    
    def _create_new_target(self, measurement: np.ndarray, current_time: float) -> Target:
        """创建新目标"""
        target_id = f"T{self.next_target_id:03d}"
        self.next_target_id += 1
        
        predictor = self._create_predictor(measurement)
        
        target = Target(
            target_id=target_id,
            predictor=predictor,
            created_time=current_time,
            last_update_time=current_time,
            update_count=1,
            miss_count=0,
            confirmed=False
        )
        
        self.total_created += 1
        return target
    
    def update(self, measurements: List[np.ndarray], current_time: float):
        """
        更新多目标跟踪
        
        参数:
            measurements: 测量列表（可能来自多个目标）
            current_time: 当前时间
        """
        # 1. 数据关联
        associations, unassociated = self.associator.associate(
            self.targets, measurements
        )
        
        # 2. 更新已关联的目标
        for target_idx, meas_idx in associations.items():
            target = self.targets[target_idx]
            measurement = measurements[meas_idx]
            
            # 更新预测器
            if self.use_recovery:
                target.predictor.update(measurement, current_time, measurement_confidence=1.0)
            else:
                target.predictor.update(measurement, current_time)
            
            # 更新目标信息
            target.last_update_time = current_time
            target.update_count += 1
            target.miss_count = 0
            
            # 确认目标
            if not target.confirmed and target.update_count >= self.confirmation_threshold:
                target.confirmed = True
                self.total_confirmed += 1
            
            # 更新目标属性
            state = target.predictor.get_current_state()
            target.update_from_state(state)
        
        # 3. 处理未关联的目标（丢失）
        for i, target in enumerate(self.targets):
            if i not in associations:
                target.miss_count += 1
                
                # 如果支持恢复，尝试预测
                if self.use_recovery:
                    target.predictor.update(None, current_time, measurement_confidence=0.0)
        
        # 4. 创建新目标（从未关联的测量）
        if len(self.targets) < self.max_targets:
            for meas_idx in unassociated:
                new_target = self._create_new_target(measurements[meas_idx], current_time)
                self.targets.append(new_target)
        
        # 5. 删除长时间丢失的目标
        self._prune_dead_targets()
    
    def _prune_dead_targets(self):
        """删除死亡目标"""
        alive_targets = []
        
        for target in self.targets:
            # 删除条件：连续丢失次数过多 且 未确认
            if target.miss_count >= self.deletion_threshold and not target.confirmed:
                target.alive = False
                self.total_deleted += 1
            # 或者：已确认但长时间丢失
            elif target.miss_count >= self.deletion_threshold * 2 and target.confirmed:
                target.alive = False
                self.total_deleted += 1
            else:
                alive_targets.append(target)
        
        self.targets = alive_targets
    
    def predict_all(self, delay_ms_list: List[int]) -> Dict[str, Dict]:
        """
        预测所有目标
        
        返回:
            {target_id: predictions}
        """
        all_predictions = {}
        
        for target in self.targets:
            if target.confirmed:  # 只预测已确认的目标
                predictions = target.predictor.predict_and_evaluate(delay_ms_list)
                all_predictions[target.target_id] = predictions
        
        return all_predictions
    
    def get_all_states(self) -> List[Dict]:
        """获取所有目标的状态"""
        states = []
        
        for target in self.targets:
            state = target.predictor.get_current_state()
            state['target_id'] = target.target_id
            state['confirmed'] = target.confirmed
            state['update_count'] = target.update_count
            state['miss_count'] = target.miss_count
            state['alive'] = target.alive
            state['age'] = target.last_update_time - target.created_time
            states.append(state)
        
        return states
    
    def get_confirmed_targets(self) -> List[Target]:
        """获取已确认的目标"""
        return [t for t in self.targets if t.confirmed]
    
    def get_target_by_id(self, target_id: str) -> Optional[Target]:
        """根据ID获取目标"""
        for target in self.targets:
            if target.target_id == target_id:
                return target
        return None
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_targets': len(self.targets),
            'confirmed_targets': len(self.get_confirmed_targets()),
            'tentative_targets': len([t for t in self.targets if not t.confirmed]),
            'total_created': self.total_created,
            'total_deleted': self.total_deleted,
            'total_confirmed': self.total_confirmed
        }


class ClusteringAssociator:
    """基于聚类的数据关联（适合密集目标场景）"""
    
    def __init__(self, cluster_radius: float = 1.0):
        """
        参数:
            cluster_radius: 聚类半径
        """
        self.cluster_radius = cluster_radius
    
    def cluster_measurements(self, measurements: List[np.ndarray]) -> List[List[int]]:
        """
        将测量聚类
        
        返回:
            聚类列表，每个聚类包含测量索引
        """
        if not measurements:
            return []
        
        clusters = []
        assigned = set()
        
        for i, meas in enumerate(measurements):
            if i in assigned:
                continue
            
            # 创建新聚类
            cluster = [i]
            assigned.add(i)
            
            # 找到所有在半径内的测量
            for j, other_meas in enumerate(measurements):
                if j != i and j not in assigned:
                    dist = np.linalg.norm(meas - other_meas)
                    if dist < self.cluster_radius:
                        cluster.append(j)
                        assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters


class JPDAAssociator:
    """联合概率数据关联 (JPDA)"""
    
    def __init__(self, gate_threshold: float = 3.0,
                 detection_probability: float = 0.9):
        """
        参数:
            gate_threshold: 门限阈值
            detection_probability: 检测概率
        """
        self.gate_threshold = gate_threshold
        self.pd = detection_probability
    
    def compute_association_probabilities(self, targets: List[Target],
                                         measurements: List[np.ndarray]) -> np.ndarray:
        """
        计算关联概率矩阵
        
        返回:
            概率矩阵 [n_targets x (n_measurements + 1)]
            最后一列是"无关联"的概率
        """
        n_targets = len(targets)
        n_meas = len(measurements)
        
        # 初始化概率矩阵
        prob_matrix = np.zeros((n_targets, n_meas + 1))
        
        for i, target in enumerate(targets):
            state = target.predictor.get_current_state()
            pred_pos = np.array(state['position'])
            
            # 计算每个测量的似然
            likelihoods = []
            for meas in measurements:
                dist = np.linalg.norm(pred_pos - meas)
                
                if dist < self.gate_threshold:
                    # 高斯似然（简化）
                    likelihood = np.exp(-0.5 * (dist / self.gate_threshold)**2)
                    likelihoods.append(likelihood)
                else:
                    likelihoods.append(0.0)
            
            # 归一化
            total = sum(likelihoods) + (1 - self.pd)  # 加上无检测概率
            
            if total > 0:
                for j in range(n_meas):
                    prob_matrix[i, j] = likelihoods[j] / total
                prob_matrix[i, n_meas] = (1 - self.pd) / total  # 无关联
        
        return prob_matrix


class MultiTargetTrackerAdvanced(MultiTargetTracker):
    """高级多目标跟踪器 - 支持JPDA"""
    
    def __init__(self, use_jpda: bool = False, **kwargs):
        """
        参数:
            use_jpda: 是否使用JPDA
            **kwargs: 传递给基类的参数
        """
        super().__init__(**kwargs)
        self.use_jpda = use_jpda
        
        if use_jpda:
            self.jpda = JPDAAssociator(
                gate_threshold=kwargs.get('association_threshold', 3.0)
            )
    
    def update_with_jpda(self, measurements: List[np.ndarray], current_time: float):
        """使用JPDA更新"""
        if not self.targets:
            # 没有目标，所有测量创建新目标
            for meas in measurements:
                if len(self.targets) < self.max_targets:
                    new_target = self._create_new_target(meas, current_time)
                    self.targets.append(new_target)
            return
        
        # 计算关联概率
        prob_matrix = self.jpda.compute_association_probabilities(self.targets, measurements)
        
        # 使用概率加权更新（JPDA核心思想）
        for i, target in enumerate(self.targets):
            # 计算加权平均测量
            weighted_meas = np.zeros(len(measurements[0]))
            total_prob = 0.0
            
            for j, meas in enumerate(measurements):
                prob = prob_matrix[i, j]
                if prob > 0.1:  # 只考虑概率较高的
                    weighted_meas += prob * meas
                    total_prob += prob
            
            if total_prob > 0.1:
                weighted_meas /= total_prob
                
                # 更新
                if self.use_recovery:
                    target.predictor.update(weighted_meas, current_time, 
                                          measurement_confidence=total_prob)
                else:
                    target.predictor.update(weighted_meas, current_time)
                
                target.last_update_time = current_time
                target.update_count += 1
                target.miss_count = 0
            else:
                # 未关联
                target.miss_count += 1


def create_multi_target_tracker(predictor_type: str = "2D",
                                use_imm: bool = False,
                                use_adaptive: bool = False,
                                use_recovery: bool = False,
                                use_jpda: bool = False,
                                **kwargs) -> MultiTargetTracker:
    """
    创建多目标跟踪器的便捷函数
    
    参数:
        predictor_type: "2D" 或 "3D"
        use_imm: 使用IMM多模型
        use_adaptive: 使用自适应滤波
        use_recovery: 使用跟踪恢复
        use_jpda: 使用JPDA关联
        **kwargs: 其他参数
    
    返回:
        MultiTargetTracker 或 MultiTargetTrackerAdvanced
    """
    config = {
        'predictor_type': predictor_type,
        'use_imm': use_imm,
        'use_adaptive': use_adaptive,
        'use_recovery': use_recovery,
        **kwargs
    }
    
    if use_jpda:
        return MultiTargetTrackerAdvanced(use_jpda=True, **config)
    else:
        return MultiTargetTracker(**config)

