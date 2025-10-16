"""
FPV预测算子包装器 - 标准化算子接口实现

基于OperatorIO协议，提供标准化的输入/输出接口，支持热插拔和跨语言调用
使用JSON格式进行数据交换（兼容Protobuf结构）
"""

import json
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class OperatorIOWrapper:
    """算子IO包装器基类"""
    
    # 错误码定义
    ERROR_CODES = {
        1000: "数据解析失败",
        1001: "数据类型不兼容",
        1002: "维度不匹配",
        1003: "时间戳异常",
        1004: "参数缺失",
        1005: "算子状态异常",
        2000: "预测器未初始化",
        2001: "位置数据无效"
    }
    
    @staticmethod
    def create_metadata(io_id: str, data_type: str, shape: List[int],
                       source: str = "", timestamp: Optional[int] = None,
                       ext: Optional[Dict[str, str]] = None) -> Dict:
        """创建元数据"""
        return {
            "io_id": io_id,
            "data_type": data_type,
            "shape": shape,
            "source": source,
            "timestamp": timestamp or int(time.time() * 1000),
            "ext": ext or {}
        }
    
    @staticmethod
    def create_error(code: int, detail: str = "") -> Dict:
        """创建错误信息"""
        return {
            "code": code,
            "msg": OperatorIOWrapper.ERROR_CODES.get(code, "未知错误"),
            "detail": detail
        }
    
    @staticmethod
    def create_operator_io(metadata: Dict, data_bodies: List[Dict],
                          control_info: Optional[Dict] = None,
                          error: Optional[Dict] = None) -> Dict:
        """创建OperatorIO结构"""
        return {
            "metadata": metadata,
            "data_bodies": data_bodies,
            "control_info": control_info or {},
            "error": error
        }


class FPVPredictorOperator(OperatorIOWrapper):
    """FPV预测算子 - 标准化接口封装（支持高级功能）"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化预测算子
        
        参数:
            config: 配置字典，包含：
                - type: "2D", "3D", "IMM_2D", "IMM_3D"
                - initial_position: 初始位置列表
                - measurement_std: 测量噪声标准差
                - process_std: 过程噪声标准差
                - prediction_delays: 预测延迟列表（毫秒）
                - features: "none", "adaptive", "recovery", "both" (高级功能)
                - adaptation_rate: 自适应速率 (可选)
                - adaptation_window: 自适应窗口 (可选)
                - recovery_search_radius: 搜索半径 (可选)
                - max_miss_count: 最大丢失次数 (可选)
        """
        self.config = config
        self.predictor = None
        self.predictor_type = config.get("type", "2D")
        self.prediction_delays = config.get("prediction_delays", [50, 200, 500, 1000])
        self.features = config.get("features", "none")
        self.operator_id = f"fpv_predictor_{self.predictor_type}_{self.features}_{int(time.time())}"
        
        # 尝试初始化预测器
        self._initialize_predictor(config)
    
    def _initialize_predictor(self, config: Dict):
        """内部方法：初始化预测器（支持高级功能）"""
        try:
            from operator import FlyPredictor, FlyPredictor3D
            
            initial_pos = config.get("initial_position")
            measurement_std = config.get("measurement_std", 0.1)
            process_std = config.get("process_std", 0.5)
            
            # 创建基础预测器
            base_predictor = None
            
            if self.predictor_type in ["2D", "IMM_2D"]:
                if not initial_pos or len(initial_pos) < 2:
                    raise ValueError("2D预测器需要至少2个初始坐标")
                
                if self.predictor_type == "IMM_2D":
                    # 使用IMM
                    from operator_imm import IMMPredictor2D
                    base_predictor = IMMPredictor2D(
                        initial_pos=initial_pos[:2],
                        measurement_std=measurement_std
                    )
                else:
                    # 普通2D
                    base_predictor = FlyPredictor(
                        initial_pos=initial_pos[:2],
                        measurement_std=measurement_std,
                        process_std=process_std
                    )
                    
            elif self.predictor_type in ["3D", "IMM_3D"]:
                if not initial_pos or len(initial_pos) < 3:
                    raise ValueError("3D预测器需要至少3个初始坐标")
                
                if self.predictor_type == "IMM_3D":
                    # 使用IMM 3D
                    from operator_imm import IMMPredictor3D
                    base_predictor = IMMPredictor3D(
                        initial_pos=initial_pos[:3],
                        measurement_std=measurement_std
                    )
                else:
                    # 普通3D
                    base_predictor = FlyPredictor3D(
                        initial_pos=initial_pos[:3],
                        measurement_std=measurement_std,
                        process_std=process_std
                    )
            else:
                raise ValueError(f"不支持的预测器类型: {self.predictor_type}")
            
            # 应用高级功能
            self.predictor = self._apply_advanced_features(base_predictor, config)
                
        except ImportError as e:
            print(f"警告: 无法导入预测器模块 - {e}")
            self.predictor = None
        except Exception as e:
            print(f"初始化预测器失败: {e}")
            self.predictor = None
    
    def _apply_advanced_features(self, base_predictor, config: Dict):
        """应用高级功能（自适应、恢复等）"""
        predictor = base_predictor
        
        # 应用自适应滤波
        if self.features in ["adaptive", "both"]:
            try:
                from operator_adaptive import AdaptiveFilter
                adaptation_rate = config.get("adaptation_rate", 0.1)
                adaptation_window = config.get("adaptation_window", 10)
                
                predictor = AdaptiveFilter(
                    predictor,
                    adaptation_rate=adaptation_rate,
                    window_size=adaptation_window,
                    enable_adaptation=True
                )
                print(f"✓ 已启用自适应滤波 (rate={adaptation_rate}, window={adaptation_window})")
            except ImportError:
                print("警告: 无法导入operator_adaptive，跳过自适应功能")
        
        # 应用跟踪恢复
        if self.features in ["recovery", "both"]:
            try:
                from operator_track_recovery import TrackRecoveryFilter
                search_radius = config.get("recovery_search_radius", 5.0)
                max_miss = config.get("max_miss_count", 5)
                
                predictor = TrackRecoveryFilter(
                    predictor,
                    position_threshold=2.0,
                    max_miss_count=max_miss,
                    search_radius=search_radius
                )
                print(f"✓ 已启用跟踪恢复 (radius={search_radius}m, max_miss={max_miss})")
            except ImportError:
                print("警告: 无法导入operator_track_recovery，跳过恢复功能")
        
        return predictor
    
    def process_input(self, operator_io: Dict) -> Dict:
        """
        处理输入的OperatorIO，执行预测算子逻辑
        
        参数:
            operator_io: 标准OperatorIO格式的输入
        
        返回:
            标准OperatorIO格式的输出
        """
        # 解析控制信息
        control_info = operator_io.get("control_info", {})
        action = control_info.get("op_action", "compute")
        
        # 根据动作调用相应方法
        if action == "reset":
            return self._handle_reset(operator_io)
        elif action == "update":
            return self._handle_update(operator_io)
        elif action == "predict":
            return self._handle_predict(operator_io)
        elif action == "get_state":
            return self._handle_get_state(operator_io)
        elif action == "compute":
            # compute = update + predict
            update_result = self._handle_update(operator_io)
            if update_result.get("error"):
                return update_result
            return self._handle_predict(operator_io)
        else:
            return self._create_error_response(
                1001, 
                f"不支持的动作: {action}",
                operator_io.get("metadata", {})
            )
    
    def _handle_update(self, operator_io: Dict) -> Dict:
        """处理更新动作（更新测量数据，支持无测量情况）"""
        if not self.predictor:
            return self._create_error_response(2000, "预测器未初始化", 
                                              operator_io.get("metadata", {}))
        
        try:
            # 检查是否有测量数据
            control_info = operator_io.get("control_info", {})
            params = control_info.get("params", {})
            has_measurement = params.get("has_measurement", "true") == "true"
            
            # 解析位置数据
            data_bodies = operator_io.get("data_bodies", [])
            
            # 无测量的情况（用于跟踪恢复）
            if not has_measurement or not data_bodies:
                # 检查是否支持跟踪恢复
                if hasattr(self.predictor, 'update') and self.features in ["recovery", "both"]:
                    # 使用None调用update（TrackRecoveryFilter支持）
                    timestamp = operator_io.get("metadata", {}).get("timestamp", time.time() * 1000) / 1000
                    self.predictor.update(None, timestamp, measurement_confidence=0.0)
                    
                    metadata = self.create_metadata(
                        io_id=f"{self.operator_id}_update_{int(time.time()*1000)}",
                        data_type="update_ack",
                        shape=[],
                        source=self.operator_id
                    )
                    
                    return self.create_operator_io(
                        metadata=metadata,
                        data_bodies=[{"json_struct": json.dumps({"status": "no_measurement"})}],
                        control_info={"op_action": "update", "status": "no_measurement"}
                    )
                else:
                    return self._create_error_response(1004, "缺少数据体且不支持恢复模式",
                                                      operator_io.get("metadata", {}))
            
            # 提取位置和时间
            first_body = data_bodies[0]
            
            # 支持多种输入格式
            if "position" in first_body:
                # 标准PositionData格式
                pos_data = first_body["position"]
                position = pos_data.get("coordinates", [])
                timestamp = pos_data.get("timestamp", time.time())
            elif "float_array" in first_body:
                # 简单数组格式（匹配新的 FloatArray 结构）
                float_array = first_body["float_array"]
                position = float_array.get("values", float_array) if isinstance(float_array, dict) else float_array
                timestamp = operator_io.get("metadata", {}).get("timestamp", time.time() * 1000) / 1000
            else:
                return self._create_error_response(1001, "无法解析位置数据",
                                                  operator_io.get("metadata", {}))
            
            # 验证维度
            expected_dim = 2 if self.predictor_type in ["2D", "IMM_2D"] else 3
            if len(position) < expected_dim:
                return self._create_error_response(
                    1002,
                    f"维度不匹配: 期望{expected_dim}维，实际{len(position)}维",
                    operator_io.get("metadata", {})
                )
            
            # 获取测量置信度（用于跟踪恢复）
            confidence = 1.0
            if "position" in first_body:
                confidence = first_body["position"].get("confidence", 1.0)
            
            # 更新预测器
            # 检查是否支持confidence参数（TrackRecoveryFilter支持）
            if self.features in ["recovery", "both"]:
                self.predictor.update(position, timestamp, measurement_confidence=confidence)
            else:
                self.predictor.update(position, timestamp)
            
            # 返回成功响应
            metadata = self.create_metadata(
                io_id=f"{self.operator_id}_update_{int(time.time()*1000)}",
                data_type="update_ack",
                shape=[],
                source=self.operator_id
            )
            
            return self.create_operator_io(
                metadata=metadata,
                data_bodies=[{"json_struct": json.dumps({"status": "updated"})}],
                control_info={"op_action": "update", "status": "success"}
            )
            
        except Exception as e:
            return self._create_error_response(1005, str(e), 
                                              operator_io.get("metadata", {}))
    
    def _handle_predict(self, operator_io: Dict) -> Dict:
        """处理预测动作"""
        if not self.predictor:
            return self._create_error_response(2000, "预测器未初始化",
                                              operator_io.get("metadata", {}))
        
        try:
            # 获取预测延迟参数
            control_info = operator_io.get("control_info", {})
            params = control_info.get("params", {})
            delays = params.get("prediction_delays")
            
            if delays:
                # 解析延迟列表
                if isinstance(delays, str):
                    delays = json.loads(delays)
                prediction_delays = delays
            else:
                prediction_delays = self.prediction_delays
            
            # 执行预测
            predictions = self.predictor.predict_and_evaluate(prediction_delays)
            
            # 构造预测结果
            prediction_result = {
                "predictions": [],
                "prediction_time": int(time.time() * 1000),
                "predictor_type": self.predictor_type
            }
            
            for delay_ms, pred in predictions.items():
                prediction_result["predictions"].append({
                    "delay_ms": delay_ms,
                    "predicted_position": pred["predicted_position"],
                    "accuracy_score": pred["accuracy_score"],
                    "fire_feasibility": pred["fire_feasibility"]
                })
            
            # 创建响应
            metadata = self.create_metadata(
                io_id=f"{self.operator_id}_predict_{int(time.time()*1000)}",
                data_type="prediction_result",
                shape=[len(prediction_result["predictions"])],
                source=self.operator_id
            )
            
            data_body = {
                "prediction": prediction_result
            }
            
            return self.create_operator_io(
                metadata=metadata,
                data_bodies=[data_body],
                control_info={"op_action": "predict", "status": "success"}
            )
            
        except Exception as e:
            return self._create_error_response(1005, str(e),
                                              operator_io.get("metadata", {}))
    
    def _handle_get_state(self, operator_io: Dict) -> Dict:
        """处理获取状态动作"""
        if not self.predictor:
            return self._create_error_response(2000, "预测器未初始化",
                                              operator_io.get("metadata", {}))
        
        try:
            # 获取当前状态
            state = self.predictor.get_current_state()
            
            # 构造状态估计结果（包含所有高级功能信息）
            state_estimate = {
                "position": state["position"],
                "velocity": state.get("velocity", [0, 0]),
                "speed": state.get("speed", 0.0),
                "uncertainty": state["uncertainty"],
                "timestamp": int(time.time() * 1000),
                "predictor_type": self.predictor_type,
                "enabled_features": self.features
            }
            
            # 3D特有信息
            if self.predictor_type in ["3D", "IMM_3D"]:
                state_estimate["acceleration"] = state.get("acceleration", [0, 0, 0])
                state_estimate["acceleration_magnitude"] = state.get("acceleration_magnitude", 0.0)
            
            # IMM特有信息
            if "active_model" in state:
                state_estimate["active_model"] = state["active_model"]
                state_estimate["model_probabilities"] = state.get("model_probabilities", [])
            
            # 自适应滤波信息
            if "adaptation_count" in state:
                state_estimate["adaptive_stats"] = {
                    "adaptation_count": state.get("adaptation_count", 0),
                    "adaptation_rate": state.get("adaptation_rate", 0.0),
                    "is_consistent": state.get("is_consistent", True)
                }
            
            # 跟踪恢复信息
            if "track_status" in state:
                state_estimate["track_info"] = {
                    "status": state.get("track_status", "tracking"),
                    "quality": state.get("track_quality", 1.0),
                    "loss_count": state.get("loss_count", 0),
                    "recovery_count": state.get("recovery_count", 0),
                    "is_stable": state.get("is_stable", True)
                }
                
                # 搜索区域信息
                if "search_center" in state and state["search_center"] is not None:
                    state_estimate["track_info"]["search_center"] = state["search_center"]
                    state_estimate["track_info"]["search_radius"] = state.get("search_radius", 0.0)
            
            # 创建响应
            metadata = self.create_metadata(
                io_id=f"{self.operator_id}_state_{int(time.time()*1000)}",
                data_type="state_estimate",
                shape=[len(state_estimate["position"])],
                source=self.operator_id
            )
            
            return self.create_operator_io(
                metadata=metadata,
                data_bodies=[{"state": state_estimate}],
                control_info={"op_action": "get_state", "status": "success"}
            )
            
        except Exception as e:
            return self._create_error_response(1005, str(e),
                                              operator_io.get("metadata", {}))
    
    def _handle_reset(self, operator_io: Dict) -> Dict:
        """处理重置动作"""
        if not self.predictor:
            return self._create_error_response(2000, "预测器未初始化",
                                              operator_io.get("metadata", {}))
        
        try:
            # 解析新的初始位置
            data_bodies = operator_io.get("data_bodies", [])
            if data_bodies and "float_array" in data_bodies[0]:
                float_array = data_bodies[0]["float_array"]
                new_position = float_array.get("values", float_array) if isinstance(float_array, dict) else float_array
            else:
                new_position = self.config["initial_position"]
            
            # 重置预测器
            self.predictor.reset(new_position, 0.0)
            
            # 创建响应
            metadata = self.create_metadata(
                io_id=f"{self.operator_id}_reset_{int(time.time()*1000)}",
                data_type="reset_ack",
                shape=[],
                source=self.operator_id
            )
            
            return self.create_operator_io(
                metadata=metadata,
                data_bodies=[{"json_struct": json.dumps({"status": "reset"})}],
                control_info={"op_action": "reset", "status": "success"}
            )
            
        except Exception as e:
            return self._create_error_response(1005, str(e),
                                              operator_io.get("metadata", {}))
    
    def _create_error_response(self, code: int, detail: str, 
                              input_metadata: Dict) -> Dict:
        """创建错误响应"""
        metadata = self.create_metadata(
            io_id=f"{self.operator_id}_error_{int(time.time()*1000)}",
            data_type="error",
            shape=[],
            source=self.operator_id
        )
        
        error = self.create_error(code, detail)
        
        return self.create_operator_io(
            metadata=metadata,
            data_bodies=[],
            error=error
        )
    
    def to_json(self, data: Dict) -> str:
        """转换为JSON字符串"""
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    def from_json(self, json_str: str) -> Dict:
        """从JSON字符串解析"""
        return json.loads(json_str)


# 便捷函数：创建标准输入
def create_position_input(position: List[float], timestamp: float = None,
                         source: str = "sensor", dimension: str = "2D") -> Dict:
    """
    创建位置输入的OperatorIO
    
    参数:
        position: 位置坐标列表
        timestamp: 时间戳（秒）
        source: 数据来源
        dimension: 维度 "2D" 或 "3D"
    """
    metadata = OperatorIOWrapper.create_metadata(
        io_id=f"input_{int(time.time()*1000)}",
        data_type=f"position_{dimension.lower()}",
        shape=[len(position)],
        source=source,
        timestamp=int((timestamp or time.time()) * 1000)
    )
    
    data_body = {
        "position": {
            "dimension": dimension,
            "coordinates": position,
            "timestamp": timestamp or time.time(),
            "confidence": 1.0,
            "sensor_id": source
        }
    }
    
    return OperatorIOWrapper.create_operator_io(
        metadata=metadata,
        data_bodies=[data_body],
        control_info={"op_action": "compute", "priority": 1}
    )


def create_update_only_input(position: List[float], timestamp: float = None) -> Dict:
    """创建仅更新（不预测）的输入"""
    io_data = create_position_input(position, timestamp)
    io_data["control_info"]["op_action"] = "update"
    return io_data


def create_predict_input(prediction_delays: List[int] = None) -> Dict:
    """创建预测请求的输入"""
    metadata = OperatorIOWrapper.create_metadata(
        io_id=f"predict_request_{int(time.time()*1000)}",
        data_type="predict_request",
        shape=[],
        source="user"
    )
    
    params = {}
    if prediction_delays:
        params["prediction_delays"] = json.dumps(prediction_delays)
    
    return OperatorIOWrapper.create_operator_io(
        metadata=metadata,
        data_bodies=[],
        control_info={"op_action": "predict", "params": params}
    )

