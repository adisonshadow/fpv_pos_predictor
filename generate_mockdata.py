"""
FPV 运动轨迹模拟数据生成器

生成多种典型的FPV运动模式的模拟数据，用于测试和验证预测算法
"""

import numpy as np
import json
from pathlib import Path


class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, dt=0.05, noise_std=0.1):
        """
        初始化生成器
        
        参数:
            dt: 时间步长 (秒)
            noise_std: 传感器噪声标准差 (米)
        """
        self.dt = dt
        self.noise_std = noise_std
    
    def add_noise(self, position, noise_std=None):
        """添加传感器噪声"""
        if noise_std is None:
            noise_std = self.noise_std
        return position + np.random.randn(len(position)) * noise_std
    
    # ==================== 2D 运动模式 ====================
    
    def generate_2d_straight_line(self, duration=2.0, speed=2.5, angle=45):
        """
        生成2D直线运动数据
        
        参数:
            duration: 持续时间 (秒)
            speed: 速度 (m/s)
            angle: 运动方向角度 (度)
        """
        angle_rad = np.deg2rad(angle)
        vx = speed * np.cos(angle_rad)
        vy = speed * np.sin(angle_rad)
        
        data = []
        time = 0.0
        pos = np.array([0.0, 0.0])
        
        while time <= duration:
            # 位置更新
            pos = pos + np.array([vx, vy]) * self.dt
            
            # 添加噪声并记录
            noisy_pos = self.add_noise(pos)
            data.append({
                'time': round(time, 3),
                'true_position': pos.tolist(),
                'measured_position': noisy_pos.tolist(),
                'velocity': [vx, vy],
                'scenario': 'straight_line'
            })
            
            time += self.dt
        
        return data
    
    def generate_2d_circular_motion(self, duration=3.0, radius=3.0, angular_velocity=1.0):
        """
        生成2D圆周运动数据
        
        参数:
            duration: 持续时间 (秒)
            radius: 半径 (米)
            angular_velocity: 角速度 (rad/s)
        """
        data = []
        time = 0.0
        
        while time <= duration:
            # 圆周运动位置
            theta = angular_velocity * time
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            pos = np.array([x, y])
            
            # 速度 (切向)
            vx = -radius * angular_velocity * np.sin(theta)
            vy = radius * angular_velocity * np.cos(theta)
            
            # 添加噪声并记录
            noisy_pos = self.add_noise(pos)
            data.append({
                'time': round(time, 3),
                'true_position': pos.tolist(),
                'measured_position': noisy_pos.tolist(),
                'velocity': [vx, vy],
                'angular_velocity': angular_velocity,
                'scenario': 'circular_motion'
            })
            
            time += self.dt
        
        return data
    
    def generate_2d_zigzag(self, duration=4.0, speed=3.0, turn_interval=0.5):
        """
        生成2D之字形机动数据
        
        参数:
            duration: 持续时间 (秒)
            speed: 速度 (m/s)
            turn_interval: 转向间隔 (秒)
        """
        data = []
        time = 0.0
        pos = np.array([0.0, 0.0])
        direction = 1  # 1 或 -1
        
        while time <= duration:
            # 每隔 turn_interval 改变方向
            if int(time / turn_interval) % 2 == 0:
                direction = 1
            else:
                direction = -1
            
            # 速度
            vx = speed
            vy = direction * speed * 0.5
            
            # 位置更新
            pos = pos + np.array([vx, vy]) * self.dt
            
            # 添加噪声并记录
            noisy_pos = self.add_noise(pos)
            data.append({
                'time': round(time, 3),
                'true_position': pos.tolist(),
                'measured_position': noisy_pos.tolist(),
                'velocity': [vx, vy],
                'scenario': 'zigzag'
            })
            
            time += self.dt
        
        return data
    
    # ==================== 3D 运动模式 ====================
    
    def generate_3d_straight_climb(self, duration=3.0, horizontal_speed=4.0, 
                                   vertical_speed=2.0, angle=30):
        """
        生成3D直线爬升数据
        
        参数:
            duration: 持续时间 (秒)
            horizontal_speed: 水平速度 (m/s)
            vertical_speed: 垂直速度 (m/s)
            angle: 水平方向角度 (度)
        """
        angle_rad = np.deg2rad(angle)
        vx = horizontal_speed * np.cos(angle_rad)
        vy = horizontal_speed * np.sin(angle_rad)
        vz = vertical_speed
        
        data = []
        time = 0.0
        pos = np.array([0.0, 0.0, 10.0])  # 从10米高度开始
        
        while time <= duration:
            # 位置更新
            pos = pos + np.array([vx, vy, vz]) * self.dt
            
            # 添加噪声并记录
            noisy_pos = self.add_noise(pos, self.noise_std * 1.2)  # 3D噪声稍大
            data.append({
                'time': round(time, 3),
                'true_position': pos.tolist(),
                'measured_position': noisy_pos.tolist(),
                'velocity': [vx, vy, vz],
                'scenario': '3d_straight_climb'
            })
            
            time += self.dt
        
        return data
    
    def generate_3d_helical_climb(self, duration=4.0, radius=2.5, 
                                  angular_velocity=0.8, climb_rate=1.5):
        """
        生成3D螺旋爬升数据
        
        参数:
            duration: 持续时间 (秒)
            radius: 螺旋半径 (米)
            angular_velocity: 角速度 (rad/s)
            climb_rate: 爬升速率 (m/s)
        """
        data = []
        time = 0.0
        z_start = 10.0
        
        while time <= duration:
            # 螺旋运动
            theta = angular_velocity * time
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = z_start + climb_rate * time
            pos = np.array([x, y, z])
            
            # 速度
            vx = -radius * angular_velocity * np.sin(theta)
            vy = radius * angular_velocity * np.cos(theta)
            vz = climb_rate
            
            # 添加噪声并记录
            noisy_pos = self.add_noise(pos, self.noise_std * 1.2)
            data.append({
                'time': round(time, 3),
                'true_position': pos.tolist(),
                'measured_position': noisy_pos.tolist(),
                'velocity': [vx, vy, vz],
                'scenario': '3d_helical_climb'
            })
            
            time += self.dt
        
        return data
    
    def generate_3d_aggressive_maneuver(self, duration=3.0):
        """
        生成3D剧烈机动数据（模拟FPV高机动性）
        
        参数:
            duration: 持续时间 (秒)
        """
        data = []
        time = 0.0
        pos = np.array([0.0, 0.0, 15.0])
        vel = np.array([3.0, 0.0, 1.0])
        
        while time <= duration:
            # 随机加速度（模拟剧烈机动）
            if time > 0.5 and time < 2.5:
                # 剧烈机动阶段
                acc = np.array([
                    np.sin(time * 5) * 8.0,
                    np.cos(time * 3) * 8.0,
                    np.sin(time * 2) * 3.0
                ])
            else:
                # 平稳阶段
                acc = np.array([0.0, 0.0, -0.5])  # 轻微下降
            
            # 更新速度和位置
            vel = vel + acc * self.dt
            pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
            
            # 限制速度
            speed = np.linalg.norm(vel)
            if speed > 15.0:
                vel = vel / speed * 15.0
            
            # 添加噪声并记录
            noisy_pos = self.add_noise(pos, self.noise_std * 1.5)
            data.append({
                'time': round(time, 3),
                'true_position': pos.tolist(),
                'measured_position': noisy_pos.tolist(),
                'velocity': vel.tolist(),
                'acceleration': acc.tolist(),
                'scenario': '3d_aggressive_maneuver'
            })
            
            time += self.dt
        
        return data
    
    def generate_3d_hover_to_dive(self, duration=3.0, dive_start_time=1.0):
        """
        生成3D悬停后俯冲数据
        
        参数:
            duration: 持续时间 (秒)
            dive_start_time: 开始俯冲的时间 (秒)
        """
        data = []
        time = 0.0
        pos = np.array([0.0, 0.0, 20.0])  # 从20米高度开始
        vel = np.array([0.0, 0.0, 0.0])
        
        while time <= duration:
            if time < dive_start_time:
                # 悬停阶段（轻微漂移）
                acc = np.random.randn(3) * 0.2
            else:
                # 俯冲阶段
                acc = np.array([5.0, 2.0, -10.0])  # 向前下方加速
            
            # 更新速度和位置
            vel = vel + acc * self.dt
            pos = pos + vel * self.dt + 0.5 * acc * self.dt**2
            
            # 添加噪声并记录
            noisy_pos = self.add_noise(pos, self.noise_std * 1.3)
            data.append({
                'time': round(time, 3),
                'true_position': pos.tolist(),
                'measured_position': noisy_pos.tolist(),
                'velocity': vel.tolist(),
                'acceleration': acc.tolist(),
                'scenario': '3d_hover_to_dive'
            })
            
            time += self.dt
        
        return data


def save_mock_data(data, filename, output_dir='mockdata'):
    """保存模拟数据到JSON文件"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已保存: {filepath} ({len(data)} 个数据点)")
    return filepath


def generate_all_scenarios():
    """生成所有场景的模拟数据"""
    print("="*80)
    print("FPV 运动轨迹模拟数据生成")
    print("="*80)
    
    generator = MockDataGenerator(dt=0.05, noise_std=0.12)
    
    scenarios = []
    
    # === 2D 场景 ===
    print("\n[2D 场景]")
    
    # 1. 直线运动
    data = generator.generate_2d_straight_line(duration=2.0, speed=2.5, angle=45)
    filepath = save_mock_data(data, '2d_straight_line.json')
    scenarios.append({
        'name': '2D 直线运动',
        'file': str(filepath),
        'dimension': '2D',
        'points': len(data),
        'duration': 2.0
    })
    
    # 2. 圆周运动
    data = generator.generate_2d_circular_motion(duration=3.0, radius=3.0, angular_velocity=1.0)
    filepath = save_mock_data(data, '2d_circular_motion.json')
    scenarios.append({
        'name': '2D 圆周运动',
        'file': str(filepath),
        'dimension': '2D',
        'points': len(data),
        'duration': 3.0
    })
    
    # 3. 之字形机动
    data = generator.generate_2d_zigzag(duration=4.0, speed=3.0, turn_interval=0.5)
    filepath = save_mock_data(data, '2d_zigzag.json')
    scenarios.append({
        'name': '2D 之字形机动',
        'file': str(filepath),
        'dimension': '2D',
        'points': len(data),
        'duration': 4.0
    })
    
    # === 3D 场景 ===
    print("\n[3D 场景]")
    
    # 4. 直线爬升
    data = generator.generate_3d_straight_climb(duration=3.0, horizontal_speed=4.0, 
                                                vertical_speed=2.0, angle=30)
    filepath = save_mock_data(data, '3d_straight_climb.json')
    scenarios.append({
        'name': '3D 直线爬升',
        'file': str(filepath),
        'dimension': '3D',
        'points': len(data),
        'duration': 3.0
    })
    
    # 5. 螺旋爬升
    data = generator.generate_3d_helical_climb(duration=4.0, radius=2.5, 
                                              angular_velocity=0.8, climb_rate=1.5)
    filepath = save_mock_data(data, '3d_helical_climb.json')
    scenarios.append({
        'name': '3D 螺旋爬升',
        'file': str(filepath),
        'dimension': '3D',
        'points': len(data),
        'duration': 4.0
    })
    
    # 6. 剧烈机动
    data = generator.generate_3d_aggressive_maneuver(duration=3.0)
    filepath = save_mock_data(data, '3d_aggressive_maneuver.json')
    scenarios.append({
        'name': '3D 剧烈机动',
        'file': str(filepath),
        'dimension': '3D',
        'points': len(data),
        'duration': 3.0
    })
    
    # 7. 悬停后俯冲
    data = generator.generate_3d_hover_to_dive(duration=3.0, dive_start_time=1.0)
    filepath = save_mock_data(data, '3d_hover_to_dive.json')
    scenarios.append({
        'name': '3D 悬停后俯冲',
        'file': str(filepath),
        'dimension': '3D',
        'points': len(data),
        'duration': 3.0
    })
    
    # 保存场景索引
    index_file = Path('mockdata') / 'index.json'
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': '2025-10-15',
            'scenarios': scenarios,
            'total_scenarios': len(scenarios)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 场景索引已保存: {index_file}")
    
    print("\n" + "="*80)
    print(f"✓ 完成! 共生成 {len(scenarios)} 个场景的模拟数据")
    print("="*80)
    
    # 打印汇总
    print("\n场景汇总:")
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i}. {scenario['name']:20s} | {scenario['dimension']} | "
              f"{scenario['points']:3d} 点 | {scenario['duration']:.1f}s")


if __name__ == '__main__':
    generate_all_scenarios()

