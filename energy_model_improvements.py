"""
基于 Raj & Murray (2020) 论文的能耗模型改进

核心改进：
1. 使用Liu et al. (2017)的真实功耗模型
2. 引入"最大航程速度"概念
3. 正确建模速度-续航-航程关系
4. 区分载货/空载的不同功耗特性
"""

import numpy as np
import math
from typing import Dict, List, Tuple


class RealisticDronePowerModel:
    """
    基于Liu et al. (2017)的真实无人机功耗模型
    
    功耗公式：
    - 起飞/降落: P^tl(w, v_ve) = k1*(W+w)*g*[v/2 + sqrt((v/2)^2 + (W+w)*g/k2^2)] + c2*((W+w)*g)^1.5
    - 水平巡航: P^c(w, v_ho) = (c1+c2)*[((W+w)*g - c5*(v*cos(α))^2)^2 + (c4*v^2)^2]^0.75 + c4*v^3
    - 悬停: P^h(w) = (c1+c2)*((W+w)*g)^1.5
    """
    
    # 模型系数 (Liu et al. 2017, Table 4)
    k1 = 0.8554      # [unitless]
    k2 = 0.3051      # sqrt(kg/m)
    c1 = 2.8037      # sqrt(m/kg)
    c2 = 0.3177      # sqrt(m/kg)
    c4 = 0.0296      # kg/m
    c5 = 0.0279      # N·s/m
    
    # 物理常数
    g = 9.8          # 重力加速度 m/s²
    alpha = 10.0     # 攻角（度）
    
    def __init__(
        self,
        drone_empty_weight: float = 1.5,    # 机身重量 kg
        battery_capacity: float = 500.0,     # 电池容量 kJ
        takeoff_speed: float = 10.0,         # 起飞速度 m/s
        landing_speed: float = 5.0,          # 降落速度 m/s
        cruise_altitude: float = 50.0,       # 巡航高度 m
    ):
        self.W = drone_empty_weight
        self.battery_capacity = battery_capacity
        self.v_takeoff = takeoff_speed
        self.v_landing = landing_speed
        self.h = cruise_altitude
        self.alpha_rad = math.radians(self.alpha)
    
    def power_takeoff_landing(self, parcel_weight: float, vertical_speed: float) -> float:
        """
        起飞/降落阶段功耗 (W)
        
        Args:
            parcel_weight: 包裹重量 (kg)
            vertical_speed: 垂直速度 (m/s)
        """
        total_weight = self.W + parcel_weight
        term1 = self.k1 * total_weight * self.g * (
            vertical_speed / 2 + 
            math.sqrt((vertical_speed / 2)**2 + total_weight * self.g / self.k2**2)
        )
        term2 = self.c2 * (total_weight * self.g)**1.5
        return term1 + term2
    
    def power_cruise(self, parcel_weight: float, horizontal_speed: float) -> float:
        """
        水平巡航阶段功耗 (W)
        
        Args:
            parcel_weight: 包裹重量 (kg)
            horizontal_speed: 水平速度 (m/s)
        """
        total_weight = self.W + parcel_weight
        
        # 升力项
        lift_term = (total_weight * self.g - 
                     self.c5 * (horizontal_speed * math.cos(self.alpha_rad))**2)**2
        # 阻力项
        drag_term = (self.c4 * horizontal_speed**2)**2
        
        power = ((self.c1 + self.c2) * (lift_term + drag_term)**0.75 + 
                 self.c4 * horizontal_speed**3)
        
        return power
    
    def power_hover(self, parcel_weight: float) -> float:
        """
        悬停阶段功耗 (W)
        """
        total_weight = self.W + parcel_weight
        return (self.c1 + self.c2) * (total_weight * self.g)**1.5
    
    def energy_per_meter(self, parcel_weight: float, speed: float) -> float:
        """
        单位距离能耗 (J/m = W·s/m)
        
        能耗率 = 功耗 / 速度
        """
        if speed <= 0:
            return float('inf')
        power = self.power_cruise(parcel_weight, speed)
        return power / speed  # J/m
    
    def flight_time_at_speed(self, parcel_weight: float, speed: float) -> float:
        """
        给定速度下的最大飞行时间 (s)
        
        续航时间 = 电池容量 / 巡航功耗
        """
        power = self.power_cruise(parcel_weight, speed)
        return (self.battery_capacity * 1000) / power  # 转换kJ -> J
    
    def flight_range_at_speed(self, parcel_weight: float, speed: float) -> float:
        """
        给定速度下的最大飞行距离 (m)
        
        航程 = 速度 × 续航时间
        """
        endurance = self.flight_time_at_speed(parcel_weight, speed)
        return speed * endurance
    
    def max_range_speed(self, parcel_weight: float, 
                        max_allowable_speed: float = 40.0,
                        search_resolution: float = 0.5) -> float:
        """
        计算最大航程速度
        
        最大航程速度是使飞行距离最大化的速度
        
        Args:
            parcel_weight: 包裹重量 (kg)
            max_allowable_speed: 最大允许速度 (m/s)
            search_resolution: 搜索分辨率 (m/s)
        
        Returns:
            最大航程速度 (m/s)
        """
        speeds = np.arange(1.0, max_allowable_speed + search_resolution, search_resolution)
        ranges = [self.flight_range_at_speed(parcel_weight, v) for v in speeds]
        
        max_range_idx = np.argmax(ranges)
        return speeds[max_range_idx]
    
    def sortie_energy(
        self,
        parcel_weight: float,
        outbound_distance: float,
        return_distance: float,
        outbound_speed: float,
        return_speed: float,
        service_time: float = 60.0,  # 服务时间 (s)
    ) -> Dict:
        """
        计算完整任务的能耗
        
        任务包括：起飞 -> 去程巡航 -> 降落 -> 服务 -> 起飞 -> 返程巡航 -> 悬停等待 -> 降落
        
        Returns:
            包含各阶段能耗的字典
        """
        # 1. 起飞（载货）
        E_takeoff1 = self.power_takeoff_landing(parcel_weight, self.v_takeoff) * (self.h / self.v_takeoff)
        
        # 2. 去程巡航（载货）
        E_outbound = self.power_cruise(parcel_weight, outbound_speed) * (outbound_distance / outbound_speed)
        
        # 3. 降落（载货）
        E_land1 = self.power_takeoff_landing(parcel_weight, self.v_landing) * (self.h / self.v_landing)
        
        # 4. 服务时间（忽略地面能耗）
        E_service = 0
        
        # 5. 起飞（空载）
        E_takeoff2 = self.power_takeoff_landing(0, self.v_takeoff) * (self.h / self.v_takeoff)
        
        # 6. 返程巡航（空载）
        E_return = self.power_cruise(0, return_speed) * (return_distance / return_speed)
        
        # 7. 降落（空载）
        E_land2 = self.power_takeoff_landing(0, self.v_landing) * (self.h / self.v_landing)
        
        total = E_takeoff1 + E_outbound + E_land1 + E_takeoff2 + E_return + E_land2
        
        return {
            'takeoff_loaded': E_takeoff1 / 1000,      # kJ
            'cruise_outbound': E_outbound / 1000,
            'landing_loaded': E_land1 / 1000,
            'takeoff_empty': E_takeoff2 / 1000,
            'cruise_return': E_return / 1000,
            'landing_empty': E_land2 / 1000,
            'total': total / 1000,                    # kJ
            'remaining': self.battery_capacity - total / 1000,
            'feasible': total / 1000 <= self.battery_capacity
        }
    
    def max_hover_time(self, parcel_weight: float, energy_used: float) -> float:
        """
        计算剩余能量可悬停的最大时间 (s)
        
        Args:
            parcel_weight: 包裹重量 (kg)
            energy_used: 已使用能量 (kJ)
        """
        remaining = (self.battery_capacity - energy_used) * 1000  # J
        if remaining <= 0:
            return 0
        return remaining / self.power_hover(parcel_weight)


def compute_energy_params_for_gurobi(
    power_model: RealisticDronePowerModel,
    speeds: List[float],
    max_payload: float = 5.0,
) -> Dict:
    """
    为Gurobi模型预计算能耗参数
    
    由于Gurobi难以直接处理非线性功耗公式，我们使用分段线性近似
    
    Args:
        power_model: 功耗模型实例
        speeds: 速度选项列表
        max_payload: 最大载重
    
    Returns:
        每个速度的能耗参数
    """
    params = {}
    
    for v in speeds:
        # 计算空载和满载时的单位距离能耗
        energy_empty = power_model.energy_per_meter(0, v)
        energy_loaded = power_model.energy_per_meter(max_payload, v)
        
        # 计算最大航程
        range_empty = power_model.flight_range_at_speed(0, v)
        range_loaded = power_model.flight_range_at_speed(max_payload, v)
        
        params[v] = {
            # 单位距离能耗 (J/m)
            'energy_per_meter_empty': energy_empty,
            'energy_per_meter_loaded': energy_loaded,
            
            # 最大航程 (m)
            'max_range_empty': range_empty,
            'max_range_loaded': range_loaded,
            
            # 最大续航时间 (s)
            'max_endurance_empty': power_model.flight_time_at_speed(0, v),
            'max_endurance_loaded': power_model.flight_time_at_speed(max_payload, v),
            
            # 巡航功率 (W)
            'power_empty': power_model.power_cruise(0, v),
            'power_loaded': power_model.power_cruise(max_payload, v),
        }
    
    return params


def get_optimal_speed_for_distance(
    power_model: RealisticDronePowerModel,
    distance: float,
    parcel_weight: float,
    available_speeds: List[float],
    max_allowable_speed: float = 40.0,
    truck_arrival_time: float = None,
    launch_time: float = 0,
) -> Tuple[float, Dict]:
    """
    根据距离和约束选择最优速度
    
    策略：
    1. 如果距离超过最大航程速度下的范围，选择最大航程速度
    2. 如果需要减少等待时间，尝试加速
    3. 否则选择最大航程速度以节省能耗
    
    Args:
        power_model: 功耗模型
        distance: 飞行距离 (m)
        parcel_weight: 包裹重量 (kg)
        available_speeds: 可选速度列表
        max_allowable_speed: 最大允许速度
        truck_arrival_time: 卡车预计到达回收点时间
        launch_time: 起飞时间
    
    Returns:
        最优速度和相关信息
    """
    # 计算最大航程速度
    max_range_speed = power_model.max_range_speed(parcel_weight, max_allowable_speed)
    
    # 检查每个速度的可行性
    feasible_speeds = []
    for v in available_speeds:
        if v > max_allowable_speed:
            continue
        max_range = power_model.flight_range_at_speed(parcel_weight, v)
        if max_range >= distance * 1.1:  # 留10%安全余量
            feasible_speeds.append(v)
    
    if not feasible_speeds:
        # 没有可行速度，选择最大航程速度（尽可能远）
        return max_range_speed, {'status': 'infeasible', 'reason': 'distance too far'}
    
    # 如果有卡车到达时间约束
    if truck_arrival_time is not None:
        # 计算需要的最小速度
        available_time = truck_arrival_time - launch_time
        if available_time > 0:
            required_speed = distance / available_time
            
            # 找到满足时间约束的最小速度
            for v in sorted(feasible_speeds):
                flight_time = distance / v
                if flight_time <= available_time:
                    return v, {
                        'status': 'optimal',
                        'reason': 'meet truck arrival',
                        'flight_time': flight_time,
                        'energy': power_model.energy_per_meter(parcel_weight, v) * distance / 1000
                    }
    
    # 默认选择最接近最大航程速度的可行速度（最节能）
    closest_to_max_range = min(feasible_speeds, key=lambda v: abs(v - max_range_speed))
    
    return closest_to_max_range, {
        'status': 'optimal',
        'reason': 'max range speed',
        'max_range_speed': max_range_speed,
        'selected_speed': closest_to_max_range,
        'energy': power_model.energy_per_meter(parcel_weight, closest_to_max_range) * distance / 1000
    }


# ==================== 用于更新您的Gurobi模型的代码 ====================

def create_improved_energy_params(drone_speeds: List[float], 
                                   drone_empty_weight: float = 1.5,
                                   battery_capacity: float = 500.0) -> Dict:
    """
    创建改进的能耗参数，可直接替换您模型中的 _compute_energy_params 方法
    
    使用方法：
    在您的 EnRTSPDVSModelSOCP 类中，将 _compute_energy_params 方法替换为：
    
    def _compute_energy_params(self) -> Dict:
        return create_improved_energy_params(
            self.drone_speeds,
            self.drone_empty_weight,
            self.battery_capacity
        )
    """
    power_model = RealisticDronePowerModel(
        drone_empty_weight=drone_empty_weight,
        battery_capacity=battery_capacity
    )
    
    params = {}
    
    for v in drone_speeds:
        # 空载参数
        empty_energy_rate = power_model.energy_per_meter(0, v)  # J/m
        empty_max_range = power_model.flight_range_at_speed(0, v)  # m
        
        # 满载参数 (使用5kg作为参考)
        ref_payload = 5.0
        loaded_energy_rate = power_model.energy_per_meter(ref_payload, v)
        loaded_max_range = power_model.flight_range_at_speed(ref_payload, v)
        
        # 计算线性近似参数
        # E = α + β*Q + γ*v²  (您原来的形式)
        # 但我们根据真实模型来设置这些值
        
        # α: 空载基础能耗率 (J/m)
        alpha = empty_energy_rate
        
        # β: 载荷对能耗的影响 (J/(m·kg))
        # 通过有限差分近似
        beta = (loaded_energy_rate - empty_energy_rate) / ref_payload
        
        # γ: 这个参数在真实模型中已经包含在功耗公式中
        # 设为0因为速度影响已经在α和β中体现
        gamma = 0
        
        params[v] = {
            'alpha': alpha / 1000,        # 转换为 kJ/m
            'beta': beta / 1000,          # 转换为 kJ/(m·kg)
            'gamma': gamma,
            
            # 新增：最大航程约束参数
            'max_range_empty': empty_max_range,
            'max_range_loaded': loaded_max_range,
            
            # 新增：用于速度选择启发式
            'is_near_max_range_speed': abs(v - power_model.max_range_speed(0, max(drone_speeds))) < 2,
        }
        
        print(f"速度 {v:5.1f} m/s: α={params[v]['alpha']:.4f}, β={params[v]['beta']:.4f}, "
              f"空载航程={empty_max_range/1000:.1f}km, 载货航程={loaded_max_range/1000:.1f}km")
    
    return params


def add_range_constraints_to_model(model, drone_trips, vars_dict, energy_params, 
                                    demands, safety_margin=0.1):
    """
    添加基于速度的航程约束到Gurobi模型
    
    这是关键改进：确保选择的速度能够覆盖所需的飞行距离
    
    使用方法：
    在 build() 方法中调用此函数
    """
    import gurobipy as gp
    from gurobipy import GRB
    
    y = vars_dict['y']
    v_out = vars_dict['v_out']
    v_ret = vars_dict['v_ret']
    r_out = vars_dict['r_out']
    r_ret = vars_dict['r_ret']
    
    V = list(energy_params.keys())
    
    for (i, j, k) in drone_trips:
        Q_k = demands.get(k, 1.0)
        
        for v in V:
            # 去程航程约束（载货）
            # 如果选择速度v，飞行距离不能超过该速度下的最大航程
            max_range_loaded = energy_params[v]['max_range_loaded'] * (1 - safety_margin)
            
            model.addConstr(
                r_out[i, j, k] <= max_range_loaded + 
                model.big_M * (1 - v_out[i, j, k, v]),
                name=f"range_out_{i}_{j}_{k}_{v}"
            )
            
            # 返程航程约束（空载）
            max_range_empty = energy_params[v]['max_range_empty'] * (1 - safety_margin)
            
            model.addConstr(
                r_ret[i, j, k] <= max_range_empty + 
                model.big_M * (1 - v_ret[i, j, k, v]),
                name=f"range_ret_{i}_{j}_{k}_{v}"
            )
    
    print(f"添加了 {len(drone_trips) * len(V) * 2} 个航程约束")


# ==================== 测试和演示 ====================

def demo_power_model():
    """演示功耗模型的特性"""
    print("=" * 70)
    print("Liu et al. (2017) 无人机功耗模型演示")
    print("=" * 70)
    
    model = RealisticDronePowerModel(
        drone_empty_weight=1.5,
        battery_capacity=500.0
    )
    
    speeds = [5, 10, 15, 18, 20, 25, 30, 35, 40]
    payloads = [0, 2, 5]
    
    print("\n1. 不同速度和载重下的功耗 (W)")
    print("-" * 60)
    print(f"{'速度(m/s)':<12}", end="")
    for w in payloads:
        print(f"载重{w}kg", end="     ")
    print()
    
    for v in speeds:
        print(f"{v:<12}", end="")
        for w in payloads:
            power = model.power_cruise(w, v)
            print(f"{power:8.1f}", end="    ")
        print()
    
    print("\n2. 不同速度和载重下的最大航程 (km)")
    print("-" * 60)
    print(f"{'速度(m/s)':<12}", end="")
    for w in payloads:
        print(f"载重{w}kg", end="     ")
    print()
    
    for v in speeds:
        print(f"{v:<12}", end="")
        for w in payloads:
            range_m = model.flight_range_at_speed(w, v)
            print(f"{range_m/1000:8.1f}", end="    ")
        print()
    
    print("\n3. 最大航程速度")
    print("-" * 60)
    for w in payloads:
        max_range_speed = model.max_range_speed(w, 40.0)
        max_range = model.flight_range_at_speed(w, max_range_speed)
        print(f"载重 {w} kg: 最大航程速度 = {max_range_speed:.1f} m/s, "
              f"最大航程 = {max_range/1000:.1f} km")
    
    print("\n4. 完整任务能耗示例")
    print("-" * 60)
    
    # 模拟一个任务
    sortie = model.sortie_energy(
        parcel_weight=2.0,
        outbound_distance=5000,    # 5km去程
        return_distance=4000,       # 4km返程
        outbound_speed=18.0,        # 去程速度
        return_speed=15.0,          # 返程速度
        service_time=60.0
    )
    
    print(f"任务参数: 去程5km@18m/s, 返程4km@15m/s, 载重2kg")
    print(f"  起飞(载货): {sortie['takeoff_loaded']:.2f} kJ")
    print(f"  去程巡航:   {sortie['cruise_outbound']:.2f} kJ")
    print(f"  降落(载货): {sortie['landing_loaded']:.2f} kJ")
    print(f"  起飞(空载): {sortie['takeoff_empty']:.2f} kJ")
    print(f"  返程巡航:   {sortie['cruise_return']:.2f} kJ")
    print(f"  降落(空载): {sortie['landing_empty']:.2f} kJ")
    print(f"  总能耗:     {sortie['total']:.2f} kJ / {model.battery_capacity:.0f} kJ")
    print(f"  可行性:     {'✓ 可行' if sortie['feasible'] else '✗ 不可行'}")


def demo_improved_params():
    """演示改进的参数计算"""
    print("\n" + "=" * 70)
    print("改进的Gurobi模型参数")
    print("=" * 70 + "\n")
    
    speeds = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    params = create_improved_energy_params(speeds)
    
    return params


if __name__ == "__main__":
    demo_power_model()
    params = demo_improved_params()
