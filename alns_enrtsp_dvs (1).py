"""
ALNS算法求解弧中同步的速度可变无人机与卡车配送优化问题 (enRTSP-DVS)
Adaptive Large Neighborhood Search for En-Route Synchronization 
Truck-Drone Problem with Drone Variable Speed

核心特点：
1. 无人机在卡车行驶的弧上某点起飞（途中起飞）
2. 无人机服务客户后在另一条弧上某点降落（途中降落）
3. λ参数表示起飞/降落点在弧上的位置比例[0,1]

Version 2: 修复途中起飞建模和卡车路线闭合问题
"""

import numpy as np
import random
import math
import copy
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ==================== 数据结构定义 ====================

@dataclass
class DroneMission:
    """
    无人机任务数据结构
    
    途中起飞模型：
    - 无人机在弧(launch_arc_start, launch_arc_end)上的lambda_launch位置起飞
    - 服务客户customer
    - 在弧(recover_arc_start, recover_arc_end)上的lambda_recover位置降落
    """
    # 起飞弧的起点和终点
    launch_arc_start: int
    launch_arc_end: int
    # 起飞点在弧上的位置 [0,1]
    lambda_launch: float
    
    # 服务的客户
    customer: int
    
    # 回收弧的起点和终点
    recover_arc_start: int
    recover_arc_end: int
    # 回收点在弧上的位置 [0,1]
    lambda_recover: float
    
    # 速度选择
    outbound_speed: float = 15.0  # 去程速度
    return_speed: float = 15.0    # 返程速度
    
    def copy(self):
        return DroneMission(
            launch_arc_start=self.launch_arc_start,
            launch_arc_end=self.launch_arc_end,
            lambda_launch=self.lambda_launch,
            customer=self.customer,
            recover_arc_start=self.recover_arc_start,
            recover_arc_end=self.recover_arc_end,
            lambda_recover=self.lambda_recover,
            outbound_speed=self.outbound_speed,
            return_speed=self.return_speed
        )


@dataclass
class Solution:
    """解的数据结构"""
    # 卡车路线：完整的闭合回路 [depot, c1, c2, ..., cn, depot]
    truck_route: List[int] = field(default_factory=list)
    # 无人机任务列表
    drone_missions: List[DroneMission] = field(default_factory=list)
    # 目标函数值
    objective: float = float('inf')
    
    def copy(self):
        new_sol = Solution()
        new_sol.truck_route = self.truck_route.copy()
        new_sol.drone_missions = [m.copy() for m in self.drone_missions]
        new_sol.objective = self.objective
        return new_sol
    
    def get_drone_served_customers(self) -> Set[int]:
        """获取无人机服务的客户集合"""
        return {m.customer for m in self.drone_missions}
    
    def get_truck_served_customers(self, depot: int = 0) -> Set[int]:
        """获取卡车服务的客户集合（排除仓库和无人机服务的客户）"""
        drone_served = self.get_drone_served_customers()
        return {c for c in self.truck_route if c != depot and c not in drone_served}
    
    def get_arcs(self) -> List[Tuple[int, int]]:
        """获取卡车路线的所有弧"""
        arcs = []
        for i in range(len(self.truck_route) - 1):
            arcs.append((self.truck_route[i], self.truck_route[i+1]))
        return arcs
    
    def is_closed_route(self, depot: int = 0) -> bool:
        """检查卡车路线是否闭合"""
        if len(self.truck_route) < 2:
            return False
        return self.truck_route[0] == depot and self.truck_route[-1] == depot


# ==================== 能耗模型 ====================

class SimpleDroneEnergyModel:
    """简化的无人机能耗模型"""
    
    def __init__(self, drone_empty_weight: float = 2.0, battery_capacity: float = 500.0):
        self.drone_empty_weight = drone_empty_weight
        self.battery_capacity = battery_capacity
        self.c0 = 50.0
        self.c1 = 200.0
        self.c2 = 0.01
    
    def power_at_speed(self, payload: float, speed: float) -> float:
        """计算给定载荷和速度下的功率 (W)"""
        total_weight = self.drone_empty_weight + payload
        weight_factor = (total_weight / self.drone_empty_weight) ** 1.5
        if speed < 1:
            speed = 1
        power = (self.c0 + self.c1 / speed + self.c2 * speed ** 3) * weight_factor
        return power
    
    def energy_for_distance(self, payload: float, speed: float, distance: float) -> float:
        """计算飞行给定距离所需能量 (Wh)"""
        if distance <= 0 or speed <= 0:
            return 0
        power = self.power_at_speed(payload, speed)
        time_hours = (distance / speed) / 3600
        return power * time_hours
    
    def max_range_at_speed(self, payload: float, speed: float) -> float:
        """计算给定速度下的最大航程"""
        power = self.power_at_speed(payload, speed)
        if power <= 0:
            return 0
        max_time_hours = self.battery_capacity / power
        return speed * max_time_hours * 3600
    
    def get_optimal_speed(self, payload: float, available_speeds: List[float]) -> float:
        """获取给定载荷下的最优速度（最大航程）"""
        best_speed = available_speeds[0]
        max_range = 0
        for speed in available_speeds:
            range_at_speed = self.max_range_at_speed(payload, speed)
            if range_at_speed > max_range:
                max_range = range_at_speed
                best_speed = speed
        return best_speed


# ==================== 问题实例类 ====================

class EnRTSPDVSInstance:
    """enRTSP-DVS问题实例"""
    
    def __init__(
        self,
        node_coords: Dict[int, Tuple[float, float]],
        customers: List[int],
        depot: int = 0,
        demands: Optional[Dict[int, float]] = None,
        time_windows: Optional[Dict[int, Tuple[float, float]]] = None,
        service_times: Optional[Dict[int, float]] = None,
        drone_speeds: List[float] = None,
        truck_speed: float = 10.0,
        battery_capacity: float = 500.0,
        drone_max_load: float = 50.0,
        drone_empty_weight: float = 2.0,
        truck_cost: float = 1.0,
        drone_cost: float = 0.5,
        energy_cost: float = 0.1,
        wait_cost: float = 0.5,
        safety_margin: float = 0.1
    ):
        self.node_coords = node_coords
        self.customers = customers
        self.depot = depot
        self.demands = demands or {c: 1.0 for c in customers}
        self.time_windows = time_windows or {c: (0, 1e6) for c in customers}
        self.service_times = service_times or {c: 0 for c in customers}
        
        self.drone_speeds = drone_speeds or [12.0, 15.0, 18.0, 22.0, 25.0, 30.0]
        self.truck_speed = truck_speed
        self.battery_capacity = battery_capacity
        self.drone_max_load = drone_max_load
        self.drone_empty_weight = drone_empty_weight
        
        self.truck_cost = truck_cost
        self.drone_cost = drone_cost
        self.energy_cost = energy_cost
        self.wait_cost = wait_cost
        self.safety_margin = safety_margin
        
        self._compute_distances()
        self.energy_model = SimpleDroneEnergyModel(drone_empty_weight, battery_capacity)
    
    def _compute_distances(self):
        """计算距离矩阵"""
        self.dist = {}
        all_nodes = [self.depot] + self.customers
        
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    xi, yi = self.node_coords[i]
                    xj, yj = self.node_coords[j]
                    self.dist[i, j] = math.sqrt((xi - xj)**2 + (yi - yj)**2)
        
        self.truck_time = {(i, j): self.dist[i, j] / self.truck_speed 
                          for (i, j) in self.dist}
    
    def get_point_on_arc(self, arc_start: int, arc_end: int, lam: float) -> Tuple[float, float]:
        """
        获取弧(arc_start, arc_end)上lambda位置的坐标
        lam=0 表示起点，lam=1 表示终点
        """
        x1, y1 = self.node_coords[arc_start]
        x2, y2 = self.node_coords[arc_end]
        x = x1 + lam * (x2 - x1)
        y = y1 + lam * (y2 - y1)
        return (x, y)
    
    def euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """计算两点之间的欧几里得距离"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def is_drone_eligible(self, customer: int) -> bool:
        """判断客户是否可以由无人机服务"""
        demand = self.demands.get(customer, 0)
        return demand <= self.drone_max_load

    def get_node_lambda_on_arc(self, node: int, arc_start: int, arc_end: int) -> float:
        """
        计算节点在弧上的lambda位置 [0,1]
        通过距离比例: lambda = dist(arc_start, node) / dist(arc_start, arc_end)
        """
        d_total = self.dist.get((arc_start, arc_end), 0)
        if d_total < 1e-9:
            return 0.0
        d_start_to_node = self.dist.get((arc_start, node), 0)
        return min(1.0, max(0.0, d_start_to_node / d_total))

    def calculate_truck_segment_time(
        self, route: List[int], launch_arc_idx: int, recover_arc_idx: int,
        lambda_launch: float, lambda_recover: float
    ) -> float:
        """
        计算卡车从起飞点到回收点的行驶时间

        Args:
            route: 卡车路线节点列表
            launch_arc_idx: 起飞弧在路线中的索引 (弧 route[idx]->route[idx+1])
            recover_arc_idx: 回收弧在路线中的索引
            lambda_launch: 起飞点在起飞弧上的位置 [0,1]
            lambda_recover: 回收点在回收弧上的位置 [0,1]
        """
        total_time = 0.0

        if launch_arc_idx == recover_arc_idx:
            # 起飞和回收在同一条弧上
            arc_start = route[launch_arc_idx]
            arc_end = route[launch_arc_idx + 1]
            full_time = self.truck_time.get((arc_start, arc_end), 0)
            total_time = (lambda_recover - lambda_launch) * full_time
            return max(0.0, total_time)

        # 起飞弧的剩余部分
        arc_s = route[launch_arc_idx]
        arc_e = route[launch_arc_idx + 1]
        full_time = self.truck_time.get((arc_s, arc_e), 0)
        total_time += (1 - lambda_launch) * full_time

        # 中间完整弧
        for i in range(launch_arc_idx + 1, recover_arc_idx):
            total_time += self.truck_time.get((route[i], route[i + 1]), 0)

        # 回收弧的前半部分
        arc_s = route[recover_arc_idx]
        arc_e = route[recover_arc_idx + 1]
        full_time = self.truck_time.get((arc_s, arc_e), 0)
        total_time += lambda_recover * full_time

        return total_time


# ==================== ALNS算法核心 ====================

class ALNSSolver:
    """ALNS求解器"""
    
    def __init__(
        self,
        instance: EnRTSPDVSInstance,
        seed: int = 42,
        max_iterations: int = 10000,
        segment_length: int = 100,
        initial_temperature: float = None,  # None = auto-calculate from initial solution
        temp_factor: float = 0.004,         # Sacramento: T_st = alpha * f(s_init)
        cooling_rate: float = 0.9995,       # Fallback geometric (linear cooling preferred)
        min_removal: int = 1,
        max_removal_ratio: float = 0.20,    # Sacramento: psi=0.15, recommended 0.15-0.20
        max_removal_cap: int = 40,          # Sacramento: c_lim=40
        reaction_factor: float = 0.9,       # Sacramento: lambda=0.9 (was 0.1, way too low)
        sigma1: float = 33,
        sigma2: float = 9,
        sigma3: float = 13,
        no_improve_max: int = 1000,         # Sacramento: restore best after N no-improve
    ):
        self.instance = instance
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.max_iterations = max_iterations
        self.segment_length = segment_length
        self.initial_temperature = initial_temperature
        self.temp_factor = temp_factor
        self.cooling_rate = cooling_rate
        self.min_removal = min_removal
        self.max_removal_ratio = max_removal_ratio
        self.max_removal_cap = max_removal_cap
        self.reaction_factor = reaction_factor
        self.no_improve_max = no_improve_max

        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3

        self._init_operators()

        # Cache frequently used computations
        self.sorted_speeds = sorted(self.instance.drone_speeds)
        self.battery_limit = self.instance.battery_capacity * (1 - self.instance.safety_margin)

        self.stats = {
            'iterations': 0,
            'improvements': 0,
            'accepts': 0,
            'best_found_iteration': 0,
            'restorations': 0
        }
    
    def _init_operators(self):
        """初始化破坏和修复算子"""
        self.destroy_operators = [
            ('random_removal', self._random_removal),
            ('worst_removal', self._worst_removal),
            ('shaw_removal', self._shaw_removal),
            ('drone_to_truck_removal', self._drone_to_truck_removal),
            ('cluster_removal', self._cluster_removal),
            ('speed_reoptimize', self._speed_reoptimize_removal),
        ]

        self.repair_operators = [
            ('greedy_insert', self._greedy_insert),
            ('regret_insert', self._regret_insert),
            ('drone_first_insert', self._drone_first_insert),
        ]
        
        self.destroy_weights = [1.0] * len(self.destroy_operators)
        self.repair_weights = [1.0] * len(self.repair_operators)
        
        self.destroy_scores = [0.0] * len(self.destroy_operators)
        self.repair_scores = [0.0] * len(self.repair_operators)
        self.destroy_usage = [0] * len(self.destroy_operators)
        self.repair_usage = [0] * len(self.repair_operators)
    
    def solve(self, verbose: bool = True) -> Solution:
        """
        执行ALNS求解

        改进 (基于 Sacramento et al., 2019):
        - 自适应初始温度: T_st = alpha * f(s_init)
        - 线性冷却: T = T_st * (1 - iter/max_iter)
        - 恢复机制: 连续no_improve_max次无改进后恢复到最优解
        - 反应因子 lambda=0.9 (快速适应算子性能)
        - 破坏比例 psi=0.20 (适度破坏)
        """
        start_time = time.time()

        if verbose:
            print("=" * 60)
            print("ALNS求解 enRTSP-DVS 问题 (途中起飞模型)")
            print("=" * 60)
            print("\n[1] 构造初始解...")

        current_solution = self._construct_initial_solution()
        best_solution = current_solution.copy()

        if verbose:
            print(f"    初始解目标值: {current_solution.objective:.4f}")
            print(f"    卡车路线闭合: {current_solution.is_closed_route(self.instance.depot)}")
            print(f"    卡车服务客户数: {len(current_solution.get_truck_served_customers(self.instance.depot))}")
            print(f"    无人机任务数: {len(current_solution.drone_missions)}")

        # 自适应初始温度 (Sacramento: T_st = alpha * f(s_init))
        if self.initial_temperature is None:
            temperature = self.temp_factor * current_solution.objective
            if len(self.instance.customers) < 20:
                temperature *= 1.1  # Sacramento: 小实例提高10%避免过早收敛
        else:
            temperature = self.initial_temperature
        initial_temp = temperature

        if verbose:
            print(f"    初始温度: {temperature:.4f}")
            print(f"\n[2] 开始ALNS迭代 (最大迭代: {self.max_iterations})...")

        iter_no_improve = 0  # 连续无改进计数器

        for iteration in range(self.max_iterations):
            destroy_idx = self._select_operator(self.destroy_weights)
            repair_idx = self._select_operator(self.repair_weights)

            destroy_name, destroy_op = self.destroy_operators[destroy_idx]
            repair_name, repair_op = self.repair_operators[repair_idx]

            new_solution = current_solution.copy()
            removed_customers = destroy_op(new_solution)
            repair_op(new_solution, removed_customers)

            # 确保路线闭合
            self._ensure_closed_route(new_solution)

            self._evaluate_solution(new_solution)

            self.destroy_usage[destroy_idx] += 1
            self.repair_usage[repair_idx] += 1

            score = 0
            if new_solution.objective < best_solution.objective:
                best_solution = new_solution.copy()
                current_solution = new_solution
                score = self.sigma1
                self.stats['improvements'] += 1
                self.stats['best_found_iteration'] = iteration
                iter_no_improve = 0

                if verbose and iteration % 500 == 0:
                    print(f"    迭代 {iteration}: 新最优解 {best_solution.objective:.4f}")

            elif new_solution.objective < current_solution.objective:
                current_solution = new_solution
                score = self.sigma2
                self.stats['accepts'] += 1
                iter_no_improve = 0

            elif self._accept_worse(new_solution.objective - current_solution.objective, temperature):
                current_solution = new_solution
                score = self.sigma3
                self.stats['accepts'] += 1
                iter_no_improve += 1

            else:
                iter_no_improve += 1

            self.destroy_scores[destroy_idx] += score
            self.repair_scores[repair_idx] += score

            if (iteration + 1) % self.segment_length == 0:
                self._update_weights()

            # 线性冷却 (Sacramento: T = T_st * (1 - t_elap/t_max))
            progress = (iteration + 1) / self.max_iterations
            temperature = initial_temp * (1 - progress)
            temperature = max(temperature, 1e-6)  # 避免温度为0

            # 恢复机制 (Sacramento: noImprovMax=1000)
            if iter_no_improve >= self.no_improve_max:
                current_solution = best_solution.copy()
                iter_no_improve = 0
                self.stats['restorations'] = self.stats.get('restorations', 0) + 1
                if verbose:
                    print(f"    迭代 {iteration}: 恢复到最优解 (连续{self.no_improve_max}次无改进)")

            self.stats['iterations'] = iteration + 1

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\n[3] ALNS求解完成!")
            print(f"    最终目标值: {best_solution.objective:.4f}")
            print(f"    总迭代次数: {self.stats['iterations']}")
            print(f"    改进次数: {self.stats['improvements']}")
            print(f"    最优解迭代: {self.stats['best_found_iteration']}")
            print(f"    恢复次数: {self.stats.get('restorations', 0)}")
            print(f"    求解时间: {elapsed_time:.2f}秒")

        return best_solution
    
    def _ensure_closed_route(self, solution: Solution):
        """确保卡车路线是闭合的"""
        depot = self.instance.depot
        
        # 确保以仓库开始
        if not solution.truck_route or solution.truck_route[0] != depot:
            if depot in solution.truck_route:
                solution.truck_route.remove(depot)
            solution.truck_route.insert(0, depot)
        
        # 确保以仓库结束
        if solution.truck_route[-1] != depot:
            solution.truck_route.append(depot)
        
        # 移除中间的重复仓库
        new_route = [depot]
        for i in range(1, len(solution.truck_route) - 1):
            if solution.truck_route[i] != depot:
                new_route.append(solution.truck_route[i])
        new_route.append(depot)
        solution.truck_route = new_route
    
    # ==================== 初始解构造 ====================
    
    def _construct_initial_solution(self) -> Solution:
        """构造初始解"""
        solution = Solution()
        depot = self.instance.depot
        
        # 使用最近邻启发式构造初始卡车路线
        unvisited = set(self.instance.customers)
        solution.truck_route = [depot]
        current = depot
        
        while unvisited:
            nearest = min(unvisited, key=lambda c: self.instance.dist.get((current, c), float('inf')))
            solution.truck_route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # 闭合路线
        solution.truck_route.append(depot)
        
        # 2-opt改进
        self._two_opt_improvement(solution)
        
        # 评估解
        self._evaluate_solution(solution)
        
        # 尝试将部分客户分配给无人机
        self._initial_drone_assignment(solution)
        
        return solution
    
    def _two_opt_improvement(self, solution: Solution):
        """2-opt改进卡车路线"""
        route = solution.truck_route
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    a, b = route[i-1], route[i]
                    c, d = route[j], route[j+1]
                    
                    current = self.instance.dist.get((a, b), 0) + self.instance.dist.get((c, d), 0)
                    new = self.instance.dist.get((a, c), 0) + self.instance.dist.get((b, d), 0)
                    
                    if new < current - 0.001:
                        route[i:j+1] = route[i:j+1][::-1]
                        improved = True
                        break
                if improved:
                    break
    
    def _initial_drone_assignment(self, solution: Solution):
        """
        初始解中的无人机分配

        改进: 使用完整路线弧验证, 允许同弧起飞/回收 (launch_idx <= recover_idx)
        """
        arcs = solution.get_arcs()

        if len(arcs) < 1:
            return

        # 找出适合无人机服务的客户
        candidates = []

        for customer in self.instance.customers:
            if not self.instance.is_drone_eligible(customer):
                continue

            if customer not in solution.truck_route:
                continue

            customer_idx = solution.truck_route.index(customer)

            best_mission, saving = self._find_best_drone_mission(solution, customer, customer_idx)

            if best_mission is not None and saving > 0:
                candidates.append((customer, customer_idx, best_mission, saving))

        # 按节省排序
        candidates.sort(key=lambda x: x[3], reverse=True)

        # 分配无人机任务
        assigned_customers = set()
        for customer, idx, mission, saving in candidates:
            if customer not in assigned_customers:
                # 在完整路线弧中验证（含无人机客户节点）
                current_arcs = solution.get_arcs()
                launch_arc = (mission.launch_arc_start, mission.launch_arc_end)
                recover_arc = (mission.recover_arc_start, mission.recover_arc_end)

                if launch_arc in current_arcs and recover_arc in current_arcs:
                    launch_idx = current_arcs.index(launch_arc)
                    recover_idx = current_arcs.index(recover_arc)

                    if launch_idx <= recover_idx:  # 允许同弧（途中起飞同弧回收）
                        solution.drone_missions.append(mission)
                        assigned_customers.add(customer)

        self._evaluate_solution(solution)
    
    def _find_best_drone_mission(self, solution: Solution, customer: int,
                                  customer_idx: int) -> Tuple[Optional[DroneMission], float]:
        """
        为客户找到最佳无人机任务（支持真正的途中起飞/降落和变速策略）

        核心思路: 基于移除客户后的卡车路线搜索起飞/回收点,
        确保无人机真正从弧的中间起飞和降落，而非在客户节点上。

        性能优化:
        - 弧搜索窗口限制为 ±5 (平衡质量与速度)
        - 两阶段lambda搜索: 粗筛5点 -> 精搜仅对top-3候选
        - 智能速度选择: 先用目标时间估算最佳速度，只测3个候选
        """
        route = solution.truck_route

        if customer_idx <= 0 or customer_idx >= len(route) - 1:
            return None, 0

        best_mission = None
        best_saving = 0

        # 计算从卡车路线移除客户的节省
        prev_node = route[customer_idx - 1]
        next_node = route[customer_idx + 1]

        truck_detour_cost = (
            self.instance.dist.get((prev_node, customer), 0) +
            self.instance.dist.get((customer, next_node), 0) -
            self.instance.dist.get((prev_node, next_node), 0)
        ) * self.instance.truck_cost

        customer_coords = self.instance.node_coords[customer]
        payload = self.instance.demands.get(customer, 0)
        service_time = self.instance.service_times.get(customer, 0)

        # 构建移除客户后的路线和弧
        reduced_route = [n for n in route if n != customer]
        if len(reduced_route) < 2:
            return None, 0
        reduced_arcs = [(reduced_route[i], reduced_route[i+1])
                        for i in range(len(reduced_route) - 1)]
        num_arcs = len(reduced_arcs)
        if num_arcs < 1:
            return None, 0

        # 找到客户在reduced_route中应该在的位置附近
        # (prev_node和next_node在reduced_route中相邻)
        center_idx = 0
        for i in range(len(reduced_arcs)):
            if reduced_arcs[i] == (prev_node, next_node):
                center_idx = i
                break

        # 限制搜索范围: 中心 ±5 弧
        ARC_WINDOW = 5
        arc_start = max(0, center_idx - ARC_WINDOW)
        arc_end = min(num_arcs, center_idx + ARC_WINDOW + 1)

        # 排序的速度列表（用于智能速度选择）
        sorted_speeds = sorted(self.instance.drone_speeds)
        battery_limit = self.instance.battery_capacity * (1 - self.instance.safety_margin)

        # 阶段1: 粗筛lambda + 快速速度选择 -> 收集候选
        coarse_lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]
        candidates = []  # (saving, launch_idx, recover_idx, lam_l, lam_r, speed_out, speed_ret)

        for launch_arc_idx in range(arc_start, arc_end):
            launch_arc = reduced_arcs[launch_arc_idx]
            for recover_arc_idx in range(launch_arc_idx, min(num_arcs, launch_arc_idx + ARC_WINDOW + 1)):
                recover_arc = reduced_arcs[recover_arc_idx]

                for lam_launch in coarse_lambdas:
                    launch_point = self.instance.get_point_on_arc(
                        launch_arc[0], launch_arc[1], lam_launch)
                    dist_out = self.instance.euclidean_distance(launch_point, customer_coords)

                    for lam_recover in coarse_lambdas:
                        if launch_arc_idx == recover_arc_idx and lam_recover <= lam_launch:
                            continue

                        recover_point = self.instance.get_point_on_arc(
                            recover_arc[0], recover_arc[1], lam_recover)
                        dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)

                        # 卡车段时间
                        truck_seg_time = self.instance.calculate_truck_segment_time(
                            reduced_route, launch_arc_idx, recover_arc_idx, lam_launch, lam_recover)

                        # 智能速度选择: 找匹配truck_seg_time的速度
                        best_combo = self._select_best_speed_combo(
                            payload, dist_out, dist_ret, service_time,
                            truck_seg_time, battery_limit, sorted_speeds)

                        if best_combo is None:
                            continue

                        s_out, s_ret, total_cost = best_combo
                        saving = truck_detour_cost - total_cost

                        if saving > 0:
                            candidates.append((saving, launch_arc_idx, recover_arc_idx,
                                             lam_launch, lam_recover, s_out, s_ret))

                        if saving > best_saving:
                            best_saving = saving
                            best_mission = DroneMission(
                                launch_arc_start=launch_arc[0],
                                launch_arc_end=launch_arc[1],
                                lambda_launch=lam_launch,
                                customer=customer,
                                recover_arc_start=recover_arc[0],
                                recover_arc_end=recover_arc[1],
                                lambda_recover=lam_recover,
                                outbound_speed=s_out,
                                return_speed=s_ret
                            )

        # 阶段2: 对top候选做精细lambda搜索
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            fine_lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

            for _, la_idx, re_idx, _, _, _, _ in candidates[:3]:
                launch_arc = reduced_arcs[la_idx]
                recover_arc = reduced_arcs[re_idx]

                for lam_launch in fine_lambdas:
                    launch_point = self.instance.get_point_on_arc(
                        launch_arc[0], launch_arc[1], lam_launch)
                    dist_out = self.instance.euclidean_distance(launch_point, customer_coords)

                    for lam_recover in fine_lambdas:
                        if la_idx == re_idx and lam_recover <= lam_launch:
                            continue

                        recover_point = self.instance.get_point_on_arc(
                            recover_arc[0], recover_arc[1], lam_recover)
                        dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)

                        truck_seg_time = self.instance.calculate_truck_segment_time(
                            reduced_route, la_idx, re_idx, lam_launch, lam_recover)

                        best_combo = self._select_best_speed_combo(
                            payload, dist_out, dist_ret, service_time,
                            truck_seg_time, battery_limit, sorted_speeds)

                        if best_combo is None:
                            continue

                        s_out, s_ret, total_cost = best_combo
                        saving = truck_detour_cost - total_cost

                        if saving > best_saving:
                            best_saving = saving
                            best_mission = DroneMission(
                                launch_arc_start=launch_arc[0],
                                launch_arc_end=launch_arc[1],
                                lambda_launch=lam_launch,
                                customer=customer,
                                recover_arc_start=recover_arc[0],
                                recover_arc_end=recover_arc[1],
                                lambda_recover=lam_recover,
                                outbound_speed=s_out,
                                return_speed=s_ret
                            )

        return best_mission, best_saving

    def _select_best_speed_combo(
        self, payload: float, dist_out: float, dist_ret: float,
        service_time: float, truck_seg_time: float,
        battery_limit: float, sorted_speeds: List[float]
    ) -> Optional[Tuple[float, float, float]]:
        """
        智能速度选择: 基于目标同步时间选择最佳速度组合

        策略:
        1. 计算目标飞行时间 = truck_seg_time - service_time
        2. 找最接近目标时间的速度（同步优先）
        3. 同时测试最快速度（最小等待）和最节能速度
        返回: (speed_out, speed_ret, total_cost) 或 None
        """
        target_flight_time = max(0.1, truck_seg_time - service_time)
        total_dist = dist_out + dist_ret

        best_result = None
        best_cost = float('inf')

        # 候选速度策略
        speed_candidates = set()

        # 策略1: 匹配目标时间的速度
        if total_dist > 0 and target_flight_time > 0:
            target_speed = total_dist / target_flight_time
            # 找最接近的可用速度
            closest = min(sorted_speeds, key=lambda v: abs(v - target_speed))
            speed_candidates.add(closest)
            # 也加上相邻速度
            idx = sorted_speeds.index(closest)
            if idx > 0:
                speed_candidates.add(sorted_speeds[idx - 1])
            if idx < len(sorted_speeds) - 1:
                speed_candidates.add(sorted_speeds[idx + 1])

        # 策略2: 最快速度（最小飞行时间）
        speed_candidates.add(sorted_speeds[-1])

        # 策略3: 最慢可行速度（最节能）
        speed_candidates.add(sorted_speeds[0])

        # 测试所有候选组合
        for s_out in speed_candidates:
            if dist_out > 0:
                energy_out = self.instance.energy_model.energy_for_distance(payload, s_out, dist_out)
                time_out = dist_out / s_out
            else:
                energy_out = 0
                time_out = 0

            for s_ret in speed_candidates:
                if dist_ret > 0:
                    energy_ret = self.instance.energy_model.energy_for_distance(0, s_ret, dist_ret)
                    time_ret = dist_ret / s_ret
                else:
                    energy_ret = 0
                    time_ret = 0

                total_energy = energy_out + energy_ret
                if total_energy > battery_limit:
                    continue

                drone_time = time_out + service_time + time_ret
                wait_time = abs(drone_time - truck_seg_time)

                cost = ((dist_out + dist_ret) * self.instance.drone_cost +
                        total_energy * self.instance.energy_cost +
                        wait_time * self.instance.wait_cost)

                if cost < best_cost:
                    best_cost = cost
                    best_result = (s_out, s_ret, cost)

        return best_result
    
    # ==================== 破坏算子 ====================
    
    def _random_removal(self, solution: Solution) -> List[int]:
        """随机移除算子"""
        num_remove = self._get_removal_count(len(self.instance.customers))
        all_customers = list(self.instance.customers)
        removed = self.rng.sample(all_customers, min(num_remove, len(all_customers)))
        self._remove_customers(solution, removed)
        return removed
    
    def _worst_removal(self, solution: Solution) -> List[int]:
        """最差移除算子"""
        num_remove = self._get_removal_count(len(self.instance.customers))
        
        savings = []
        for customer in self.instance.customers:
            original_obj = solution.objective
            test_solution = solution.copy()
            self._remove_customers(test_solution, [customer])
            self._ensure_closed_route(test_solution)
            self._evaluate_solution(test_solution)
            saving = original_obj - test_solution.objective
            savings.append((customer, saving))
        
        savings.sort(key=lambda x: x[1], reverse=True)
        
        removed = []
        for customer, _ in savings:
            if len(removed) >= num_remove:
                break
            if self.rng.random() < 0.8:
                removed.append(customer)
        
        self._remove_customers(solution, removed)
        return removed
    
    def _shaw_removal(self, solution: Solution) -> List[int]:
        """Shaw移除算子"""
        num_remove = self._get_removal_count(len(self.instance.customers))
        
        seed_customer = self.rng.choice(self.instance.customers)
        removed = [seed_customer]
        
        similarities = []
        for customer in self.instance.customers:
            if customer == seed_customer:
                continue
            dist = self.instance.dist.get((seed_customer, customer), float('inf'))
            demand_diff = abs(self.instance.demands.get(seed_customer, 0) - 
                            self.instance.demands.get(customer, 0))
            similarity = dist + 0.1 * demand_diff
            similarities.append((customer, similarity))
        
        similarities.sort(key=lambda x: x[1])
        
        for customer, _ in similarities:
            if len(removed) >= num_remove:
                break
            if self.rng.random() < 0.9:
                removed.append(customer)
        
        self._remove_customers(solution, removed)
        return removed
    
    def _drone_to_truck_removal(self, solution: Solution) -> List[int]:
        """将无人机服务的客户移回卡车"""
        if not solution.drone_missions:
            return self._random_removal(solution)
        
        num_remove = min(self._get_removal_count(len(self.instance.customers)), 
                        len(solution.drone_missions))
        
        missions_to_remove = self.rng.sample(solution.drone_missions, num_remove)
        removed = [m.customer for m in missions_to_remove]
        
        solution.drone_missions = [m for m in solution.drone_missions 
                                   if m not in missions_to_remove]
        
        return removed
    
    def _cluster_removal(self, solution: Solution) -> List[int]:
        """聚类移除"""
        num_remove = self._get_removal_count(len(self.instance.customers))
        
        center = self.rng.choice(self.instance.customers)
        center_coords = self.instance.node_coords[center]
        
        distances = []
        for customer in self.instance.customers:
            coords = self.instance.node_coords[customer]
            dist = math.sqrt((coords[0] - center_coords[0])**2 + 
                           (coords[1] - center_coords[1])**2)
            distances.append((customer, dist))
        
        distances.sort(key=lambda x: x[1])
        removed = [c for c, _ in distances[:num_remove]]
        
        self._remove_customers(solution, removed)
        return removed
    
    def _speed_reoptimize_removal(self, solution: Solution) -> List[int]:
        """
        速度重优化算子: 移除无人机任务中的客户并重新分配,
        目的是通过重新搜索获得更好的速度组合和lambda位置
        """
        if not solution.drone_missions:
            return self._random_removal(solution)

        # 随机选取一部分无人机任务的客户
        num_remove = min(
            max(1, len(solution.drone_missions)),
            self._get_removal_count(len(self.instance.customers))
        )
        missions_to_remove = self.rng.sample(
            solution.drone_missions, min(num_remove, len(solution.drone_missions)))
        removed = [m.customer for m in missions_to_remove]

        # 移除这些drone任务
        solution.drone_missions = [m for m in solution.drone_missions
                                   if m not in missions_to_remove]

        # 同时从卡车路线移除这些客户（由repair重新插入）
        depot = self.instance.depot
        solution.truck_route = [n for n in solution.truck_route
                               if n == depot or n not in removed]

        return removed

    def _remove_customers(self, solution: Solution, customers: List[int]):
        """从解中移除客户"""
        depot = self.instance.depot

        # 从卡车路线中移除
        solution.truck_route = [n for n in solution.truck_route
                               if n == depot or n not in customers]

        # 移除相关的无人机任务
        solution.drone_missions = [m for m in solution.drone_missions
                                   if m.customer not in customers]

    def _get_removal_count(self, total_customers: int) -> int:
        """计算移除客户数量"""
        max_removal = max(self.min_removal, int(total_customers * self.max_removal_ratio))
        return self.rng.randint(self.min_removal, max_removal)
    
    # ==================== 修复算子 ====================
    
    def _greedy_insert(self, solution: Solution, customers: List[int]):
        """贪婪插入算子"""
        for customer in customers:
            best_pos = None
            best_cost = float('inf')
            
            for pos in range(1, len(solution.truck_route)):
                cost = self._calculate_insertion_cost(solution, customer, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            
            if best_pos is not None:
                solution.truck_route.insert(best_pos, customer)
        
        self._ensure_closed_route(solution)
        self._evaluate_solution(solution)
    
    def _regret_insert(self, solution: Solution, customers: List[int]):
        """后悔插入算子"""
        remaining = customers.copy()
        
        while remaining:
            regrets = []
            for customer in remaining:
                costs = []
                for pos in range(1, len(solution.truck_route)):
                    cost = self._calculate_insertion_cost(solution, customer, pos)
                    costs.append((pos, cost))
                
                costs.sort(key=lambda x: x[1])
                
                if len(costs) >= 2:
                    regret = costs[1][1] - costs[0][1]
                elif len(costs) == 1:
                    regret = 0
                else:
                    regret = float('inf')
                
                best_pos = costs[0][0] if costs else 1
                regrets.append((customer, regret, best_pos))
            
            regrets.sort(key=lambda x: x[1], reverse=True)
            customer, _, pos = regrets[0]
            
            solution.truck_route.insert(pos, customer)
            remaining.remove(customer)
        
        self._ensure_closed_route(solution)
        self._evaluate_solution(solution)
    
    def _drone_first_insert(self, solution: Solution, customers: List[int]):
        """
        优先尝试无人机服务的修复算子

        改进: 先在当前路线上尝试无人机任务（不插入客户到卡车路线），
              成功则仅添加drone mission；失败则回退到卡车插入。
        """
        remaining = customers.copy()
        drone_assigned = set()

        # 阶段1: 对无人机可服务的客户，基于当前卡车路线尝试直接建立任务
        for customer in customers[:]:
            if not self.instance.is_drone_eligible(customer):
                continue

            # 基于当前路线寻找最佳drone任务（不需要customer在路线中）
            mission, saving = self._find_best_drone_mission_external(
                solution, customer)

            if mission is not None and saving > 0:
                solution.drone_missions.append(mission)
                drone_assigned.add(customer)
                remaining.remove(customer)
            else:
                # 无人机不划算，先插入卡车路线再尝试
                best_pos = 1
                best_cost = float('inf')
                for pos in range(1, len(solution.truck_route)):
                    cost = self._calculate_insertion_cost(solution, customer, pos)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = pos

                solution.truck_route.insert(best_pos, customer)
                self._ensure_closed_route(solution)

                # 插入后再次尝试创建无人机任务
                customer_idx = solution.truck_route.index(customer)
                mission2, saving2 = self._find_best_drone_mission(
                    solution, customer, customer_idx)

                if mission2 is not None and saving2 > 0:
                    solution.drone_missions.append(mission2)
                    drone_assigned.add(customer)

                remaining.remove(customer)

        # 阶段2: 剩余客户贪婪插入卡车路线
        for customer in remaining:
            best_pos = 1
            best_cost = float('inf')
            for pos in range(1, len(solution.truck_route)):
                cost = self._calculate_insertion_cost(solution, customer, pos)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
            solution.truck_route.insert(best_pos, customer)

        self._ensure_closed_route(solution)
        self._evaluate_solution(solution)

    def _find_best_drone_mission_external(
        self, solution: Solution, customer: int
    ) -> Tuple[Optional[DroneMission], float]:
        """
        为不在卡车路线中的客户寻找最佳无人机任务

        直接基于当前卡车路线的弧结构搜索起飞/回收点,
        不需要客户在路线中（真正的外部分配）

        性能优化: 限制弧窗口 + 智能速度选择
        """
        route = solution.truck_route
        arcs = solution.get_arcs()
        num_arcs = len(arcs)

        if num_arcs < 1:
            return None, 0

        best_mission = None
        best_cost = float('inf')

        customer_coords = self.instance.node_coords[customer]
        payload = self.instance.demands.get(customer, 0)
        service_time = self.instance.service_times.get(customer, 0)

        sorted_speeds = sorted(self.instance.drone_speeds)
        battery_limit = self.instance.battery_capacity * (1 - self.instance.safety_margin)

        # 找离客户最近的弧作为搜索中心
        best_center = 0
        best_center_dist = float('inf')
        for i, (a, b) in enumerate(arcs):
            mid = self.instance.get_point_on_arc(a, b, 0.5)
            d = self.instance.euclidean_distance(mid, customer_coords)
            if d < best_center_dist:
                best_center_dist = d
                best_center = i

        ARC_WINDOW = 5
        arc_start = max(0, best_center - ARC_WINDOW)
        arc_end = min(num_arcs, best_center + ARC_WINDOW + 1)

        coarse_lambdas = [0.0, 0.3, 0.5, 0.7, 1.0]

        for launch_arc_idx in range(arc_start, arc_end):
            launch_arc = arcs[launch_arc_idx]
            for recover_arc_idx in range(launch_arc_idx, min(num_arcs, launch_arc_idx + ARC_WINDOW + 1)):
                recover_arc = arcs[recover_arc_idx]

                for lam_launch in coarse_lambdas:
                    launch_point = self.instance.get_point_on_arc(
                        launch_arc[0], launch_arc[1], lam_launch)
                    dist_out = self.instance.euclidean_distance(launch_point, customer_coords)

                    for lam_recover in coarse_lambdas:
                        if launch_arc_idx == recover_arc_idx and lam_recover <= lam_launch:
                            continue

                        recover_point = self.instance.get_point_on_arc(
                            recover_arc[0], recover_arc[1], lam_recover)
                        dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)

                        truck_seg_time = self.instance.calculate_truck_segment_time(
                            route, launch_arc_idx, recover_arc_idx, lam_launch, lam_recover)

                        combo = self._select_best_speed_combo(
                            payload, dist_out, dist_ret, service_time,
                            truck_seg_time, battery_limit, sorted_speeds)

                        if combo is None:
                            continue

                        s_out, s_ret, cost = combo

                        if cost < best_cost:
                            best_cost = cost
                            best_mission = DroneMission(
                                launch_arc_start=launch_arc[0],
                                launch_arc_end=launch_arc[1],
                                lambda_launch=lam_launch,
                                customer=customer,
                                recover_arc_start=recover_arc[0],
                                recover_arc_end=recover_arc[1],
                                lambda_recover=lam_recover,
                                outbound_speed=s_out,
                                return_speed=s_ret
                            )

        if best_mission is not None:
            return best_mission, max(0, 50 - best_cost)
        return None, 0
    
    def _calculate_insertion_cost(self, solution: Solution, customer: int, pos: int) -> float:
        """计算在指定位置插入客户的成本"""
        if pos <= 0 or pos >= len(solution.truck_route):
            return float('inf')
        
        prev_node = solution.truck_route[pos - 1]
        next_node = solution.truck_route[pos]
        
        current_dist = self.instance.dist.get((prev_node, next_node), 0)
        new_dist = (self.instance.dist.get((prev_node, customer), float('inf')) + 
                   self.instance.dist.get((customer, next_node), float('inf')))
        
        return (new_dist - current_dist) * self.instance.truck_cost
    
    # ==================== 解评估 ====================
    
    def _evaluate_solution(self, solution: Solution):
        """
        评估解的目标函数值

        改进: 使用完整路线弧(含无人机客户)验证任务 + 同步等待成本
        """
        total_cost = 0
        depot = self.instance.depot

        # 获取无人机服务的客户
        drone_served = solution.get_drone_served_customers()

        # 构建实际的卡车路线（跳过无人机服务的客户）
        actual_truck_route = []
        for node in solution.truck_route:
            if node == depot or node not in drone_served:
                actual_truck_route.append(node)

        # 确保路线闭合
        if actual_truck_route and actual_truck_route[0] != depot:
            actual_truck_route.insert(0, depot)
        if actual_truck_route and actual_truck_route[-1] != depot:
            actual_truck_route.append(depot)

        # 实际卡车弧（不含无人机客户）-- 任务弧基于此验证
        actual_arcs = [(actual_truck_route[i], actual_truck_route[i+1])
                       for i in range(len(actual_truck_route) - 1)]

        # 1. 卡车运输成本
        for i in range(len(actual_truck_route) - 1):
            dist = self.instance.dist.get((actual_truck_route[i], actual_truck_route[i+1]), 0)
            total_cost += dist * self.instance.truck_cost

        # 2. 无人机成本（基于实际卡车弧验证）
        for mission in solution.drone_missions:
            launch_arc = (mission.launch_arc_start, mission.launch_arc_end)
            recover_arc = (mission.recover_arc_start, mission.recover_arc_end)

            # 在实际卡车弧中验证（任务是基于reduced route构建的）
            if launch_arc not in actual_arcs:
                total_cost += 1e5
                continue
            if recover_arc not in actual_arcs:
                total_cost += 1e5
                continue

            launch_idx = actual_arcs.index(launch_arc)
            recover_idx = actual_arcs.index(recover_arc)

            if launch_idx > recover_idx:
                total_cost += 1e5
                continue

            # 计算飞行距离
            customer_coords = self.instance.node_coords[mission.customer]
            launch_point = self.instance.get_point_on_arc(
                mission.launch_arc_start, mission.launch_arc_end, mission.lambda_launch)
            recover_point = self.instance.get_point_on_arc(
                mission.recover_arc_start, mission.recover_arc_end, mission.lambda_recover)

            dist_out = self.instance.euclidean_distance(launch_point, customer_coords)
            dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)

            # 距离成本
            total_cost += (dist_out + dist_ret) * self.instance.drone_cost

            # 能耗成本
            payload = self.instance.demands.get(mission.customer, 0)
            energy_out = self.instance.energy_model.energy_for_distance(
                payload, mission.outbound_speed, dist_out)
            energy_ret = self.instance.energy_model.energy_for_distance(
                0, mission.return_speed, dist_ret)
            total_cost += (energy_out + energy_ret) * self.instance.energy_cost

            # 同步等待成本
            service_time = self.instance.service_times.get(mission.customer, 0)
            drone_time = (dist_out / mission.outbound_speed +
                          service_time +
                          dist_ret / mission.return_speed)
            truck_segment_time = self.instance.calculate_truck_segment_time(
                actual_truck_route, launch_idx, recover_idx,
                mission.lambda_launch, mission.lambda_recover)
            wait_time = abs(drone_time - truck_segment_time)
            total_cost += wait_time * self.instance.wait_cost

        # 3. 检查可行性
        served = solution.get_truck_served_customers(depot) | drone_served
        unserved = set(self.instance.customers) - served
        if unserved:
            total_cost += len(unserved) * 1e6
        
        solution.objective = total_cost
    
    # ==================== 自适应权重更新 ====================
    
    def _select_operator(self, weights: List[float]) -> int:
        """基于轮盘赌选择算子"""
        total = sum(weights)
        r = self.rng.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(weights) - 1
    
    def _update_weights(self):
        """更新算子权重"""
        for i in range(len(self.destroy_operators)):
            if self.destroy_usage[i] > 0:
                performance = self.destroy_scores[i] / self.destroy_usage[i]
                self.destroy_weights[i] = (self.destroy_weights[i] * (1 - self.reaction_factor) + 
                                          self.reaction_factor * max(performance, 0.1))
        
        for i in range(len(self.repair_operators)):
            if self.repair_usage[i] > 0:
                performance = self.repair_scores[i] / self.repair_usage[i]
                self.repair_weights[i] = (self.repair_weights[i] * (1 - self.reaction_factor) + 
                                         self.reaction_factor * max(performance, 0.1))
        
        self.destroy_scores = [0.0] * len(self.destroy_operators)
        self.repair_scores = [0.0] * len(self.repair_operators)
        self.destroy_usage = [0] * len(self.destroy_operators)
        self.repair_usage = [0] * len(self.repair_operators)
    
    def _accept_worse(self, delta: float, temperature: float) -> bool:
        """模拟退火接受准则"""
        if temperature <= 0:
            return False
        prob = math.exp(-delta / temperature)
        return self.rng.random() < prob
    
    # ==================== 结果输出 ====================
    
    def print_solution(self, solution: Solution):
        """打印解的详细信息"""
        print("\n" + "=" * 60)
        print("ALNS求解结果 (途中起飞模型)")
        print("=" * 60)
        
        depot = self.instance.depot
        drone_served = solution.get_drone_served_customers()
        
        # 构建实际卡车路线
        actual_truck_route = []
        for node in solution.truck_route:
            if node == depot or node not in drone_served:
                actual_truck_route.append(node)
        
        # 计算统计
        truck_dist = 0
        for i in range(len(actual_truck_route) - 1):
            truck_dist += self.instance.dist.get(
                (actual_truck_route[i], actual_truck_route[i+1]), 0)
        
        drone_dist = 0
        total_energy = 0
        for mission in solution.drone_missions:
            customer_coords = self.instance.node_coords[mission.customer]
            launch_point = self.instance.get_point_on_arc(
                mission.launch_arc_start, mission.launch_arc_end, mission.lambda_launch)
            recover_point = self.instance.get_point_on_arc(
                mission.recover_arc_start, mission.recover_arc_end, mission.lambda_recover)
            
            dist_out = self.instance.euclidean_distance(launch_point, customer_coords)
            dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)
            drone_dist += dist_out + dist_ret
            
            payload = self.instance.demands.get(mission.customer, 0)
            energy_out = self.instance.energy_model.energy_for_distance(
                payload, mission.outbound_speed, dist_out)
            energy_ret = self.instance.energy_model.energy_for_distance(
                0, mission.return_speed, dist_ret)
            total_energy += energy_out + energy_ret
        
        print(f"\n目标函数值: {solution.objective:.4f}")
        print(f"卡车总行驶距离: {truck_dist:.2f}")
        print(f"无人机总飞行距离: {drone_dist:.2f}")
        print(f"总能耗: {total_energy:.2f} Wh")
        
        # 打印卡车路线
        print(f"\n卡车实际路线 ({len(actual_truck_route)} 节点):")
        route_str = " -> ".join(map(str, actual_truck_route))
        print(f"  {route_str}")
        
        # 检查闭合
        if actual_truck_route[0] == depot and actual_truck_route[-1] == depot:
            print(f"  ✓ 路线已闭合 (仓库{depot} -> ... -> 仓库{depot})")
        else:
            print(f"  ✗ 警告: 路线未闭合!")
        
        truck_served = solution.get_truck_served_customers(depot)
        print(f"  卡车服务的客户: {sorted(truck_served)}")
        
        # 打印无人机任务
        # 速度使用统计
        speed_usage = defaultdict(int)
        total_wait = 0.0
        mid_arc_count = 0

        print(f"\n无人机任务 ({len(solution.drone_missions)} 个):")
        # 构建actual arcs for sync calculation (与 _evaluate_solution 一致)
        actual_arcs_for_print = [(actual_truck_route[i], actual_truck_route[i+1])
                                 for i in range(len(actual_truck_route) - 1)]

        for idx, mission in enumerate(solution.drone_missions, 1):
            customer_coords = self.instance.node_coords[mission.customer]
            launch_point = self.instance.get_point_on_arc(
                mission.launch_arc_start, mission.launch_arc_end, mission.lambda_launch)
            recover_point = self.instance.get_point_on_arc(
                mission.recover_arc_start, mission.recover_arc_end, mission.lambda_recover)

            dist_out = self.instance.euclidean_distance(launch_point, customer_coords)
            dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)

            payload = self.instance.demands.get(mission.customer, 0)
            energy_out = self.instance.energy_model.energy_for_distance(
                payload, mission.outbound_speed, dist_out)
            energy_ret = self.instance.energy_model.energy_for_distance(
                0, mission.return_speed, dist_ret)

            # 同步计算
            service_time = self.instance.service_times.get(mission.customer, 0)
            drone_time = (dist_out / mission.outbound_speed + service_time +
                          dist_ret / mission.return_speed)
            launch_arc = (mission.launch_arc_start, mission.launch_arc_end)
            recover_arc = (mission.recover_arc_start, mission.recover_arc_end)
            l_idx = actual_arcs_for_print.index(launch_arc) if launch_arc in actual_arcs_for_print else 0
            r_idx = actual_arcs_for_print.index(recover_arc) if recover_arc in actual_arcs_for_print else l_idx
            truck_seg_time = self.instance.calculate_truck_segment_time(
                actual_truck_route, l_idx, r_idx,
                mission.lambda_launch, mission.lambda_recover)
            wait = abs(drone_time - truck_seg_time)
            total_wait += wait

            # 途中起飞检测
            is_mid_launch = 0 < mission.lambda_launch < 1
            is_mid_recover = 0 < mission.lambda_recover < 1
            if is_mid_launch or is_mid_recover:
                mid_arc_count += 1

            speed_usage[mission.outbound_speed] += 1
            speed_usage[mission.return_speed] += 1

            print(f"\n  任务 {idx}: 服务客户 {mission.customer} (需求: {payload})")
            print(f"    起飞: 弧({mission.launch_arc_start}->{mission.launch_arc_end})上 λ={mission.lambda_launch:.2f}"
                  f"{'  [途中起飞]' if is_mid_launch else ''}")
            print(f"           坐标: ({launch_point[0]:.1f}, {launch_point[1]:.1f})")
            print(f"    降落: 弧({mission.recover_arc_start}->{mission.recover_arc_end})上 λ={mission.lambda_recover:.2f}"
                  f"{'  [途中降落]' if is_mid_recover else ''}")
            print(f"           坐标: ({recover_point[0]:.1f}, {recover_point[1]:.1f})")
            print(f"    去程: {dist_out:.1f}m @ {mission.outbound_speed:.1f}m/s"
                  f"  (飞行时间: {dist_out / mission.outbound_speed:.1f}s)")
            print(f"    返程: {dist_ret:.1f}m @ {mission.return_speed:.1f}m/s"
                  f"  (飞行时间: {dist_ret / mission.return_speed:.1f}s)")
            print(f"    能耗: {energy_out + energy_ret:.2f} Wh")
            print(f"    同步: 无人机={drone_time:.1f}s, 卡车段={truck_seg_time:.1f}s, "
                  f"等待={wait:.1f}s")

        # 汇总统计
        if solution.drone_missions:
            print(f"\n  --- 无人机任务汇总 ---")
            print(f"  途中起飞/降落任务数: {mid_arc_count}/{len(solution.drone_missions)}")
            print(f"  总同步等待时间: {total_wait:.1f}s "
                  f"(平均: {total_wait / len(solution.drone_missions):.1f}s)")
            print(f"  速度使用分布: ", end="")
            for spd in sorted(speed_usage.keys()):
                print(f"{spd:.0f}m/s({speed_usage[spd]}次) ", end="")
            print()
            # 速度多样性
            unique_speeds = len(speed_usage)
            print(f"  使用了 {unique_speeds}/{len(self.instance.drone_speeds)} 种不同速度")


# ==================== 可视化 ====================

def visualize_solution(instance: EnRTSPDVSInstance, solution: Solution, 
                       save_path: str = None):
    """可视化解 - 正确显示途中起飞/降落"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib未安装，跳过可视化")
        return
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    depot = instance.depot
    drone_served = solution.get_drone_served_customers()
    
    # 构建实际卡车路线
    actual_truck_route = []
    for node in solution.truck_route:
        if node == depot or node not in drone_served:
            actual_truck_route.append(node)
    
    # 绘制仓库
    depot_coord = instance.node_coords[depot]
    ax.scatter(depot_coord[0], depot_coord[1], c='red', s=300, marker='s', 
               zorder=10, label='Depot')
    ax.annotate('Depot', depot_coord, fontsize=12, ha='center', va='bottom',
               fontweight='bold')
    
    # 绘制客户节点
    for customer in instance.customers:
        coord = instance.node_coords[customer]
        if customer in drone_served:
            ax.scatter(coord[0], coord[1], c='blue', s=150, marker='o', zorder=8)
            ax.annotate(f'{customer}(D)', coord, fontsize=9, ha='center', va='bottom')
        else:
            ax.scatter(coord[0], coord[1], c='green', s=150, marker='o', zorder=8)
            ax.annotate(f'{customer}(T)', coord, fontsize=9, ha='center', va='bottom')
    
    # 绘制卡车路线
    for i in range(len(actual_truck_route) - 1):
        from_node = actual_truck_route[i]
        to_node = actual_truck_route[i + 1]
        from_coord = instance.node_coords[from_node]
        to_coord = instance.node_coords[to_node]
        
        ax.annotate('', xy=to_coord, xytext=from_coord,
                   arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    
    # 绘制无人机任务
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    for idx, mission in enumerate(solution.drone_missions):
        color = colors[idx % len(colors)]
        
        # 计算起飞点和降落点
        launch_point = instance.get_point_on_arc(
            mission.launch_arc_start, mission.launch_arc_end, mission.lambda_launch)
        recover_point = instance.get_point_on_arc(
            mission.recover_arc_start, mission.recover_arc_end, mission.lambda_recover)
        customer_coord = instance.node_coords[mission.customer]
        
        # 绘制起飞点（小三角形）
        ax.scatter(launch_point[0], launch_point[1], c=color, s=100, marker='^', 
                  zorder=9, edgecolors='black', linewidths=1)
        
        # 绘制降落点（小三角形倒置）
        ax.scatter(recover_point[0], recover_point[1], c=color, s=100, marker='v', 
                  zorder=9, edgecolors='black', linewidths=1)
        
        # 去程（起飞点 -> 客户）
        ax.annotate('', xy=customer_coord, xytext=launch_point,
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, 
                                  linestyle='--'))
        
        # 返程（客户 -> 降落点）
        ax.annotate('', xy=recover_point, xytext=customer_coord,
                   arrowprops=dict(arrowstyle='->', color=color, lw=2, 
                                  linestyle='--'))
    
    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=15, label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=12, label='Truck customer'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=12, label='Drone customer'),
        Line2D([0], [0], color='green', lw=2, label='Truck route'),
        Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Drone flight'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
               markersize=10, label='Launch point'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', 
               markersize=10, label='Recovery point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    ax.set_title(f'enRTSP-DVS Solution (En-Route Launch/Recovery)\n'
                f'Objective: {solution.objective:.2f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
    return fig


# ==================== 实例生成 ====================

def create_drone_friendly_instance(num_customers: int = 12, seed: int = 42) -> EnRTSPDVSInstance:
    """创建对无人机友好的实例"""
    random.seed(seed)
    np.random.seed(seed)
    
    depot_coord = (400, 400)
    node_coords = {0: depot_coord}
    customers = []
    demands = {}
    time_windows = {}
    service_times = {}
    
    # 主路线客户（60%）
    main_route_customers = int(num_customers * 0.6)
    for i in range(1, main_route_customers + 1):
        angle = 2 * math.pi * (i - 1) / main_route_customers
        radius = 200 + random.uniform(-20, 20)
        x = depot_coord[0] + radius * math.cos(angle)
        y = depot_coord[1] + radius * math.sin(angle)
        
        node_coords[i] = (x, y)
        customers.append(i)
        demands[i] = random.randint(20, 40)
        time_windows[i] = (0, 10000)
        service_times[i] = 10
    
    # 偏离路线的客户（适合无人机）
    for i in range(main_route_customers + 1, num_customers + 1):
        ref_customer = random.randint(1, main_route_customers)
        ref_coord = node_coords[ref_customer]
        
        offset_dist = random.uniform(100, 200)
        dx = ref_coord[0] - depot_coord[0]
        dy = ref_coord[1] - depot_coord[1]
        perp_angle = math.atan2(dy, dx) + math.pi/2 + random.uniform(-0.5, 0.5)
        
        x = ref_coord[0] + offset_dist * math.cos(perp_angle)
        y = ref_coord[1] + offset_dist * math.sin(perp_angle)
        
        node_coords[i] = (x, y)
        customers.append(i)
        demands[i] = random.randint(3, 10)
        time_windows[i] = (0, 10000)
        service_times[i] = 5
    
    return EnRTSPDVSInstance(
        node_coords=node_coords,
        customers=customers,
        depot=0,
        demands=demands,
        time_windows=time_windows,
        service_times=service_times,
        drone_speeds=[12.0, 15.0, 18.0, 22.0, 25.0, 30.0],
        truck_speed=8.0,
        battery_capacity=800.0,
        drone_max_load=50.0,
        truck_cost=2.0,
        drone_cost=0.2,
        energy_cost=0.01,
    )


def get_truck_only_solution(instance: EnRTSPDVSInstance) -> Solution:
    """获取纯卡车配送解"""
    solution = Solution()
    depot = instance.depot
    
    unvisited = set(instance.customers)
    solution.truck_route = [depot]
    current = depot
    
    while unvisited:
        nearest = min(unvisited, key=lambda c: instance.dist.get((current, c), float('inf')))
        solution.truck_route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    solution.truck_route.append(depot)
    
    # 2-opt
    improved = True
    while improved:
        improved = False
        route = solution.truck_route
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                a, b = route[i-1], route[i]
                c, d = route[j], route[j+1]
                current_cost = instance.dist.get((a, b), 0) + instance.dist.get((c, d), 0)
                new_cost = instance.dist.get((a, c), 0) + instance.dist.get((b, d), 0)
                if new_cost < current_cost - 0.001:
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
                    break
            if improved:
                break
    
    # 计算成本
    total_cost = 0
    for i in range(len(solution.truck_route) - 1):
        dist = instance.dist.get((solution.truck_route[i], solution.truck_route[i+1]), 0)
        total_cost += dist * instance.truck_cost
    solution.objective = total_cost
    
    return solution


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("=" * 70)
    print("ALNS算法求解 enRTSP-DVS 问题")
    print("(弧中同步的速度可变无人机与卡车配送优化问题 - 途中起飞模型)")
    print("=" * 70)
    
    # 配置
    NUM_CUSTOMERS = 15
    SEED = 42
    MAX_ITERATIONS = 8000
    
    # 创建实例
    print("\n[1] 创建问题实例...")
    instance = create_drone_friendly_instance(num_customers=NUM_CUSTOMERS, seed=SEED)
    
    print(f"    客户数量: {len(instance.customers)}")
    print(f"    仓库位置: {instance.node_coords[instance.depot]}")
    print(f"    卡车成本: {instance.truck_cost}")
    print(f"    无人机成本: {instance.drone_cost}")
    
    # 求解
    print("\n[2] 初始化ALNS求解器...")
    solver = ALNSSolver(
        instance=instance,
        seed=SEED,
        max_iterations=MAX_ITERATIONS,
        initial_temperature=100.0,
        cooling_rate=0.9995,
    )
    
    print("\n[3] 开始求解...")
    solution = solver.solve(verbose=True)
    
    # 打印结果
    solver.print_solution(solution)
    
    # 对比实验
    print("\n" + "=" * 60)
    print("[4] 对比实验...")
    truck_only = get_truck_only_solution(instance)
    print(f"\n  纯卡车配送成本: {truck_only.objective:.2f}")
    print(f"  ALNS求解成本: {solution.objective:.2f}")
    if truck_only.objective > 0:
        improvement = (truck_only.objective - solution.objective) / truck_only.objective * 100
        print(f"  改进百分比: {improvement:.2f}%")
    
    # 可视化
    print("\n[5] 生成可视化...")
    try:
        visualize_solution(instance, solution, 
                          save_path=r"D:\研究生、\python\途中起飞+变速")
    except Exception as e:
        print(f"可视化失败: {e}")
    
    return instance, solution, solver


if __name__ == "__main__":
    instance, solution, solver = main()
