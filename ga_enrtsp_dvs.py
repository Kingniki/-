"""
遗传算法求解弧中同步的速度可变无人机与卡车配送优化问题 (enRTSP-DVS)
Genetic Algorithm for En-Route Synchronization Truck-Drone Problem
with Drone Variable Speed

核心特点：
1. 无人机在卡车行驶的弧上某点起飞（途中起飞）
2. 无人机服务客户后在另一条弧上某点降落（途中降落）
3. λ参数表示起飞/降落点在弧上的位置比例[0,1]

遗传算法组件：
- 染色体编码：卡车路线排列 + 无人机任务分配
- 选择：锦标赛选择 / 轮盘赌选择
- 交叉：顺序交叉(OX) / 部分映射交叉(PMX)
- 变异：交换变异、插入变异、2-opt变异、无人机任务变异
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
    launch_arc_start: int
    launch_arc_end: int
    lambda_launch: float
    customer: int
    recover_arc_start: int
    recover_arc_end: int
    lambda_recover: float
    outbound_speed: float = 15.0
    return_speed: float = 15.0

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
    truck_route: List[int] = field(default_factory=list)
    drone_missions: List[DroneMission] = field(default_factory=list)
    objective: float = float('inf')

    def copy(self):
        new_sol = Solution()
        new_sol.truck_route = self.truck_route.copy()
        new_sol.drone_missions = [m.copy() for m in self.drone_missions]
        new_sol.objective = self.objective
        return new_sol

    def get_drone_served_customers(self) -> Set[int]:
        return {m.customer for m in self.drone_missions}

    def get_truck_served_customers(self, depot: int = 0) -> Set[int]:
        drone_served = self.get_drone_served_customers()
        return {c for c in self.truck_route if c != depot and c not in drone_served}

    def get_arcs(self) -> List[Tuple[int, int]]:
        arcs = []
        for i in range(len(self.truck_route) - 1):
            arcs.append((self.truck_route[i], self.truck_route[i+1]))
        return arcs

    def is_closed_route(self, depot: int = 0) -> bool:
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
        total_weight = self.drone_empty_weight + payload
        weight_factor = (total_weight / self.drone_empty_weight) ** 1.5
        if speed < 1:
            speed = 1
        power = (self.c0 + self.c1 / speed + self.c2 * speed ** 3) * weight_factor
        return power

    def energy_for_distance(self, payload: float, speed: float, distance: float) -> float:
        if distance <= 0 or speed <= 0:
            return 0
        power = self.power_at_speed(payload, speed)
        time_hours = (distance / speed) / 3600
        return power * time_hours

    def max_range_at_speed(self, payload: float, speed: float) -> float:
        power = self.power_at_speed(payload, speed)
        if power <= 0:
            return 0
        max_time_hours = self.battery_capacity / power
        return speed * max_time_hours * 3600

    def get_optimal_speed(self, payload: float, available_speeds: List[float]) -> float:
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
        safety_margin: float = 0.1,
        drone_prep_time: float = 10.0
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
        self.drone_prep_time = drone_prep_time

        self._compute_distances()
        self.energy_model = SimpleDroneEnergyModel(drone_empty_weight, battery_capacity)

    def _compute_distances(self):
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
        x1, y1 = self.node_coords[arc_start]
        x2, y2 = self.node_coords[arc_end]
        x = x1 + lam * (x2 - x1)
        y = y1 + lam * (y2 - y1)
        return (x, y)

    def euclidean_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def is_drone_eligible(self, customer: int) -> bool:
        demand = self.demands.get(customer, 0)
        return demand <= self.drone_max_load

    def calculate_truck_segment_time(
        self, route: List[int], launch_arc_idx: int, recover_arc_idx: int,
        lambda_launch: float, lambda_recover: float
    ) -> float:
        total_time = 0.0

        if launch_arc_idx == recover_arc_idx:
            arc_start = route[launch_arc_idx]
            arc_end = route[launch_arc_idx + 1]
            full_time = self.truck_time.get((arc_start, arc_end), 0)
            total_time = (lambda_recover - lambda_launch) * full_time
            return max(0.0, total_time)

        arc_s = route[launch_arc_idx]
        arc_e = route[launch_arc_idx + 1]
        full_time = self.truck_time.get((arc_s, arc_e), 0)
        total_time += (1 - lambda_launch) * full_time

        for i in range(launch_arc_idx + 1, recover_arc_idx):
            total_time += self.truck_time.get((route[i], route[i + 1]), 0)

        arc_s = route[recover_arc_idx]
        arc_e = route[recover_arc_idx + 1]
        full_time = self.truck_time.get((arc_s, arc_e), 0)
        total_time += lambda_recover * full_time

        return total_time


# ==================== 染色体编码 ====================

@dataclass
class Chromosome:
    """
    染色体结构

    编码方式：
    - truck_sequence: 卡车访问客户的顺序（不含仓库）
    - drone_assignments: 无人机服务的客户集合
    - drone_missions: 解码后的无人机任务列表
    """
    truck_sequence: List[int] = field(default_factory=list)
    drone_assignments: Set[int] = field(default_factory=set)
    drone_missions: List[DroneMission] = field(default_factory=list)
    fitness: float = float('inf')

    def copy(self):
        new_chrom = Chromosome()
        new_chrom.truck_sequence = self.truck_sequence.copy()
        new_chrom.drone_assignments = self.drone_assignments.copy()
        new_chrom.drone_missions = [m.copy() for m in self.drone_missions]
        new_chrom.fitness = self.fitness
        return new_chrom

    def to_solution(self, depot: int = 0) -> Solution:
        """将染色体解码为Solution"""
        solution = Solution()
        solution.truck_route = [depot] + self.truck_sequence + [depot]
        solution.drone_missions = [m.copy() for m in self.drone_missions]
        solution.objective = self.fitness
        return solution


# ==================== 遗传算法求解器 ====================

class GASolver:
    """遗传算法求解器"""

    def __init__(
        self,
        instance: EnRTSPDVSInstance,
        seed: int = 42,
        population_size: int = 100,
        max_generations: int = 500,
        elite_size: int = 5,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        tournament_size: int = 5,
        selection_method: str = "tournament",  # "tournament" or "roulette"
        crossover_method: str = "ox",  # "ox" or "pmx"
    ):
        self.instance = instance
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.population_size = population_size
        self.max_generations = max_generations
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.crossover_method = crossover_method

        # 缓存
        self.sorted_speeds = sorted(self.instance.drone_speeds)
        self.battery_limit = self.instance.battery_capacity * (1 - self.instance.safety_margin)

        self.stats = {
            'generations': 0,
            'best_fitness': float('inf'),
            'best_generation': 0,
            'avg_fitness_history': [],
            'best_fitness_history': [],
        }

    # ==================== 种群初始化 ====================

    def _initialize_population(self) -> List[Chromosome]:
        """初始化种群"""
        population = []

        for i in range(self.population_size):
            chrom = self._create_random_chromosome()
            self._decode_and_evaluate(chrom)
            population.append(chrom)

        return population

    def _create_random_chromosome(self) -> Chromosome:
        """创建随机染色体"""
        chrom = Chromosome()

        # 随机排列客户顺序
        customers = self.instance.customers.copy()
        self.rng.shuffle(customers)
        chrom.truck_sequence = customers

        # 随机选择一部分客户由无人机服务
        eligible_customers = [c for c in customers if self.instance.is_drone_eligible(c)]
        num_drone = self.rng.randint(0, min(len(eligible_customers), len(customers) // 2))
        if num_drone > 0:
            drone_customers = self.rng.sample(eligible_customers, num_drone)
            chrom.drone_assignments = set(drone_customers)

        return chrom

    def _create_greedy_chromosome(self) -> Chromosome:
        """创建贪婪染色体（最近邻）"""
        chrom = Chromosome()
        depot = self.instance.depot

        unvisited = set(self.instance.customers)
        route = []
        current = depot

        while unvisited:
            nearest = min(unvisited, key=lambda c: self.instance.dist.get((current, c), float('inf')))
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        chrom.truck_sequence = route
        return chrom

    # ==================== 解码与评估 ====================

    def _decode_and_evaluate(self, chrom: Chromosome):
        """解码染色体并评估适应度"""
        depot = self.instance.depot

        # 构建卡车路线（包含所有客户）
        full_route = [depot] + chrom.truck_sequence + [depot]

        # 构建实际卡车路线（排除无人机服务的客户）
        actual_route = [depot]
        for c in chrom.truck_sequence:
            if c not in chrom.drone_assignments:
                actual_route.append(c)
        actual_route.append(depot)

        # 为无人机客户分配任务
        chrom.drone_missions = []
        actual_arcs = [(actual_route[i], actual_route[i+1]) for i in range(len(actual_route) - 1)]

        # 按在路线中的位置排序无人机客户
        drone_customers_ordered = []
        for c in chrom.truck_sequence:
            if c in chrom.drone_assignments:
                # 找到客户在原始序列中的位置
                pos = chrom.truck_sequence.index(c)
                drone_customers_ordered.append((pos, c))
        drone_customers_ordered.sort()

        # 为每个无人机客户找最佳任务
        for _, customer in drone_customers_ordered:
            mission = self._find_best_mission_for_customer(
                customer, actual_route, actual_arcs, chrom.drone_missions
            )
            if mission is not None:
                chrom.drone_missions.append(mission)
            else:
                # 无法为该客户分配无人机任务，移回卡车
                chrom.drone_assignments.discard(customer)

        # 计算适应度
        chrom.fitness = self._calculate_fitness(chrom)

    def _find_best_mission_for_customer(
        self, customer: int, route: List[int],
        arcs: List[Tuple[int, int]], existing_missions: List[DroneMission]
    ) -> Optional[DroneMission]:
        """为客户找到最佳无人机任务"""
        best_mission = None
        best_cost = float('inf')

        customer_coords = self.instance.node_coords[customer]
        payload = self.instance.demands.get(customer, 0)

        for launch_idx, launch_arc in enumerate(arcs):
            for recover_idx in range(launch_idx, len(arcs)):
                recover_arc = arcs[recover_idx]

                # 尝试不同的lambda组合
                for lambda_l in [0.0, 0.3, 0.5, 0.7, 1.0]:
                    for lambda_r in [0.0, 0.3, 0.5, 0.7, 1.0]:
                        if launch_idx == recover_idx and lambda_r <= lambda_l:
                            continue

                        # 计算起飞和降落点
                        launch_point = self.instance.get_point_on_arc(
                            launch_arc[0], launch_arc[1], lambda_l)
                        recover_point = self.instance.get_point_on_arc(
                            recover_arc[0], recover_arc[1], lambda_r)

                        # 计算飞行距离
                        dist_out = self.instance.euclidean_distance(launch_point, customer_coords)
                        dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)

                        # 选择最佳速度
                        best_speed = self._select_best_speed(payload, dist_out, dist_ret)
                        if best_speed is None:
                            continue

                        # 创建任务
                        mission = DroneMission(
                            launch_arc_start=launch_arc[0],
                            launch_arc_end=launch_arc[1],
                            lambda_launch=lambda_l,
                            customer=customer,
                            recover_arc_start=recover_arc[0],
                            recover_arc_end=recover_arc[1],
                            lambda_recover=lambda_r,
                            outbound_speed=best_speed,
                            return_speed=best_speed
                        )

                        # 检查是否与现有任务重叠
                        if self._check_mission_overlap(mission, existing_missions, arcs):
                            continue

                        # 计算成本
                        cost = self._calculate_mission_cost(mission, route, launch_idx, recover_idx)

                        if cost < best_cost:
                            best_cost = cost
                            best_mission = mission

        return best_mission

    def _select_best_speed(self, payload: float, dist_out: float, dist_ret: float) -> Optional[float]:
        """选择满足电池约束的最佳速度"""
        for speed in self.sorted_speeds:
            energy_out = self.instance.energy_model.energy_for_distance(payload, speed, dist_out)
            energy_ret = self.instance.energy_model.energy_for_distance(0, speed, dist_ret)
            total_energy = energy_out + energy_ret

            if total_energy <= self.battery_limit:
                return speed

        return None

    def _calculate_mission_cost(
        self, mission: DroneMission, route: List[int],
        launch_idx: int, recover_idx: int
    ) -> float:
        """计算单个任务的成本"""
        cost = 0.0

        customer_coords = self.instance.node_coords[mission.customer]
        launch_point = self.instance.get_point_on_arc(
            mission.launch_arc_start, mission.launch_arc_end, mission.lambda_launch)
        recover_point = self.instance.get_point_on_arc(
            mission.recover_arc_start, mission.recover_arc_end, mission.lambda_recover)

        dist_out = self.instance.euclidean_distance(launch_point, customer_coords)
        dist_ret = self.instance.euclidean_distance(customer_coords, recover_point)

        # 距离成本
        cost += (dist_out + dist_ret) * self.instance.drone_cost

        # 能耗成本
        payload = self.instance.demands.get(mission.customer, 0)
        energy_out = self.instance.energy_model.energy_for_distance(
            payload, mission.outbound_speed, dist_out)
        energy_ret = self.instance.energy_model.energy_for_distance(
            0, mission.return_speed, dist_ret)
        cost += (energy_out + energy_ret) * self.instance.energy_cost

        # 同步等待成本
        service_time = self.instance.service_times.get(mission.customer, 0)
        drone_time = dist_out / mission.outbound_speed + service_time + dist_ret / mission.return_speed
        truck_time = self.instance.calculate_truck_segment_time(
            route, launch_idx, recover_idx, mission.lambda_launch, mission.lambda_recover)
        wait_time = abs(drone_time - truck_time)
        cost += wait_time * self.instance.wait_cost

        return cost

    def _calculate_fitness(self, chrom: Chromosome) -> float:
        """计算染色体的适应度（目标函数值，越小越好）"""
        total_cost = 0.0
        depot = self.instance.depot

        # 构建实际卡车路线
        actual_route = [depot]
        for c in chrom.truck_sequence:
            if c not in chrom.drone_assignments:
                actual_route.append(c)
        actual_route.append(depot)

        actual_arcs = [(actual_route[i], actual_route[i+1]) for i in range(len(actual_route) - 1)]

        # 1. 卡车运输成本
        for i in range(len(actual_route) - 1):
            dist = self.instance.dist.get((actual_route[i], actual_route[i+1]), 0)
            total_cost += dist * self.instance.truck_cost

        # 2. 无人机任务成本
        for mission in chrom.drone_missions:
            launch_arc = (mission.launch_arc_start, mission.launch_arc_end)
            recover_arc = (mission.recover_arc_start, mission.recover_arc_end)

            if launch_arc not in actual_arcs or recover_arc not in actual_arcs:
                total_cost += 1e5
                continue

            launch_idx = actual_arcs.index(launch_arc)
            recover_idx = actual_arcs.index(recover_arc)

            if launch_idx > recover_idx:
                total_cost += 1e5
                continue

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
            drone_time = dist_out / mission.outbound_speed + service_time + dist_ret / mission.return_speed
            truck_time = self.instance.calculate_truck_segment_time(
                actual_route, launch_idx, recover_idx,
                mission.lambda_launch, mission.lambda_recover)
            wait_time = abs(drone_time - truck_time)
            total_cost += wait_time * self.instance.wait_cost

        # 3. 任务重叠和准备时间约束
        mission_positions = []
        for mission in chrom.drone_missions:
            pos = self._get_mission_route_position(mission, actual_arcs)
            if pos is not None:
                mission_positions.append(pos)

        mission_positions.sort(key=lambda p: p[0])
        prep_time = self.instance.drone_prep_time
        for i in range(len(mission_positions) - 1):
            prev_end = mission_positions[i][1]
            next_start = mission_positions[i + 1][0]
            if prev_end > next_start:
                total_cost += 1e5
            else:
                gap_time = self._calculate_position_travel_time(prev_end, next_start, actual_arcs)
                if gap_time < prep_time:
                    total_cost += 1e4

        # 4. 检查所有客户是否被服务
        served = set()
        for c in chrom.truck_sequence:
            if c not in chrom.drone_assignments:
                served.add(c)
        for m in chrom.drone_missions:
            served.add(m.customer)

        unserved = set(self.instance.customers) - served
        if unserved:
            total_cost += len(unserved) * 1e6

        return total_cost

    def _get_mission_route_position(
        self, mission: DroneMission, arcs: List[Tuple[int, int]]
    ) -> Optional[Tuple[float, float]]:
        """计算任务在路线上的位置区间"""
        launch_arc = (mission.launch_arc_start, mission.launch_arc_end)
        recover_arc = (mission.recover_arc_start, mission.recover_arc_end)

        if launch_arc not in arcs or recover_arc not in arcs:
            return None

        launch_idx = arcs.index(launch_arc)
        recover_idx = arcs.index(recover_arc)

        start_pos = launch_idx + mission.lambda_launch
        end_pos = recover_idx + mission.lambda_recover

        return (start_pos, end_pos)

    def _calculate_position_travel_time(
        self, pos1: float, pos2: float, arcs: List[Tuple[int, int]]
    ) -> float:
        """计算两个位置之间的卡车行驶时间"""
        if pos2 <= pos1:
            return 0.0

        arc_idx1 = int(pos1)
        lambda1 = pos1 - arc_idx1
        arc_idx2 = int(pos2)
        lambda2 = pos2 - arc_idx2

        if arc_idx1 >= len(arcs) or arc_idx2 >= len(arcs):
            return 0.0

        total_time = 0.0

        if arc_idx1 == arc_idx2:
            arc = arcs[arc_idx1]
            arc_time = self.instance.truck_time.get(arc, 0)
            total_time = (lambda2 - lambda1) * arc_time
        else:
            arc = arcs[arc_idx1]
            arc_time = self.instance.truck_time.get(arc, 0)
            total_time += (1 - lambda1) * arc_time

            for i in range(arc_idx1 + 1, arc_idx2):
                if i < len(arcs):
                    arc = arcs[i]
                    total_time += self.instance.truck_time.get(arc, 0)

            arc = arcs[arc_idx2]
            arc_time = self.instance.truck_time.get(arc, 0)
            total_time += lambda2 * arc_time

        return total_time

    def _check_mission_overlap(
        self, new_mission: DroneMission,
        existing_missions: List[DroneMission],
        arcs: List[Tuple[int, int]]
    ) -> bool:
        """检查任务是否重叠"""
        new_pos = self._get_mission_route_position(new_mission, arcs)
        if new_pos is None:
            return True

        new_start, new_end = new_pos
        prep_time = self.instance.drone_prep_time

        for existing in existing_missions:
            ex_pos = self._get_mission_route_position(existing, arcs)
            if ex_pos is None:
                continue

            ex_start, ex_end = ex_pos

            if new_start < ex_end and ex_start < new_end:
                return True

            if new_start >= ex_end:
                gap_time = self._calculate_position_travel_time(ex_end, new_start, arcs)
                if gap_time < prep_time:
                    return True

            if ex_start >= new_end:
                gap_time = self._calculate_position_travel_time(new_end, ex_start, arcs)
                if gap_time < prep_time:
                    return True

        return False

    # ==================== 选择操作 ====================

    def _tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """锦标赛选择"""
        candidates = self.rng.sample(population, min(self.tournament_size, len(population)))
        return min(candidates, key=lambda c: c.fitness)

    def _roulette_selection(self, population: List[Chromosome]) -> Chromosome:
        """轮盘赌选择（适应度越小越好，需要转换）"""
        # 计算适应度的倒数作为选择概率
        max_fitness = max(c.fitness for c in population)
        adjusted_fitness = [max_fitness - c.fitness + 1 for c in population]
        total = sum(adjusted_fitness)

        r = self.rng.random() * total
        cumulative = 0
        for i, f in enumerate(adjusted_fitness):
            cumulative += f
            if r <= cumulative:
                return population[i]

        return population[-1]

    def _select(self, population: List[Chromosome]) -> Chromosome:
        """选择操作"""
        if self.selection_method == "tournament":
            return self._tournament_selection(population)
        else:
            return self._roulette_selection(population)

    # ==================== 交叉操作 ====================

    def _order_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """顺序交叉 (OX)"""
        seq1 = parent1.truck_sequence.copy()
        seq2 = parent2.truck_sequence.copy()
        size = len(seq1)

        if size < 3:
            return parent1.copy(), parent2.copy()

        # 随机选择交叉段
        start = self.rng.randint(0, size - 2)
        end = self.rng.randint(start + 1, size)

        # 创建子代
        child1_seq = [None] * size
        child2_seq = [None] * size

        # 复制交叉段
        child1_seq[start:end] = seq1[start:end]
        child2_seq[start:end] = seq2[start:end]

        # 填充剩余位置
        def fill_remaining(child_seq, other_seq, start, end):
            remaining = [x for x in other_seq if x not in child_seq[start:end]]
            pos = end % size
            for gene in remaining:
                while child_seq[pos] is not None:
                    pos = (pos + 1) % size
                child_seq[pos] = gene
                pos = (pos + 1) % size

        fill_remaining(child1_seq, seq2, start, end)
        fill_remaining(child2_seq, seq1, start, end)

        # 创建子代染色体
        child1 = Chromosome()
        child1.truck_sequence = child1_seq
        # 继承无人机分配（交集）
        child1.drone_assignments = parent1.drone_assignments & parent2.drone_assignments

        child2 = Chromosome()
        child2.truck_sequence = child2_seq
        child2.drone_assignments = parent1.drone_assignments & parent2.drone_assignments

        return child1, child2

    def _pmx_crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """部分映射交叉 (PMX)"""
        seq1 = parent1.truck_sequence.copy()
        seq2 = parent2.truck_sequence.copy()
        size = len(seq1)

        if size < 3:
            return parent1.copy(), parent2.copy()

        start = self.rng.randint(0, size - 2)
        end = self.rng.randint(start + 1, size)

        child1_seq = [None] * size
        child2_seq = [None] * size

        # 复制交叉段
        child1_seq[start:end] = seq1[start:end]
        child2_seq[start:end] = seq2[start:end]

        # 建立映射
        mapping1 = {}  # child1: seq2 -> seq1
        mapping2 = {}  # child2: seq1 -> seq2
        for i in range(start, end):
            mapping1[seq2[i]] = seq1[i]
            mapping2[seq1[i]] = seq2[i]

        # 填充剩余位置
        def fill_with_mapping(child_seq, other_seq, mapping, start, end):
            for i in range(size):
                if i < start or i >= end:
                    gene = other_seq[i]
                    while gene in child_seq[start:end]:
                        gene = mapping.get(gene, gene)
                    child_seq[i] = gene

        fill_with_mapping(child1_seq, seq2, mapping1, start, end)
        fill_with_mapping(child2_seq, seq1, mapping2, start, end)

        child1 = Chromosome()
        child1.truck_sequence = child1_seq
        child1.drone_assignments = parent1.drone_assignments & parent2.drone_assignments

        child2 = Chromosome()
        child2.truck_sequence = child2_seq
        child2.drone_assignments = parent1.drone_assignments & parent2.drone_assignments

        return child1, child2

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """交叉操作"""
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        if self.crossover_method == "ox":
            return self._order_crossover(parent1, parent2)
        else:
            return self._pmx_crossover(parent1, parent2)

    # ==================== 变异操作 ====================

    def _swap_mutation(self, chrom: Chromosome):
        """交换变异"""
        seq = chrom.truck_sequence
        if len(seq) < 2:
            return

        i, j = self.rng.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]

    def _insert_mutation(self, chrom: Chromosome):
        """插入变异"""
        seq = chrom.truck_sequence
        if len(seq) < 2:
            return

        i = self.rng.randint(0, len(seq) - 1)
        j = self.rng.randint(0, len(seq) - 1)
        if i != j:
            gene = seq.pop(i)
            seq.insert(j, gene)

    def _reverse_mutation(self, chrom: Chromosome):
        """逆转变异 (2-opt)"""
        seq = chrom.truck_sequence
        if len(seq) < 2:
            return

        i = self.rng.randint(0, len(seq) - 2)
        j = self.rng.randint(i + 1, len(seq))
        seq[i:j] = seq[i:j][::-1]

    def _drone_assignment_mutation(self, chrom: Chromosome):
        """无人机分配变异"""
        eligible = [c for c in chrom.truck_sequence if self.instance.is_drone_eligible(c)]

        if not eligible:
            return

        # 随机选择一个客户
        customer = self.rng.choice(eligible)

        # 切换其无人机分配状态
        if customer in chrom.drone_assignments:
            chrom.drone_assignments.discard(customer)
        else:
            chrom.drone_assignments.add(customer)

    def _mutate(self, chrom: Chromosome):
        """变异操作"""
        if self.rng.random() < self.mutation_rate:
            mutation_type = self.rng.choice(['swap', 'insert', 'reverse', 'drone'])

            if mutation_type == 'swap':
                self._swap_mutation(chrom)
            elif mutation_type == 'insert':
                self._insert_mutation(chrom)
            elif mutation_type == 'reverse':
                self._reverse_mutation(chrom)
            else:
                self._drone_assignment_mutation(chrom)

    # ==================== 局部搜索 ====================

    def _local_search(self, chrom: Chromosome):
        """局部搜索改进"""
        improved = True
        max_iterations = 50
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # 2-opt改进
            seq = chrom.truck_sequence
            best_improvement = 0
            best_i, best_j = -1, -1

            depot = self.instance.depot
            route = [depot] + seq + [depot]

            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    a, b = route[i-1], route[i]
                    c, d = route[j], route[j+1]

                    current_cost = (self.instance.dist.get((a, b), 0) +
                                   self.instance.dist.get((c, d), 0))
                    new_cost = (self.instance.dist.get((a, c), 0) +
                               self.instance.dist.get((b, d), 0))

                    improvement = current_cost - new_cost
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_i, best_j = i - 1, j - 1  # 调整为seq的索引

            if best_improvement > 0.001:
                seq[best_i:best_j+1] = seq[best_i:best_j+1][::-1]
                improved = True

    # ==================== 主求解循环 ====================

    def solve(self, verbose: bool = True) -> Solution:
        """执行遗传算法求解"""
        start_time = time.time()

        if verbose:
            print(f"\n初始化种群 (大小: {self.population_size})...")

        # 初始化种群
        population = self._initialize_population()

        # 添加贪婪解
        greedy_chrom = self._create_greedy_chromosome()
        self._decode_and_evaluate(greedy_chrom)
        population[0] = greedy_chrom

        # 排序
        population.sort(key=lambda c: c.fitness)

        best_chromosome = population[0].copy()
        self.stats['best_fitness'] = best_chromosome.fitness

        if verbose:
            print(f"初始最优适应度: {best_chromosome.fitness:.2f}")
            print(f"\n开始进化...")

        # 进化循环
        no_improve_count = 0

        for gen in range(self.max_generations):
            new_population = []

            # 精英保留
            for i in range(self.elite_size):
                new_population.append(population[i].copy())

            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self._select(population)
                parent2 = self._select(population)

                # 交叉
                child1, child2 = self._crossover(parent1, parent2)

                # 变异
                self._mutate(child1)
                self._mutate(child2)

                # 解码和评估
                self._decode_and_evaluate(child1)
                self._decode_and_evaluate(child2)

                # 局部搜索（概率性）
                if self.rng.random() < 0.1:
                    self._local_search(child1)
                    self._decode_and_evaluate(child1)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            # 更新种群
            population = new_population
            population.sort(key=lambda c: c.fitness)

            # 更新最优解
            if population[0].fitness < best_chromosome.fitness:
                best_chromosome = population[0].copy()
                self.stats['best_fitness'] = best_chromosome.fitness
                self.stats['best_generation'] = gen + 1
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 记录历史
            avg_fitness = sum(c.fitness for c in population) / len(population)
            self.stats['avg_fitness_history'].append(avg_fitness)
            self.stats['best_fitness_history'].append(best_chromosome.fitness)

            # 输出进度
            if verbose and (gen + 1) % 50 == 0:
                print(f"  代数 {gen + 1:4d}: 最优={best_chromosome.fitness:.2f}, "
                      f"平均={avg_fitness:.2f}, 无人机任务={len(best_chromosome.drone_missions)}")

            # 早停
            if no_improve_count >= 100:
                if verbose:
                    print(f"  连续{no_improve_count}代无改进，提前终止")
                break

        self.stats['generations'] = gen + 1
        self.stats['solve_time'] = time.time() - start_time

        if verbose:
            print(f"\n求解完成!")
            print(f"  总代数: {self.stats['generations']}")
            print(f"  最优解在第 {self.stats['best_generation']} 代找到")
            print(f"  求解时间: {self.stats['solve_time']:.2f}s")

        return best_chromosome.to_solution(self.instance.depot)

    # ==================== 结果输出 ====================

    def print_solution(self, solution: Solution):
        """打印解的详细信息"""
        print("\n" + "=" * 60)
        print("GA求解结果 (途中起飞模型)")
        print("=" * 60)

        print(f"\n目标函数值: {solution.objective:.4f}")

        depot = self.instance.depot
        drone_served = solution.get_drone_served_customers()

        # 构建实际卡车路线
        actual_route = [depot]
        for c in solution.truck_route:
            if c != depot and c not in drone_served:
                actual_route.append(c)
        actual_route.append(depot)

        # 计算卡车距离
        truck_dist = sum(self.instance.dist.get((actual_route[i], actual_route[i+1]), 0)
                        for i in range(len(actual_route) - 1))
        print(f"卡车总行驶距离: {truck_dist:.2f}")

        # 计算无人机距离
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

        print(f"无人机总飞行距离: {drone_dist:.2f}")
        print(f"总能耗: {total_energy:.2f} Wh")

        # 卡车路线
        print(f"\n卡车实际路线 ({len(actual_route)} 节点):")
        print(f"  {' -> '.join(map(str, actual_route))}")
        print(f"  卡车服务的客户: {sorted([c for c in actual_route if c != depot])}")

        # 无人机任务
        print(f"\n无人机任务 ({len(solution.drone_missions)} 个):")
        actual_arcs = [(actual_route[i], actual_route[i+1]) for i in range(len(actual_route) - 1)]

        for i, mission in enumerate(solution.drone_missions, 1):
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

            print(f"\n  任务 {i}: 服务客户 {mission.customer} (需求: {payload})")

            launch_type = "途中起飞" if 0 < mission.lambda_launch < 1 else ""
            print(f"    起飞: 弧({mission.launch_arc_start}->{mission.launch_arc_end})上 "
                  f"λ={mission.lambda_launch:.2f}  [{launch_type}]")

            recover_type = "途中降落" if 0 < mission.lambda_recover < 1 else ""
            print(f"    降落: 弧({mission.recover_arc_start}->{mission.recover_arc_end})上 "
                  f"λ={mission.lambda_recover:.2f}  [{recover_type}]")

            print(f"    去程: {dist_out:.1f}m @ {mission.outbound_speed:.0f}m/s")
            print(f"    返程: {dist_ret:.1f}m @ {mission.return_speed:.0f}m/s")
            print(f"    能耗: {energy_out + energy_ret:.2f} Wh")

        # 任务重叠检查
        if solution.drone_missions:
            mission_positions = []
            for m in solution.drone_missions:
                pos = self._get_mission_route_position(m, actual_arcs)
                if pos is not None:
                    mission_positions.append((m.customer, pos[0], pos[1]))
            mission_positions.sort(key=lambda x: x[1])

            overlap_count = 0
            prep_violation_count = 0
            prep_time = self.instance.drone_prep_time

            for i in range(len(mission_positions) - 1):
                prev_end = mission_positions[i][2]
                next_start = mission_positions[i + 1][1]
                if prev_end > next_start:
                    overlap_count += 1
                else:
                    gap_time = self._calculate_position_travel_time(prev_end, next_start, actual_arcs)
                    if gap_time < prep_time:
                        prep_violation_count += 1

            print(f"\n  --- 无人机任务汇总 ---")
            if overlap_count > 0:
                print(f"  !! 警告: 检测到 {overlap_count} 对任务时间窗口重叠 !!")
            elif prep_violation_count > 0:
                print(f"  !! 警告: {prep_violation_count} 对任务准备时间不足 (需{prep_time:.0f}s) !!")
            else:
                print(f"  无人机任务时间窗口无重叠且满足准备时间({prep_time:.0f}s)约束 (可行)")


# ==================== 可视化 ====================

def visualize_solution(instance: EnRTSPDVSInstance, solution: Solution,
                       save_path: str = None):
    """可视化解"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib未安装，跳过可视化")
        return None

    fig, ax = plt.subplots(figsize=(14, 12))

    depot = instance.depot
    drone_served = solution.get_drone_served_customers()

    # 绘制仓库
    dx, dy = instance.node_coords[depot]
    ax.scatter([dx], [dy], c='red', s=300, marker='s', zorder=5, label='仓库')
    ax.annotate('D', (dx, dy), fontsize=12, ha='center', va='center', fontweight='bold')

    # 绘制客户
    truck_customers = []
    drone_customers = []
    for c in instance.customers:
        if c in drone_served:
            drone_customers.append(c)
        else:
            truck_customers.append(c)

    if truck_customers:
        truck_x = [instance.node_coords[c][0] for c in truck_customers]
        truck_y = [instance.node_coords[c][1] for c in truck_customers]
        ax.scatter(truck_x, truck_y, c='blue', s=150, marker='o', zorder=4, label='卡车客户')
        for c in truck_customers:
            cx, cy = instance.node_coords[c]
            ax.annotate(str(c), (cx, cy), fontsize=9, ha='center', va='center')

    if drone_customers:
        drone_x = [instance.node_coords[c][0] for c in drone_customers]
        drone_y = [instance.node_coords[c][1] for c in drone_customers]
        ax.scatter(drone_x, drone_y, c='green', s=150, marker='^', zorder=4, label='无人机客户')
        for c in drone_customers:
            cx, cy = instance.node_coords[c]
            ax.annotate(str(c), (cx, cy), fontsize=9, ha='center', va='center')

    # 绘制卡车路线
    actual_route = [depot]
    for c in solution.truck_route:
        if c != depot and c not in drone_served:
            actual_route.append(c)
    actual_route.append(depot)

    for i in range(len(actual_route) - 1):
        x1, y1 = instance.node_coords[actual_route[i]]
        x2, y2 = instance.node_coords[actual_route[i+1]]
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    # 绘制无人机任务
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(solution.drone_missions), 1)))
    for idx, mission in enumerate(solution.drone_missions):
        color = colors[idx % len(colors)]

        launch_point = instance.get_point_on_arc(
            mission.launch_arc_start, mission.launch_arc_end, mission.lambda_launch)
        recover_point = instance.get_point_on_arc(
            mission.recover_arc_start, mission.recover_arc_end, mission.lambda_recover)
        customer_coords = instance.node_coords[mission.customer]

        # 起飞点
        ax.scatter([launch_point[0]], [launch_point[1]], c=[color], s=80, marker='>', zorder=6)
        # 降落点
        ax.scatter([recover_point[0]], [recover_point[1]], c=[color], s=80, marker='<', zorder=6)

        # 飞行路径
        ax.plot([launch_point[0], customer_coords[0]], [launch_point[1], customer_coords[1]],
               '--', color=color, lw=1.5, alpha=0.7)
        ax.plot([customer_coords[0], recover_point[0]], [customer_coords[1], recover_point[1]],
               '--', color=color, lw=1.5, alpha=0.7)

    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_title(f'GA求解结果 - 目标值: {solution.objective:.2f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        import os
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "ga_enrtsp_dvs_solution.png")
        _, ext = os.path.splitext(save_path)
        if not ext:
            save_path = save_path + ".png"
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存: {save_path}")

    plt.close(fig)
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


def load_solomon_instance(
    filepath: str,
    num_customers: Optional[int] = None,
    drone_speeds: List[float] = None,
    truck_speed: float = 10.0,
    battery_capacity: float = 800.0,
    drone_max_load: float = 50.0,
    truck_cost: float = 2.0,
    drone_cost: float = 0.2,
    energy_cost: float = 0.01,
    wait_cost: float = 0.5,
    coord_scale: float = 1.0,
    drone_prep_time: float = 10.0,
) -> EnRTSPDVSInstance:
    """解析Solomon标准算例文件"""
    import os

    node_coords = {}
    customers = []
    demands = {}
    time_windows = {}
    service_times = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    instance_name = lines[0].strip()

    vehicle_capacity = 200
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("NUMBER") and "CAPACITY" in stripped:
            parts = lines[i + 1].split()
            if len(parts) >= 2:
                vehicle_capacity = int(parts[1])
            break

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 7:
            try:
                cust_no = int(parts[0])
                x = float(parts[1]) * coord_scale
                y = float(parts[2]) * coord_scale
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])

                node_coords[cust_no] = (x, y)
                if cust_no == 0:
                    time_windows[cust_no] = (ready_time, due_date)
                    service_times[cust_no] = service_time
                    demands[cust_no] = demand
                else:
                    customers.append(cust_no)
                    demands[cust_no] = demand
                    time_windows[cust_no] = (ready_time, due_date)
                    service_times[cust_no] = service_time
            except ValueError:
                continue

    if num_customers is not None and num_customers < len(customers):
        customers = customers[:num_customers]
        all_used = set(customers) | {0}
        node_coords = {k: v for k, v in node_coords.items() if k in all_used}
        demands = {k: v for k, v in demands.items() if k in all_used}
        time_windows = {k: v for k, v in time_windows.items() if k in all_used}
        service_times = {k: v for k, v in service_times.items() if k in all_used}

    print(f"    Solomon算例: {instance_name}")
    print(f"    客户数量: {len(customers)}")
    print(f"    坐标缩放: {coord_scale}x")

    return EnRTSPDVSInstance(
        node_coords=node_coords,
        customers=customers,
        depot=0,
        demands=demands,
        time_windows=time_windows,
        service_times=service_times,
        drone_speeds=drone_speeds or [12.0, 15.0, 18.0, 22.0, 25.0, 30.0],
        truck_speed=truck_speed,
        battery_capacity=battery_capacity,
        drone_max_load=drone_max_load,
        truck_cost=truck_cost,
        drone_cost=drone_cost,
        energy_cost=energy_cost,
        wait_cost=wait_cost,
        drone_prep_time=drone_prep_time,
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

    # 2-opt改进
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

    total_cost = 0
    for i in range(len(solution.truck_route) - 1):
        dist = instance.dist.get((solution.truck_route[i], solution.truck_route[i+1]), 0)
        total_cost += dist * instance.truck_cost
    solution.objective = total_cost

    return solution


# ==================== 主函数 ====================

def main():
    """主函数 - 随机实例实验"""
    print("=" * 70)
    print("遗传算法(GA)求解 enRTSP-DVS 问题")
    print("(弧中同步的速度可变无人机与卡车配送优化问题 - 途中起飞模型)")
    print("=" * 70)

    NUM_CUSTOMERS = 15
    SEED = 42
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 300

    print("\n[1] 创建问题实例...")
    instance = create_drone_friendly_instance(num_customers=NUM_CUSTOMERS, seed=SEED)

    print(f"    客户数量: {len(instance.customers)}")
    print(f"    仓库位置: {instance.node_coords[instance.depot]}")
    print(f"    卡车成本: {instance.truck_cost}")
    print(f"    无人机成本: {instance.drone_cost}")
    print(f"    无人机准备时间: {instance.drone_prep_time}s")

    print("\n[2] 初始化GA求解器...")
    solver = GASolver(
        instance=instance,
        seed=SEED,
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
        elite_size=5,
        crossover_rate=0.85,
        mutation_rate=0.15,
        tournament_size=5,
        selection_method="tournament",
        crossover_method="ox",
    )

    print("\n[3] 开始求解...")
    solution = solver.solve(verbose=True)

    solver.print_solution(solution)

    print("\n" + "=" * 60)
    print("[4] 对比实验...")
    truck_only = get_truck_only_solution(instance)
    print(f"\n  纯卡车配送成本: {truck_only.objective:.2f}")
    print(f"  GA求解成本: {solution.objective:.2f}")
    if truck_only.objective > 0:
        improvement = (truck_only.objective - solution.objective) / truck_only.objective * 100
        print(f"  改进百分比: {improvement:.2f}%")

    print("\n[5] 生成可视化...")
    try:
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, f"ga_enrtsp_dvs_{NUM_CUSTOMERS}customers.png")
        visualize_solution(instance, solution, save_path=img_path)
    except Exception as e:
        print(f"可视化失败: {e}")

    return instance, solution, solver


def main_solomon():
    """Solomon算例实验"""
    import os

    SOLOMON_FILE = "c101.txt"
    NUM_CUSTOMERS = 25
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 300
    SEED = 42
    COORD_SCALE = 10.0

    if not os.path.exists(SOLOMON_FILE):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, SOLOMON_FILE)
        if os.path.exists(alt_path):
            SOLOMON_FILE = alt_path
        else:
            print(f"错误: 找不到Solomon算例文件 '{SOLOMON_FILE}'")
            return None

    print("=" * 70)
    print("GA求解Solomon算例 - enRTSP-DVS")
    print("=" * 70)

    print(f"\n[1] 加载Solomon算例: {os.path.basename(SOLOMON_FILE)}")
    instance = load_solomon_instance(
        SOLOMON_FILE,
        num_customers=NUM_CUSTOMERS,
        coord_scale=COORD_SCALE,
    )

    print(f"\n[2] 初始化GA求解器...")
    solver = GASolver(
        instance=instance,
        seed=SEED,
        population_size=POPULATION_SIZE,
        max_generations=MAX_GENERATIONS,
    )

    print(f"\n[3] 开始求解...")
    solution = solver.solve(verbose=True)

    solver.print_solution(solution)

    print("\n" + "=" * 60)
    print("[4] 对比实验...")
    truck_only = get_truck_only_solution(instance)
    print(f"\n  纯卡车配送成本: {truck_only.objective:.2f}")
    print(f"  GA求解成本: {solution.objective:.2f}")
    if truck_only.objective > 0:
        improvement = (truck_only.objective - solution.objective) / truck_only.objective * 100
        print(f"  改进百分比: {improvement:.2f}%")

    print("\n[5] 生成可视化...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_name = os.path.splitext(os.path.basename(SOLOMON_FILE))[0]
        img_path = os.path.join(script_dir, f"ga_solomon_{base_name}_{NUM_CUSTOMERS}.png")
        visualize_solution(instance, solution, save_path=img_path)
    except Exception as e:
        print(f"可视化失败: {e}")

    return instance, solution, solver


if __name__ == "__main__":
    # 选择运行模式:
    # main()          - 随机实例
    # main_solomon()  - Solomon算例

    instance, solution, solver = main()
    # result = main_solomon()
