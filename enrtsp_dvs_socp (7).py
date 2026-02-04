"""
弧中同步的速度可变无人机与卡车配送优化问题 (enRTSP-DVS)
En-Route Synchronization Truck-Drone Problem with Drone Variable Speed

改进版本：使用二次锥规划（SOCP）和分段线性化技术

主要改进：
1. 使用SOCP约束处理欧几里得距离计算
2. 使用分段线性化处理速度-时间关系
3. 更精确的能耗建模
4. 支持读取Solomon算例
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
import random
import os
from energy_model_improvements import (
    RealisticDronePowerModel,
    create_improved_energy_params,
    get_optimal_speed_for_distance
)


class SolomonInstanceReader:
    """
    Solomon算例读取器
    
    支持读取标准Solomon VRPTW算例格式
    """
    
    def __init__(self, filepath: str):
        """
        初始化读取器
        
        Args:
            filepath: Solomon算例文件路径
        """
        self.filepath = filepath
        self.name = ""
        self.vehicle_number = 0
        self.vehicle_capacity = 0
        self.customers = []  # 包含仓库(index=0)
        self._parse_file()
    
    def _parse_file(self):
        """解析Solomon算例文件"""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()
        
        # 第一行是算例名称
        self.name = lines[0].strip()
        
        # 找到VEHICLE部分
        vehicle_idx = -1
        customer_idx = -1
        for i, line in enumerate(lines):
            if 'VEHICLE' in line:
                vehicle_idx = i
            if 'CUST NO' in line or 'CUST NO.' in line:
                customer_idx = i
                break
        
        # 解析车辆信息
        if vehicle_idx >= 0:
            # 跳过标题行，读取数据行
            for i in range(vehicle_idx + 1, len(lines)):
                parts = lines[i].split()
                if len(parts) >= 2:
                    try:
                        self.vehicle_number = int(parts[0])
                        self.vehicle_capacity = int(parts[1])
                        break
                    except ValueError:
                        continue
        
        # 解析客户信息
        if customer_idx >= 0:
            for i in range(customer_idx + 1, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    try:
                        customer = {
                            'id': int(parts[0]),
                            'x': float(parts[1]),
                            'y': float(parts[2]),
                            'demand': float(parts[3]),
                            'ready_time': float(parts[4]),
                            'due_date': float(parts[5]),
                            'service_time': float(parts[6])
                        }
                        self.customers.append(customer)
                    except ValueError:
                        continue
        
        print(f"读取Solomon算例: {self.name}")
        print(f"  - 车辆数量: {self.vehicle_number}, 容量: {self.vehicle_capacity}")
        print(f"  - 总节点数: {len(self.customers)} (含仓库)")
    
    def get_instance(
        self, 
        num_customers: int = None, 
        random_select: bool = True,
        seed: int = None
    ) -> Dict:
        """
        获取实例数据
        
        Args:
            num_customers: 需要的客户数量 (不含仓库)，None表示全部
            random_select: 是否随机选择客户
            seed: 随机种子
        
        Returns:
            包含实例数据的字典
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 仓库始终是第一个节点 (id=0)
        depot_data = None
        customer_data = []
        
        for c in self.customers:
            if c['id'] == 0:
                depot_data = c
            else:
                customer_data.append(c)
        
        if depot_data is None:
            raise ValueError("未找到仓库节点 (id=0)")
        
        # 确定客户数量
        max_customers = len(customer_data)
        if num_customers is None:
            num_customers = max_customers
        else:
            num_customers = min(num_customers, max_customers)
        
        # 选择客户
        if random_select and num_customers < max_customers:
            selected_customers = random.sample(customer_data, num_customers)
        else:
            selected_customers = customer_data[:num_customers]
        
        # 重新编号：仓库=0, 客户=1,2,3,...
        node_coords = {0: (depot_data['x'], depot_data['y'])}
        customers = []
        demands = {}
        time_windows = {}
        service_times = {}
        
        for i, c in enumerate(selected_customers, start=1):
            node_coords[i] = (c['x'], c['y'])
            customers.append(i)
            demands[i] = c['demand']
            time_windows[i] = (c['ready_time'], c['due_date'])
            service_times[i] = c['service_time']
        
        print(f"\n生成实例:")
        print(f"  - 仓库位置: {node_coords[0]}")
        print(f"  - 选择客户数: {len(customers)}")
        if random_select:
            print(f"  - 选择方式: 随机")
        else:
            print(f"  - 选择方式: 顺序")
        
        return {
            'node_coords': node_coords,
            'customers': customers,
            'demands': demands,
            'time_windows': time_windows,
            'service_times': service_times,
            'depot': 0,
            'vehicle_capacity': self.vehicle_capacity,
            'instance_name': self.name,
        }
    
    @staticmethod
    def from_string(content: str, name: str = "custom") -> 'SolomonInstanceReader':
        """
        从字符串内容创建读取器
        
        Args:
            content: Solomon格式的字符串内容
            name: 实例名称
        
        Returns:
            SolomonInstanceReader实例
        """
        # 创建临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        reader = SolomonInstanceReader(temp_path)
        reader.name = name
        
        # 删除临时文件
        os.unlink(temp_path)
        
        return reader


# 预定义的Solomon算例数据 (C101)
SOLOMON_C101_DATA = """C101

VEHICLE
NUMBER     CAPACITY
  25         200

CUSTOMER
CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE   TIME
 
    0      40         50          0          0       1236          0   
    1      45         68         10        912        967         90   
    2      45         70         30        825        870         90   
    3      42         66         10         65        146         90   
    4      42         68         10        727        782         90   
    5      42         65         10         15         67         90   
    6      40         69         20        621        702         90   
    7      40         66         20        170        225         90   
    8      38         68         20        255        324         90   
    9      38         70         10        534        605         90   
   10      35         66         10        357        410         90   
   11      35         69         10        448        505         90   
   12      25         85         20        652        721         90   
   13      22         75         30         30         92         90   
   14      22         85         10        567        620         90   
   15      20         80         40        384        429         90   
   16      20         85         40        475        528         90   
   17      18         75         20         99        148         90   
   18      15         75         20        179        254         90   
   19      15         80         10        278        345         90   
   20      30         50         10         10         73         90   
   21      30         52         20        914        965         90   
   22      28         52         20        812        883         90   
   23      28         55         10        732        777         90   
   24      25         50         10         65        144         90   
   25      25         52         40        169        224         90   
   26      25         55         10        622        701         90   
   27      23         52         10        261        316         90   
   28      23         55         20        546        593         90   
   29      20         50         10        358        405         90   
   30      20         55         10        449        504         90   
   31      10         35         20        200        237         90   
   32      10         40         30         31        100         90   
   33       8         40         40         87        158         90   
   34       8         45         20        751        816         90   
   35       5         35         10        283        344         90   
   36       5         45         10        665        716         90   
   37       2         40         20        383        434         90   
   38       0         40         30        479        522         90   
   39       0         45         20        567        624         90   
   40      35         30         10        264        321         90   
   41      35         32         10        166        235         90   
   42      33         32         20         68        149         90   
   43      33         35         10         16         80         90   
   44      32         30         10        359        412         90   
   45      30         30         10        541        600         90   
   46      30         32         30        448        509         90   
   47      30         35         10       1054       1127         90   
   48      28         30         10        632        693         90   
   49      28         35         10       1001       1066         90   
   50      26         32         10        815        880         90   
   51      25         30         10        725        786         90   
   52      25         35         10        912        969         90   
   53      44          5         20        286        347         90   
   54      42         10         40        186        257         90   
   55      42         15         10         95        158         90   
   56      40          5         30        385        436         90   
   57      40         15         40         35         87         90   
   58      38          5         30        471        534         90   
   59      38         15         10        651        740         90   
   60      35          5         20        562        629         90   
   61      50         30         10        531        610         90   
   62      50         35         20        262        317         90   
   63      50         40         50        171        218         90   
   64      48         30         10        632        693         90   
   65      48         40         10         76        129         90   
   66      47         35         10        826        875         90   
   67      47         40         10         12         77         90   
   68      45         30         10        734        777         90   
   69      45         35         10        916        969         90   
   70      95         30         30        387        456         90   
   71      95         35         20        293        360         90   
   72      53         30         10        450        505         90   
   73      92         30         10        478        551         90   
   74      53         35         50        353        412         90   
   75      45         65         20        997       1068         90   
   76      90         35         10        203        260         90   
   77      88         30         10        574        643         90   
   78      88         35         20        109        170         90   
   79      87         30         10        668        731         90   
   80      85         25         10        769        820         90   
   81      85         35         30         47        124         90   
   82      75         55         20        369        420         90   
   83      72         55         10        265        338         90   
   84      70         58         20        458        523         90   
   85      68         60         30        555        612         90   
   86      66         55         10        173        238         90   
   87      65         55         20         85        144         90   
   88      65         60         30        645        708         90   
   89      63         58         10        737        802         90   
   90      60         55         10         20         84         90   
   91      60         60         10        836        889         90   
   92      67         85         20        368        441         90   
   93      65         85         40        475        518         90   
   94      65         82         10        285        336         90   
   95      62         80         30        196        239         90   
   96      60         80         10         95        156         90   
   97      60         85         30        561        622         90   
   98      58         75         20         30         84         90   
   99      55         80         10        743        820         90   
  100      55         85         20        647        726         90   
"""


def load_solomon_instance(
    filepath: str = None,
    num_customers: int = 10,
    random_select: bool = True,
    seed: int = None,
    use_builtin: bool = True
) -> Dict:
    """
    加载Solomon算例
    
    Args:
        filepath: 算例文件路径，None则使用内置C101算例
        num_customers: 需要的客户数量
        random_select: 是否随机选择客户
        seed: 随机种子
        use_builtin: 是否使用内置算例（当filepath为None时）
    
    Returns:
        实例数据字典
    
    Example:
        # 使用内置C101算例，随机选择15个客户
        instance = load_solomon_instance(num_customers=15, seed=42)
        
        # 从文件加载，选择全部客户
        instance = load_solomon_instance("path/to/c101.txt", num_customers=None)
    """
    if filepath is None and use_builtin:
        print("使用内置Solomon C101算例")
        reader = SolomonInstanceReader.from_string(SOLOMON_C101_DATA, "C101")
    else:
        if filepath is None:
            raise ValueError("请提供算例文件路径或设置use_builtin=True")
        reader = SolomonInstanceReader(filepath)
    
    return reader.get_instance(
        num_customers=num_customers,
        random_select=random_select,
        seed=seed
    )


class EnRTSPDVSModelSOCP:
    """
    弧中同步的速度可变无人机与卡车配送优化模型 (SOCP版本)
    
    使用二次锥规划处理非线性距离约束
    使用分段线性化处理速度-时间-能耗关系
    """
    
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
        battery_capacity: float = 1000.0,
        drone_max_load: float = 5.0,
        drone_empty_weight: float = 2.0,
        truck_cost: float = 1.0,
        drone_cost: float = 0.5,
        sync_cost: float = 0.2,
        wait_cost: float = 0.1,
        safety_margin: float = 0.1,
        big_M: float = 1e6,
        max_flight_range: float = 10000.0,
        # 分段线性化参数
        num_pwl_segments: int = 10,
    ):
        """初始化模型参数"""
        self.node_coords = node_coords
        self.customers = customers
        self.depot = depot
        self.demands = demands or {c: 1.0 for c in customers}
        self.time_windows = time_windows or {c: (0, 1e6) for c in customers}
        self.service_times = service_times or {c: 0 for c in customers}

        self.drone_speeds = drone_speeds or [5.0, 10.0, 15.0, 20.0]
        self.truck_speed = truck_speed

        self.battery_capacity = battery_capacity
        self.drone_max_load = drone_max_load
        self.drone_empty_weight = drone_empty_weight
        self.max_flight_range = max_flight_range

        self.truck_cost = truck_cost
        self.drone_cost = drone_cost
        self.sync_cost = sync_cost
        self.wait_cost = wait_cost

        self.safety_margin = safety_margin
        self.big_M = big_M
        self.num_pwl_segments = num_pwl_segments

        # 能耗模型参数 (基于多旋翼无人机能耗模型)
        # P(v) = P_blade + P_induced + P_parasite
        # 简化为: P(v) = c0 + c1/v + c2*v^3
        self.energy_params = self._compute_energy_params()

        self._build_sets()
        self._compute_distances()
        self._build_pwl_breakpoints()

        self.model = gp.Model("enRTSP_DVS_SOCP")
        self.vars = {}

    def _preprocess_speed_selection(self):
        """
        预处理：禁用无法覆盖距离的速度选择

        这可以显著减少求解时间
        """
        y = self.vars['y']
        v_out = self.vars['v_out']
        v_ret = self.vars['v_ret']

        disabled_count = 0

        for (i, j, k) in self.drone_trips:
            xk, yk = self.coords[k]
            Q_k = self.demands.get(k, 1.0)

            for v in self.V:
                params = self.energy_params[v]

                # 估算最小飞行距离（起飞点到客户）
                xi, yi = self.coords[i]
                min_dist_out = math.sqrt((xk - xi) ** 2 + (yk - yi) ** 2)

                # 如果最大航程小于估算距离，禁用该速度
                max_range = params['max_range_loaded'] if Q_k > 0 else params['max_range_empty']
                if max_range * (1 - self.safety_margin) < min_dist_out:
                    self.model.addConstr(v_out[i, j, k, v] == 0,
                                         name=f"disable_v_out_{i}_{j}_{k}_{v}")
                    disabled_count += 1

        print(f"  预处理禁用了 {disabled_count} 个不可行的速度选择")
    def _compute_energy_params(self) -> Dict:
        """计算能耗模型参数"""
        # 基于论文中的能耗模型: E = (α_v + β_v*Q + γ_v*v²) * distance
        from energy_model_improvements import RealisticDronePowerModel

        power_model = RealisticDronePowerModel(
            drone_empty_weight=self.drone_empty_weight,
            battery_capacity=self.battery_capacity
        )

        params = {}
        for v in self.drone_speeds:
            # 空载参数
            empty_energy_rate = power_model.energy_per_meter(0, v) / 1000  # kJ/m

            # 参考载重（用于计算β）
            ref_payload = 5.0
            loaded_energy_rate = power_model.energy_per_meter(ref_payload, v) / 1000

            params[v] = {
                # 基础能耗率 (kJ/m)
                'alpha': empty_energy_rate,
                # 载荷影响系数 (kJ/(m·kg))
                'beta': (loaded_energy_rate - empty_energy_rate) / ref_payload,
                # 速度影响已包含在alpha中
                'gamma': 0,

                # 新增：航程约束参数
                'max_range_empty': power_model.flight_range_at_speed(0, v),
                'max_range_loaded': power_model.flight_range_at_speed(ref_payload, v),

                # 新增：最大航程速度标记
                'is_max_range_speed': abs(v - power_model.max_range_speed(0, max(self.drone_speeds))) < 2,
            }

        return params

    def _build_sets(self):
        """构建节点集合 - 带任务预筛选版本"""
        self.C = self.customers
        self.N = [self.depot] + self.customers
        self.N_L = [self.depot] + self.customers
        self.N_R = self.customers + [self.depot]
        self.V = self.drone_speeds
        self.A = [(i, j) for i in self.N_L for j in self.N_R if i != j]

        # 预先计算距离
        self.coords = self.node_coords
        temp_dist = {}
        for i in self.N:
            for j in self.N:
                if i != j:
                    xi, yi = self.coords[i]
                    xj, yj = self.coords[j]
                    temp_dist[i, j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

        # ==================== 筛选参数（可调整）====================
        max_drone_speed = max(self.drone_speeds)
        MAX_DRONE_DISTANCE = self.max_flight_range * (1 - self.safety_margin)
        MAX_TIME_RATIO =1.5  # 无人机时间/卡车时间的最大比例
        MAX_DISTANCE_RATIO = 2.0  # 无人机距离/卡车距离的最大比例
        # ============================================================

        print("\\n" + "=" * 50)
        print("预筛选无人机任务")
        print("=" * 50)
        print(f"筛选参数: 最大时间比例={MAX_TIME_RATIO}, 最大距离比例={MAX_DISTANCE_RATIO}")

        total = 0
        filtered = {"distance": 0, "load": 0, "ratio": 0, "time": 0}

        self.drone_trips = []

        for i in self.N_L:
            for j in self.N_R:
                if i != j:
                    dist_ij = temp_dist.get((i, j), 0)
                    truck_time_ij = dist_ij / self.truck_speed if dist_ij > 0 else 0

                    for k in self.C:
                        if k != i and k != j:
                            total += 1

                            xi, yi = self.coords[i]
                            xj, yj = self.coords[j]
                            xk, yk = self.coords[k]

                            dist_ik = math.sqrt((xk - xi) ** 2 + (yk - yi) ** 2)
                            dist_kj = math.sqrt((xj - xk) ** 2 + (yj - yk) ** 2)
                            drone_dist = dist_ik + dist_kj

                            # 筛选1：距离限制
                            if drone_dist > MAX_DRONE_DISTANCE:
                                filtered["distance"] += 1
                                continue

                            # 筛选2：载重限制
                            if self.demands.get(k, 1.0) > self.drone_max_load:
                                filtered["load"] += 1
                                continue

                            # 筛选3：距离比例限制
                            if dist_ij > 0 and drone_dist / dist_ij > MAX_DISTANCE_RATIO:
                                filtered["ratio"] += 1
                                continue

                            # 筛选4：时间可行性
                            drone_time = drone_dist / max_drone_speed + self.service_times.get(k, 0)
                            if truck_time_ij > 0 and drone_time > truck_time_ij * MAX_TIME_RATIO:
                                filtered["time"] += 1
                                continue

                            self.drone_trips.append((i, j, k))

        kept = len(self.drone_trips)
        rate = (1 - kept / total) * 100 if total > 0 else 0
        print(f"筛选结果: {total} -> {kept} (筛选率: {rate:.1f}%)")
        print(f"  - 距离超限: {filtered['distance']}")
        print(f"  - 载重超限: {filtered['load']}")
        print(f"  - 比例超限: {filtered['ratio']}")
        print(f"  - 时间不可行: {filtered['time']}")
        print("=" * 50)

        print(f"\\n节点集合构建完成:")
        print(f"  - 客户数: {len(self.C)}")
        print(f"  - 弧数: {len(self.A)}")
        print(f"  - 无人机任务候选数: {len(self.drone_trips)}")
    # def _build_sets(self):
    #     """构建节点集合"""
    #     self.C = self.customers
    #     self.N = [self.depot] + self.customers
    #     self.N_L = [self.depot] + self.customers
    #     self.N_R = self.customers + [self.depot]
    #     self.V = self.drone_speeds
    #     self.A = [(i, j) for i in self.N_L for j in self.N_R if i != j]
    #
    #     # 构建有效的无人机任务三元组 (i, j, k)
    #     # i = 起飞相关节点, j = 回收相关节点, k = 服务客户
    #     self.drone_trips = []
    #     for i in self.N_L:
    #         for j in self.N_R:
    #             if i != j:
    #                 for k in self.C:
    #                     if k != i and k != j:
    #                         self.drone_trips.append((i, j, k))
    #
    #     print(f"节点集合构建完成:")
    #     print(f"  - 客户数: {len(self.C)}")
    #     print(f"  - 弧数: {len(self.A)}")
    #     print(f"  - 无人机任务候选数: {len(self.drone_trips)}")

    def _compute_distances(self):
        """计算距离矩阵"""
        self.dist = {}
        self.coords = self.node_coords

        for i in self.N:
            for j in self.N:
                if i != j:
                    xi, yi = self.coords[i]
                    xj, yj = self.coords[j]
                    self.dist[i, j] = math.sqrt((xi - xj)**2 + (yi - yj)**2)

        self.truck_time = {(i, j): self.dist[i, j] / self.truck_speed
                          for (i, j) in self.dist}

        # 计算最大可能距离（用于变量边界）
        all_coords = list(self.coords.values())
        self.max_dist = max(
            math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            for c1 in all_coords for c2 in all_coords
        ) * 1.5

    def _build_pwl_breakpoints(self):
        """构建分段线性化的断点"""
        # 用于距离-时间关系的分段线性化
        self.pwl_dist_points = np.linspace(0, self.max_dist, self.num_pwl_segments + 1)

        # 用于速度选择的能耗函数分段线性化
        self.pwl_energy = {}
        for v in self.V:
            params = self.energy_params[v]
            # 能耗率随距离线性，这里预计算系数
            self.pwl_energy[v] = {
                'empty': params['alpha'] + params['gamma'] * v**2,
                'loaded': lambda Q, p=params, vel=v: p['alpha'] + p['beta']*Q + p['gamma']*vel**2
            }

    def _add_variables(self):
        """添加决策变量"""
        M = self.big_M

        # ==================== 基础路径变量 ====================
        # x[i,j]: 卡车是否穿越弧(i,j)
        self.vars['x'] = self.model.addVars(
            self.A, vtype=GRB.BINARY, name="x"
        )

        # y[i,j,k]: 无人机任务 - 从弧(a,i)起飞，服务k，在弧(j,b)降落
        # 简化索引：i=起飞弧终点, j=降落弧起点, k=服务客户
        self.vars['y'] = self.model.addVars(
            self.drone_trips, vtype=GRB.BINARY, name="y"
        )

        # z[j]: 客户j是否被服务
        self.vars['z'] = self.model.addVars(
            self.C, vtype=GRB.BINARY, name="z"
        )

        # ==================== 顺序变量 ====================
        # o[i]: 节点i在卡车路线上的访问顺序
        self.vars['o'] = self.model.addVars(
            self.N, lb=0, ub=len(self.C) + 1, vtype=GRB.CONTINUOUS, name="o"
        )

        # ==================== 弧中同步位置变量 ====================
        # lambda_minus[i,j,k]: 起飞点在弧上的位置比例 [0,1]
        self.vars['lambda_minus'] = self.model.addVars(
            self.drone_trips, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="lam_m"
        )

        # lambda_plus[i,j,k]: 回收点在弧上的位置比例 [0,1]
        self.vars['lambda_plus'] = self.model.addVars(
            self.drone_trips, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="lam_p"
        )

        # ==================== 坐标变量 (用于SOCP) ====================
        # 起飞点坐标
        self.vars['launch_x'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="launch_x"
        )
        self.vars['launch_y'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="launch_y"
        )

        # 回收点坐标
        self.vars['recover_x'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="recover_x"
        )
        self.vars['recover_y'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="recover_y"
        )

        # ==================== 距离变量 ====================
        # r_out[i,j,k]: 起飞点到客户k的飞行距离
        self.vars['r_out'] = self.model.addVars(
            self.drone_trips, lb=0, ub=self.max_dist, vtype=GRB.CONTINUOUS, name="r_out"
        )

        # r_ret[i,j,k]: 客户k到回收点的飞行距离
        self.vars['r_ret'] = self.model.addVars(
            self.drone_trips, lb=0, ub=self.max_dist, vtype=GRB.CONTINUOUS, name="r_ret"
        )

        # ==================== 时间变量 ====================
        # tau_arr[i]: 卡车到达节点i的时间
        self.vars['tau_arr'] = self.model.addVars(
            self.N, lb=0, vtype=GRB.CONTINUOUS, name="tau_arr"
        )

        # tau_dep[i]: 卡车从节点i出发的时间
        self.vars['tau_dep'] = self.model.addVars(
            self.N, lb=0, vtype=GRB.CONTINUOUS, name="tau_dep"
        )

        # psi_launch[i,j,k]: 卡车到达起飞点的时刻
        self.vars['psi_launch'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="psi_launch"
        )

        # psi_recover[i,j,k]: 卡车到达回收点的时刻
        self.vars['psi_recover'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="psi_recover"
        )

        # tau_drone_arr[i,j,k]: 无人机到达回收点的时刻
        self.vars['tau_drone_arr'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="tau_drone_arr"
        )

        # ==================== 速度选择变量 ====================
        # v_out[i,j,k,v]: 去程是否选择速度v
        self.vars['v_out'] = self.model.addVars(
            [(i, j, k, v) for (i, j, k) in self.drone_trips for v in self.V],
            vtype=GRB.BINARY, name="v_out"
        )

        # v_ret[i,j,k,v]: 返程是否选择速度v
        self.vars['v_ret'] = self.model.addVars(
            [(i, j, k, v) for (i, j, k) in self.drone_trips for v in self.V],
            vtype=GRB.BINARY, name="v_ret"
        )

        # ==================== 飞行时间变量 ====================
        # t_out[i,j,k]: 去程飞行时间
        self.vars['t_out'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="t_out"
        )

        # t_ret[i,j,k]: 返程飞行时间
        self.vars['t_ret'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="t_ret"
        )

        # ==================== 能耗变量 ====================
        # E_out[i,j,k]: 去程能耗
        self.vars['E_out'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="E_out"
        )

        # E_ret[i,j,k]: 返程能耗
        self.vars['E_ret'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="E_ret"
        )

        # E_total[i,j,k]: 总能耗
        self.vars['E_total'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="E_total"
        )

        # ==================== 等待时间变量 ====================
        # w_drone[i,j,k]: 无人机等待卡车的时间
        self.vars['w_drone'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="w_drone"
        )

        # w_truck[i,j,k]: 卡车等待无人机的时间
        self.vars['w_truck'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="w_truck"
        )

        # ==================== 辅助变量 (用于SOCP和线性化) ====================
        # 距离平方的辅助变量
        self.vars['r_out_sq'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="r_out_sq"
        )
        self.vars['r_ret_sq'] = self.model.addVars(
            self.drone_trips, lb=0, vtype=GRB.CONTINUOUS, name="r_ret_sq"
        )

        # 坐标差的辅助变量
        self.vars['dx_out'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="dx_out"
        )
        self.vars['dy_out'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="dy_out"
        )
        self.vars['dx_ret'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="dx_ret"
        )
        self.vars['dy_ret'] = self.model.addVars(
            self.drone_trips, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="dy_ret"
        )

        # 分段线性化的辅助变量 (用于速度-距离-时间关系)
        # r_v_out[i,j,k,v]: 在速度v下分配的去程距离
        self.vars['r_v_out'] = self.model.addVars(
            [(i, j, k, v) for (i, j, k) in self.drone_trips for v in self.V],
            lb=0, vtype=GRB.CONTINUOUS, name="r_v_out"
        )

        # r_v_ret[i,j,k,v]: 在速度v下分配的返程距离
        self.vars['r_v_ret'] = self.model.addVars(
            [(i, j, k, v) for (i, j, k) in self.drone_trips for v in self.V],
            lb=0, vtype=GRB.CONTINUOUS, name="r_v_ret"
        )

    def _add_objective(self):
        """添加目标函数（改进版）"""
        x = self.vars['x']
        y = self.vars['y']
        r_out = self.vars['r_out']
        r_ret = self.vars['r_ret']
        E_total = self.vars['E_total']
        w_drone = self.vars['w_drone']
        w_truck = self.vars['w_truck']
        v_out = self.vars['v_out']
        v_ret = self.vars['v_ret']

        # 1. 卡车运输成本
        truck_cost = gp.quicksum(
            self.truck_cost * self.dist[i, j] * x[i, j]
            for (i, j) in self.A
        )

        # 2. 无人机飞行距离成本
        drone_dist_cost = gp.quicksum(
            self.drone_cost * (r_out[i, j, k] + r_ret[i, j, k])
            for (i, j, k) in self.drone_trips
        )

        # 3. 能耗成本
        energy_cost = gp.quicksum(
            self.sync_cost * E_total[i, j, k]
            for (i, j, k) in self.drone_trips
        )

        # 4. 等待成本
        wait_cost = gp.quicksum(
            self.wait_cost * (w_drone[i, j, k] + w_truck[i, j, k])
            for (i, j, k) in self.drone_trips
        )

        # 5. 新增：速度选择惩罚（鼓励选择接近最大航程速度的速度）
        # 这可以帮助求解器更快收敛
        speed_penalty_coef = 0.01  # 小权重，不影响主目标
        speed_penalty = gp.quicksum(
            speed_penalty_coef * abs(v - self._get_max_range_speed(k)) *
            (v_out[i, j, k, v] + v_ret[i, j, k, v])
            for (i, j, k) in self.drone_trips
            for v in self.V
        )

        self.model.setObjective(
            truck_cost + drone_dist_cost + energy_cost + wait_cost + speed_penalty,
            GRB.MINIMIZE
        )

    def _get_max_range_speed(self, customer: int) -> float:
        """获取服务某客户的最大航程速度"""
        Q_k = self.demands.get(customer, 1.0)
        # 根据载重选择最接近最大航程的速度
        # 简化：返回速度列表的中间值
        return self.drone_speeds[len(self.drone_speeds) // 2]

    def _add_routing_constraints(self):
        """添加路径约束"""
        x = self.vars['x']
        y = self.vars['y']
        z = self.vars['z']
        o = self.vars['o']
        M = self.big_M

        # 约束 (2): 每个客户恰好被服务一次（卡车或无人机）
        for c in self.C:
            # 被卡车服务
            truck_service = gp.quicksum(x[i, c] for i in self.N_L if (i, c) in x)
            # 被无人机服务
            drone_service = gp.quicksum(y[i, j, c] for (i, j, k) in self.drone_trips if k == c)
            self.model.addConstr(
                truck_service + drone_service == z[c],
                name=f"service_{c}"
            )

        # 所有客户必须被服务
        for c in self.C:
            self.model.addConstr(z[c] == 1, name=f"must_serve_{c}")

        # 约束 (3): 卡车流守恒
        for j in self.C:
            self.model.addConstr(
                gp.quicksum(x[i, j] for i in self.N_L if (i, j) in x) ==
                gp.quicksum(x[j, k] for k in self.N_R if (j, k) in x),
                name=f"flow_{j}"
            )

        # 约束 (12): 从仓库出发和返回
        self.model.addConstr(
            gp.quicksum(x[self.depot, j] for j in self.C if (self.depot, j) in x) == 1,
            name="depart_depot"
        )
        self.model.addConstr(
            gp.quicksum(x[i, self.depot] for i in self.C if (i, self.depot) in x) == 1,
            name="return_depot"
        )

        # 约束 (8)-(11): MTZ子回路消除
        self.model.addConstr(o[self.depot] == 0, name="depot_order")
        for (i, j) in self.A:
            if i != self.depot and j != self.depot:
                self.model.addConstr(
                    o[j] >= o[i] + 1 - M * (1 - x[i, j]),
                    name=f"mtz_{i}_{j}"
                )

        # ==================== 关键约束：无人机任务必须在卡车路径上 ====================
        # 无人机任务 (i, j, k) 的含义：
        # - 无人机在弧 (i, *) 上的某点起飞（即从节点i出发的某条弧）
        # - 无人机在弧 (*, j) 上的某点降落（即到达节点j的某条弧）
        # - 服务客户 k

        for (i, j, k) in self.drone_trips:
            # 起飞约束：必须存在一条从节点i出发的卡车弧
            launch_arcs = [(i, b) for b in self.N_R if (i, b) in x and b != k]
            if launch_arcs:
                self.model.addConstr(
                    y[i, j, k] <= gp.quicksum(x[i, b] for (_, b) in launch_arcs),
                    name=f"launch_arc_{i}_{j}_{k}"
                )
            else:
                # 如果没有可用的起飞弧，禁止该任务
                self.model.addConstr(y[i, j, k] == 0, name=f"no_launch_{i}_{j}_{k}")

            # 回收约束：必须存在一条到达节点j的卡车弧
            recover_arcs = [(a, j) for a in self.N_L if (a, j) in x and a != k]
            if recover_arcs:
                self.model.addConstr(
                    y[i, j, k] <= gp.quicksum(x[a, j] for (a, _) in recover_arcs),
                    name=f"recover_arc_{i}_{j}_{k}"
                )
            else:
                # 如果没有可用的回收弧，禁止该任务
                self.model.addConstr(y[i, j, k] == 0, name=f"no_recover_{i}_{j}_{k}")

            # 顺序约束：起飞节点i必须在回收节点j之前（或同时）被访问
            # o[i] <= o[j] 当 y[i,j,k] = 1
            self.model.addConstr(
                o[i] <= o[j] + M * (1 - y[i, j, k]),
                name=f"order_{i}_{j}_{k}"
            )

    def _add_coordinate_constraints(self):
        """
        添加坐标计算约束（线性插值）

        无人机任务 (i, j, k) 的含义：
        - 起飞点在从节点i出发的弧上，位置比例为 lambda_minus
        - 回收点在到达节点j的弧上，位置比例为 lambda_plus
        - 服务客户 k

        简化处理：假设起飞弧为 (i, next_i)，回收弧为 (prev_j, j)
        由于我们不知道具体是哪条弧，这里使用节点坐标的近似
        """
        y = self.vars['y']
        x = self.vars['x']
        lambda_minus = self.vars['lambda_minus']
        lambda_plus = self.vars['lambda_plus']
        launch_x = self.vars['launch_x']
        launch_y = self.vars['launch_y']
        recover_x = self.vars['recover_x']
        recover_y = self.vars['recover_y']
        M = self.big_M

        # 为每个无人机任务添加坐标约束
        # 使用辅助变量来处理弧的选择
        for (i, j, k) in self.drone_trips:
            xi, yi_c = self.coords[i]
            xj, yj_c = self.coords[j]
            xk, yk_c = self.coords[k]

            # ==================== 起飞点坐标 ====================
            # 起飞点在从节点i出发的某条弧上
            # 对于每条可能的起飞弧 (i, b)，如果 x[i,b]=1 且 y[i,j,k]=1
            # 则 launch_x = xi + lambda_minus * (xb - xi)

            # 简化方法：当 y[i,j,k]=1 时，起飞点坐标范围在节点i附近
            # 实际起飞弧的终点由卡车路径决定

            # 找所有可能的起飞弧终点
            possible_launch_ends = [b for b in self.N_R if (i, b) in x and b != k]

            if possible_launch_ends:
                # 使用加权平均近似
                for b in possible_launch_ends:
                    xb, yb = self.coords[b]

                    # 当 x[i,b]=1 且 y[i,j,k]=1 时，约束生效
                    self.model.addConstr(
                        launch_x[i, j, k] >= xi + lambda_minus[i, j, k] * (xb - xi)
                        - M * (2 - x[i, b] - y[i, j, k]),
                        name=f"launch_x_lb_{i}_{j}_{k}_{b}"
                    )
                    self.model.addConstr(
                        launch_x[i, j, k] <= xi + lambda_minus[i, j, k] * (xb - xi)
                        + M * (2 - x[i, b] - y[i, j, k]),
                        name=f"launch_x_ub_{i}_{j}_{k}_{b}"
                    )
                    self.model.addConstr(
                        launch_y[i, j, k] >= yi_c + lambda_minus[i, j, k] * (yb - yi_c)
                        - M * (2 - x[i, b] - y[i, j, k]),
                        name=f"launch_y_lb_{i}_{j}_{k}_{b}"
                    )
                    self.model.addConstr(
                        launch_y[i, j, k] <= yi_c + lambda_minus[i, j, k] * (yb - yi_c)
                        + M * (2 - x[i, b] - y[i, j, k]),
                        name=f"launch_y_ub_{i}_{j}_{k}_{b}"
                    )
            else:
                # 如果没有可用弧，固定在节点i
                self.model.addConstr(
                    launch_x[i, j, k] >= xi - M * (1 - y[i, j, k]),
                    name=f"launch_x_default_lb_{i}_{j}_{k}"
                )
                self.model.addConstr(
                    launch_x[i, j, k] <= xi + M * (1 - y[i, j, k]),
                    name=f"launch_x_default_ub_{i}_{j}_{k}"
                )
                self.model.addConstr(
                    launch_y[i, j, k] >= yi_c - M * (1 - y[i, j, k]),
                    name=f"launch_y_default_lb_{i}_{j}_{k}"
                )
                self.model.addConstr(
                    launch_y[i, j, k] <= yi_c + M * (1 - y[i, j, k]),
                    name=f"launch_y_default_ub_{i}_{j}_{k}"
                )

            # ==================== 回收点坐标 ====================
            # 回收点在到达节点j的某条弧上
            possible_recover_starts = [a for a in self.N_L if (a, j) in x and a != k]

            if possible_recover_starts:
                for a in possible_recover_starts:
                    xa, ya = self.coords[a]

                    # 当 x[a,j]=1 且 y[i,j,k]=1 时，约束生效
                    self.model.addConstr(
                        recover_x[i, j, k] >= xa + lambda_plus[i, j, k] * (xj - xa)
                        - M * (2 - x[a, j] - y[i, j, k]),
                        name=f"recover_x_lb_{i}_{j}_{k}_{a}"
                    )
                    self.model.addConstr(
                        recover_x[i, j, k] <= xa + lambda_plus[i, j, k] * (xj - xa)
                        + M * (2 - x[a, j] - y[i, j, k]),
                        name=f"recover_x_ub_{i}_{j}_{k}_{a}"
                    )
                    self.model.addConstr(
                        recover_y[i, j, k] >= ya + lambda_plus[i, j, k] * (yj_c - ya)
                        - M * (2 - x[a, j] - y[i, j, k]),
                        name=f"recover_y_lb_{i}_{j}_{k}_{a}"
                    )
                    self.model.addConstr(
                        recover_y[i, j, k] <= ya + lambda_plus[i, j, k] * (yj_c - ya)
                        + M * (2 - x[a, j] - y[i, j, k]),
                        name=f"recover_y_ub_{i}_{j}_{k}_{a}"
                    )
            else:
                # 如果没有可用弧，固定在节点j
                self.model.addConstr(
                    recover_x[i, j, k] >= xj - M * (1 - y[i, j, k]),
                    name=f"recover_x_default_lb_{i}_{j}_{k}"
                )
                self.model.addConstr(
                    recover_x[i, j, k] <= xj + M * (1 - y[i, j, k]),
                    name=f"recover_x_default_ub_{i}_{j}_{k}"
                )
                self.model.addConstr(
                    recover_y[i, j, k] >= yj_c - M * (1 - y[i, j, k]),
                    name=f"recover_y_default_lb_{i}_{j}_{k}"
                )
                self.model.addConstr(
                    recover_y[i, j, k] <= yj_c + M * (1 - y[i, j, k]),
                    name=f"recover_y_default_ub_{i}_{j}_{k}"
                )

            # 起飞点必须在回收点之前到达（时间上）
            # 这个约束在时间约束中处理

    def _add_socp_distance_constraints(self):
        """
        添加SOCP距离约束

        欧几里得距离: r = sqrt(dx² + dy²)
        等价于SOCP约束: r² >= dx² + dy² (当r >= 0时)
        Gurobi形式: r >= ||(dx, dy)||_2
        """
        y = self.vars['y']
        launch_x = self.vars['launch_x']
        launch_y = self.vars['launch_y']
        recover_x = self.vars['recover_x']
        recover_y = self.vars['recover_y']
        r_out = self.vars['r_out']
        r_ret = self.vars['r_ret']
        dx_out = self.vars['dx_out']
        dy_out = self.vars['dy_out']
        dx_ret = self.vars['dx_ret']
        dy_ret = self.vars['dy_ret']
        M = self.big_M

        for (i, j, k) in self.drone_trips:
            xk, yk = self.coords[k]

            # 去程：从起飞点到客户k
            # dx_out = xk - launch_x, dy_out = yk - launch_y
            self.model.addConstr(
                dx_out[i, j, k] >= xk - launch_x[i, j, k] - M * (1 - y[i, j, k]),
                name=f"dx_out_lb_{i}_{j}_{k}"
            )
            self.model.addConstr(
                dx_out[i, j, k] <= xk - launch_x[i, j, k] + M * (1 - y[i, j, k]),
                name=f"dx_out_ub_{i}_{j}_{k}"
            )
            self.model.addConstr(
                dy_out[i, j, k] >= yk - launch_y[i, j, k] - M * (1 - y[i, j, k]),
                name=f"dy_out_lb_{i}_{j}_{k}"
            )
            self.model.addConstr(
                dy_out[i, j, k] <= yk - launch_y[i, j, k] + M * (1 - y[i, j, k]),
                name=f"dy_out_ub_{i}_{j}_{k}"
            )

            # SOCP约束: r_out >= ||(dx_out, dy_out)||_2
            # 等价于: r_out² >= dx_out² + dy_out²
            self.model.addConstr(
                r_out[i, j, k] * r_out[i, j, k] >=
                dx_out[i, j, k] * dx_out[i, j, k] + dy_out[i, j, k] * dy_out[i, j, k],
                name=f"socp_out_{i}_{j}_{k}"
            )

            # 返程：从客户k到回收点
            # dx_ret = recover_x - xk, dy_ret = recover_y - yk
            self.model.addConstr(
                dx_ret[i, j, k] >= recover_x[i, j, k] - xk - M * (1 - y[i, j, k]),
                name=f"dx_ret_lb_{i}_{j}_{k}"
            )
            self.model.addConstr(
                dx_ret[i, j, k] <= recover_x[i, j, k] - xk + M * (1 - y[i, j, k]),
                name=f"dx_ret_ub_{i}_{j}_{k}"
            )
            self.model.addConstr(
                dy_ret[i, j, k] >= recover_y[i, j, k] - yk - M * (1 - y[i, j, k]),
                name=f"dy_ret_lb_{i}_{j}_{k}"
            )
            self.model.addConstr(
                dy_ret[i, j, k] <= recover_y[i, j, k] - yk + M * (1 - y[i, j, k]),
                name=f"dy_ret_ub_{i}_{j}_{k}"
            )

            # SOCP约束: r_ret >= ||(dx_ret, dy_ret)||_2
            self.model.addConstr(
                r_ret[i, j, k] * r_ret[i, j, k] >=
                dx_ret[i, j, k] * dx_ret[i, j, k] + dy_ret[i, j, k] * dy_ret[i, j, k],
                name=f"socp_ret_{i}_{j}_{k}"
            )

            # 当y[i,j,k]=0时，距离为0
            self.model.addConstr(
                r_out[i, j, k] <= M * y[i, j, k],
                name=f"r_out_zero_{i}_{j}_{k}"
            )
            self.model.addConstr(
                r_ret[i, j, k] <= M * y[i, j, k],
                name=f"r_ret_zero_{i}_{j}_{k}"
            )

    def _add_speed_selection_constraints(self):
        """添加速度选择约束（分段线性化）"""
        y = self.vars['y']
        v_out = self.vars['v_out']
        v_ret = self.vars['v_ret']
        r_out = self.vars['r_out']
        r_ret = self.vars['r_ret']
        r_v_out = self.vars['r_v_out']
        r_v_ret = self.vars['r_v_ret']
        t_out = self.vars['t_out']
        t_ret = self.vars['t_ret']
        M = self.big_M

        for (i, j, k) in self.drone_trips:
            # 约束 (40): 去程速度唯一选择
            self.model.addConstr(
                gp.quicksum(v_out[i, j, k, v] for v in self.V) == y[i, j, k],
                name=f"speed_out_unique_{i}_{j}_{k}"
            )

            # 约束 (41): 返程速度唯一选择
            self.model.addConstr(
                gp.quicksum(v_ret[i, j, k, v] for v in self.V) == y[i, j, k],
                name=f"speed_ret_unique_{i}_{j}_{k}"
            )

            # 分段线性化：距离分配
            # r_out = sum_v(r_v_out[v])
            self.model.addConstr(
                r_out[i, j, k] == gp.quicksum(r_v_out[i, j, k, v] for v in self.V),
                name=f"r_out_sum_{i}_{j}_{k}"
            )

            self.model.addConstr(
                r_ret[i, j, k] == gp.quicksum(r_v_ret[i, j, k, v] for v in self.V),
                name=f"r_ret_sum_{i}_{j}_{k}"
            )

            # 速度-距离关联
            for v in self.V:
                # r_v_out只有在选择该速度时才能有值
                self.model.addConstr(
                    r_v_out[i, j, k, v] <= M * v_out[i, j, k, v],
                    name=f"r_v_out_link_{i}_{j}_{k}_{v}"
                )
                self.model.addConstr(
                    r_v_ret[i, j, k, v] <= M * v_ret[i, j, k, v],
                    name=f"r_v_ret_link_{i}_{j}_{k}_{v}"
                )

            # 飞行时间计算: t = r / v
            # 使用分段线性化: t_out = sum_v(r_v_out[v] / v)
            self.model.addConstr(
                t_out[i, j, k] == gp.quicksum(r_v_out[i, j, k, v] / v for v in self.V),
                name=f"t_out_calc_{i}_{j}_{k}"
            )

            self.model.addConstr(
                t_ret[i, j, k] == gp.quicksum(r_v_ret[i, j, k, v] / v for v in self.V),
                name=f"t_ret_calc_{i}_{j}_{k}"
            )

    def _add_energy_constraints(self):
        """添加能耗约束"""
        y = self.vars['y']
        r_v_out = self.vars['r_v_out']
        r_v_ret = self.vars['r_v_ret']
        E_out = self.vars['E_out']
        E_ret = self.vars['E_ret']
        E_total = self.vars['E_total']

        for (i, j, k) in self.drone_trips:
            Q_k = self.demands.get(k, 1.0)  # 客户k的需求量（载荷）

            # 去程能耗（载货）: E_out = sum_v((α_v + β_v*Q + γ_v*v²) * r_v_out[v])
            outbound_energy = gp.quicksum(
                (self.energy_params[v]['alpha'] +
                 self.energy_params[v]['beta'] * Q_k +
                 self.energy_params[v]['gamma'] * v**2) * r_v_out[i, j, k, v]
                for v in self.V
            )
            self.model.addConstr(
                E_out[i, j, k] >= outbound_energy,
                name=f"E_out_{i}_{j}_{k}"
            )

            # 返程能耗（空载）: E_ret = sum_v((α_v + γ_v*v²) * r_v_ret[v])
            return_energy = gp.quicksum(
                (self.energy_params[v]['alpha'] +
                 self.energy_params[v]['gamma'] * v**2) * r_v_ret[i, j, k, v]
                for v in self.V
            )
            self.model.addConstr(
                E_ret[i, j, k] >= return_energy,
                name=f"E_ret_{i}_{j}_{k}"
            )

            # 总能耗
            self.model.addConstr(
                E_total[i, j, k] == E_out[i, j, k] + E_ret[i, j, k],
                name=f"E_total_{i}_{j}_{k}"
            )

            # 约束 (19): 安全冗余能量约束
            self.model.addConstr(
                E_total[i, j, k] <= (1 - self.safety_margin) * self.battery_capacity,
                name=f"energy_safety_{i}_{j}_{k}"
            )

    def _add_time_constraints(self):
        """添加时间约束"""
        x = self.vars['x']
        y = self.vars['y']
        tau_arr = self.vars['tau_arr']
        tau_dep = self.vars['tau_dep']
        psi_launch = self.vars['psi_launch']
        psi_recover = self.vars['psi_recover']
        tau_drone_arr = self.vars['tau_drone_arr']
        t_out = self.vars['t_out']
        t_ret = self.vars['t_ret']
        lambda_minus = self.vars['lambda_minus']
        lambda_plus = self.vars['lambda_plus']
        w_drone = self.vars['w_drone']
        w_truck = self.vars['w_truck']
        M = self.big_M

        # 约束 (13): 卡车时间连续性
        for (i, j) in self.A:
            self.model.addConstr(
                tau_arr[j] >= tau_dep[i] + self.truck_time[i, j] - M * (1 - x[i, j]),
                name=f"truck_time_{i}_{j}"
            )

        # 约束 (14): 服务时间
        for j in self.C:
            self.model.addConstr(
                tau_dep[j] >= tau_arr[j] + self.service_times.get(j, 0),
                name=f"service_time_{j}"
            )

        # 时间窗约束
        for j in self.C:
            e_j, l_j = self.time_windows.get(j, (0, M))
            self.model.addConstr(tau_arr[j] >= e_j, name=f"tw_early_{j}")
            self.model.addConstr(tau_arr[j] <= l_j, name=f"tw_late_{j}")

        # 仓库出发时间
        self.model.addConstr(tau_dep[self.depot] >= 0, name="depot_start")

        for (i, j, k) in self.drone_trips:
            # 卡车到达起飞点时刻
            # psi_launch = tau_dep[i] + lambda_minus * truck_time[i,j]
            self.model.addConstr(
                psi_launch[i, j, k] >= tau_dep[i] +
                lambda_minus[i, j, k] * self.truck_time.get((i, j), 0) - M * (1 - y[i, j, k]),
                name=f"psi_launch_lb_{i}_{j}_{k}"
            )
            self.model.addConstr(
                psi_launch[i, j, k] <= tau_dep[i] +
                lambda_minus[i, j, k] * self.truck_time.get((i, j), 0) + M * (1 - y[i, j, k]),
                name=f"psi_launch_ub_{i}_{j}_{k}"
            )

            # 卡车到达回收点时刻
            # psi_recover = tau_dep[i] + lambda_plus * truck_time[i,j]
            self.model.addConstr(
                psi_recover[i, j, k] >= tau_dep[i] +
                lambda_plus[i, j, k] * self.truck_time.get((i, j), 0) - M * (1 - y[i, j, k]),
                name=f"psi_recover_lb_{i}_{j}_{k}"
            )
            self.model.addConstr(
                psi_recover[i, j, k] <= tau_dep[i] +
                lambda_plus[i, j, k] * self.truck_time.get((i, j), 0) + M * (1 - y[i, j, k]),
                name=f"psi_recover_ub_{i}_{j}_{k}"
            )

            # 无人机到达回收点时刻
            # tau_drone_arr = psi_launch + t_out + service_time[k] + t_ret
            service_k = self.service_times.get(k, 0)
            self.model.addConstr(
                tau_drone_arr[i, j, k] >= psi_launch[i, j, k] + t_out[i, j, k] +
                service_k + t_ret[i, j, k] - M * (1 - y[i, j, k]),
                name=f"drone_arr_lb_{i}_{j}_{k}"
            )
            self.model.addConstr(
                tau_drone_arr[i, j, k] <= psi_launch[i, j, k] + t_out[i, j, k] +
                service_k + t_ret[i, j, k] + M * (1 - y[i, j, k]),
                name=f"drone_arr_ub_{i}_{j}_{k}"
            )

            # 约束 (38)-(39): 同步等待
            # 无人机等待卡车: w_drone = max(0, psi_recover - tau_drone_arr)
            self.model.addConstr(
                w_drone[i, j, k] >= psi_recover[i, j, k] - tau_drone_arr[i, j, k],
                name=f"w_drone_{i}_{j}_{k}"
            )

            # 卡车等待无人机: w_truck = max(0, tau_drone_arr - psi_recover)
            self.model.addConstr(
                w_truck[i, j, k] >= tau_drone_arr[i, j, k] - psi_recover[i, j, k],
                name=f"w_truck_{i}_{j}_{k}"
            )

    def _add_capacity_constraints(self):
        """添加容量约束"""
        y = self.vars['y']

        for (i, j, k) in self.drone_trips:
            # 约束 (18): 无人机载重约束
            Q_k = self.demands.get(k, 1.0)
            if Q_k > self.drone_max_load:
                self.model.addConstr(y[i, j, k] == 0, name=f"capacity_{i}_{j}_{k}")

    def _add_range_constraints(self):
        """
        添加基于速度的航程约束（核心改进）

        这是Raj & Murray (2020)论文的关键贡献：
        不同速度有不同的最大飞行距离
        """
        y = self.vars['y']
        v_out = self.vars['v_out']
        v_ret = self.vars['v_ret']
        r_out = self.vars['r_out']
        r_ret = self.vars['r_ret']
        M = self.big_M

        for (i, j, k) in self.drone_trips:
            Q_k = self.demands.get(k, 1.0)

            for v in self.V:
                params = self.energy_params[v]

                # 去程航程约束（载货）
                # 根据载重插值计算最大航程
                max_range_out = params['max_range_loaded']
                if Q_k < 5.0:  # 插值
                    ratio = Q_k / 5.0
                    max_range_out = (params['max_range_empty'] * (1 - ratio) +
                                     params['max_range_loaded'] * ratio)

                # 留安全余量
                safe_range_out = max_range_out * (1 - self.safety_margin)

                self.model.addConstr(
                    r_out[i, j, k] <= safe_range_out + M * (1 - v_out[i, j, k, v]),
                    name=f"range_out_{i}_{j}_{k}_{v}"
                )

                # 返程航程约束（空载）
                safe_range_ret = params['max_range_empty'] * (1 - self.safety_margin)

                self.model.addConstr(
                    r_ret[i, j, k] <= safe_range_ret + M * (1 - v_ret[i, j, k, v]),
                    name=f"range_ret_{i}_{j}_{k}_{v}"
                )

    def _add_non_overlap_constraints(self):
        """添加无人机任务不重叠约束（修复版）

        确保单无人机顺序执行任务：
        - 任务A完全结束后，任务B才能开始
        - 完成时间 = 无人机到达时间 + 无人机悬停等待时间
        """
        y = self.vars['y']
        psi_launch = self.vars['psi_launch']
        tau_drone_arr = self.vars['tau_drone_arr']
        w_drone = self.vars['w_drone']
        M = self.big_M

        n_tasks = len(self.drone_trips)
        n_constraints = n_tasks * (n_tasks - 1)
        print(f"  任务数: {n_tasks}, 预计约束数: {n_constraints}")

        for idx1, (i1, j1, k1) in enumerate(self.drone_trips):
            for idx2 in range(idx1 + 1, len(self.drone_trips)):
                i2, j2, k2 = self.drone_trips[idx2]

                # 顺序变量：order=1表示任务1先执行
                order = self.model.addVar(vtype=GRB.BINARY, name=f"ord_{idx1}_{idx2}")

                # 约束A: 如果任务1先 (order=1)
                # 任务1完成时间 <= 任务2起飞时间
                # 完成时间 = tau_drone_arr + w_drone（无人机到达并完成等待）
                self.model.addConstr(
                    tau_drone_arr[i1, j1, k1] + w_drone[i1, j1, k1] <=
                    psi_launch[i2, j2, k2] + M * (3 - y[i1, j1, k1] - y[i2, j2, k2] - order),
                    name=f"seq1_{idx1}_{idx2}"
                )

                # 约束B: 如果任务2先 (order=0)
                # 任务2完成时间 <= 任务1起飞时间
                self.model.addConstr(
                    tau_drone_arr[i2, j2, k2] + w_drone[i2, j2, k2] <=
                    psi_launch[i1, j1, k1] + M * (2 - y[i1, j1, k1] - y[i2, j2, k2] + order),
                    name=f"seq2_{idx1}_{idx2}"
                )
    # def _add_non_overlap_constraints(self):
    #     """添加无人机任务不重叠约束"""
    #     y = self.vars['y']
    #     psi_launch = self.vars['psi_launch']
    #     psi_recover = self.vars['psi_recover']
    #     tau_drone_arr = self.vars['tau_drone_arr']
    #     M = self.big_M
    #
    #     # 约束 (50): 无人机任务不重叠
    #     # 对于任意两个不同的无人机任务，其时间必须不重叠
    #     for idx1, (i1, j1, k1) in enumerate(self.drone_trips):
    #         for (i2, j2, k2) in self.drone_trips[idx1+1:]:
    #             if (i1, j1, k1) != (i2, j2, k2):
    #                 # 使用辅助二元变量表示顺序
    #                 order_var = self.model.addVar(vtype=GRB.BINARY,
    #                                               name=f"order_{i1}_{j1}_{k1}_{i2}_{j2}_{k2}")
    #
    #                 # 任务1在任务2之前完成，或任务2在任务1之前完成
    #                 # max(tau_drone_arr[1], psi_recover[1]) <= psi_launch[2] 或反之
    #
    #                 # 如果order_var=1: 任务1先于任务2
    #                 self.model.addConstr(
    #                     tau_drone_arr[i1, j1, k1] <= psi_launch[i2, j2, k2] +
    #                     M * (3 - y[i1, j1, k1] - y[i2, j2, k2] - order_var),
    #                     name=f"no_overlap_a_{i1}_{j1}_{k1}_{i2}_{j2}_{k2}"
    #                 )
    #
    #                 # 如果order_var=0: 任务2先于任务1
    #                 self.model.addConstr(
    #                     tau_drone_arr[i2, j2, k2] <= psi_launch[i1, j1, k1] +
    #                     M * (2 - y[i1, j1, k1] - y[i2, j2, k2] + order_var),
    #                     name=f"no_overlap_b_{i1}_{j1}_{k1}_{i2}_{j2}_{k2}"
    #                 )

    def build(self):
        """构建完整模型"""
        print("=" * 60)
        print("构建 enRTSP-DVS SOCP 模型 (改进版)")
        print("=" * 60)

        print("\n[1/9] 添加决策变量...")
        self._add_variables()

        print("[2/9] 添加目标函数...")
        self._add_objective()

        print("[3/9] 添加路径约束...")
        self._add_routing_constraints()

        print("[4/9] 添加坐标计算约束...")
        self._add_coordinate_constraints()

        print("[5/9] 添加SOCP距离约束...")
        self._add_socp_distance_constraints()

        print("[6/9] 添加速度选择约束（分段线性化）...")
        self._add_speed_selection_constraints()

        print("[7/9] 添加能耗约束...")
        self._add_energy_constraints()

        # ===== 新增 =====
        print("[8/9] 添加航程约束（基于变速理论）...")
        self._add_range_constraints()
        # ================

        print("[9/9] 添加时间和容量约束...")
        self._add_time_constraints()
        self._add_capacity_constraints()

        # # 对于小规模实例，添加不重叠约束
        # if len(self.drone_trips) <= 400:
        #     print("[可选] 添加无人机任务不重叠约束...")
        #     self._add_non_overlap_constraints()
        # 始终添加不重叠约束（单无人机系统必需）
        print("[必需] 添加无人机任务不重叠约束...")
        self._add_non_overlap_constraints()

        # ==================== 零卡车等待约束 ====================
        # 卡车不等待无人机，无人机悬停等待卡车
        print("[必需] 添加零卡车等待约束...")
        w_truck = self.vars['w_truck']
        y = self.vars['y']
        for (i, j, k) in self.drone_trips:
            # 硬约束：当任务被执行时，卡车等待时间必须为0
            self.model.addConstr(
                w_truck[i, j, k] <= self.big_M * (1 - y[i, j, k]),
                name=f"zero_truck_wait_{i}_{j}_{k}"
            )
        # ============================================

        self.model.update()
        print("\n模型构建完成!")
        self.model.update()
        print("\n模型构建完成!")
        print(f"  - 变量数量: {self.model.NumVars}")
        print(f"  - 约束数量: {self.model.NumConstrs}")
        print(f"  - 二元变量: {self.model.NumBinVars}")
        print(f"  - 二次约束: {self.model.NumQConstrs}")

    def solve(self, time_limit: float = 3600, gap: float = 0.01, verbose: bool = True):
        """求解模型"""
        # 设置求解参数
        self.model.setParam('TimeLimit', time_limit)
        self.model.setParam('MIPGap', gap)
        self.model.setParam('OutputFlag', 1 if verbose else 0)

        # SOCP相关参数
        self.model.setParam('NonConvex', 2)  # 允许非凸二次约束

        # 优化求解器性能的参数
        self.model.setParam('MIPFocus', 1)  # 聚焦于找到更好的可行解
        self.model.setParam('Heuristics', 0.5)  # 增加启发式搜索比例
        self.model.setParam('Cuts', 2)  # 增加切割平面

        print(f"\n开始求解...")
        print(f"  - 时间限制: {time_limit}秒")
        print(f"  - MIP Gap: {gap}")
        print("-" * 60)

        self.model.optimize()

        print("-" * 60)
        if self.model.Status == GRB.OPTIMAL:
            print(f"找到最优解! 目标值: {self.model.ObjVal:.4f}")
            print(f"  最终 MIP Gap: {self.model.MIPGap*100:.4f}%")
        elif self.model.Status == GRB.TIME_LIMIT:
            if self.model.SolCount > 0:
                print(f"达到时间限制. 最佳解: {self.model.ObjVal:.4f}, Gap: {self.model.MIPGap*100:.2f}%")
                print(f"  注意: Gap={self.model.MIPGap*100:.2f}% 表示当前解可能不是最优解")
            else:
                print("达到时间限制，未找到可行解")
        elif self.model.Status == GRB.INFEASIBLE:
            print("模型不可行!")
            print("计算IIS...")
            self.model.computeIIS()
            self.model.write("infeasible.ilp")
            print("IIS已保存到 infeasible.ilp")
        else:
            print(f"求解状态: {self.model.Status}")

    def get_solution(self) -> Optional[Dict]:
        """获取求解结果"""
        if self.model.SolCount == 0:
            return None

        solution = {
            'objective': self.model.ObjVal,
            'truck_route': [],
            'drone_missions': [],
            'total_truck_dist': 0,
            'total_drone_dist': 0,
            'total_energy': 0,
            'total_wait_time': 0,
        }

        # ==================== 提取卡车路径（修复版）====================
        x = self.vars['x']

        # 首先收集所有被选中的弧
        selected_arcs = []
        for (i, j) in self.A:
            if x[i, j].X > 0.5:
                selected_arcs.append((i, j))

        # 从仓库开始构建完整路径
        current = self.depot
        visited_nodes = {self.depot}
        max_iterations = len(self.N) + 1  # 防止无限循环
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            found_next = False

            # 查找从当前节点出发的弧
            for (i, j) in selected_arcs:
                if i == current:
                    # 如果目标是仓库，这是最后一条弧
                    if j == self.depot:
                        solution['truck_route'].append((current, j))
                        solution['total_truck_dist'] += self.dist.get((current, j), 0)
                        found_next = True
                        current = j  # 回到仓库
                        break
                    # 如果目标不是仓库且未访问过
                    elif j not in visited_nodes:
                        solution['truck_route'].append((current, j))
                        solution['total_truck_dist'] += self.dist.get((current, j), 0)
                        visited_nodes.add(j)
                        current = j
                        found_next = True
                        break

            # 如果回到仓库或找不到下一条弧，结束
            if current == self.depot and len(solution['truck_route']) > 0:
                break
            if not found_next:
                # 尝试找回仓库的弧（可能之前被跳过了）
                for (i, j) in selected_arcs:
                    if i == current and j == self.depot:
                        solution['truck_route'].append((current, j))
                        solution['total_truck_dist'] += self.dist.get((current, j), 0)
                        current = j
                        break
                break

        # 验证路径是否形成回路
        if solution['truck_route']:
            first_node = solution['truck_route'][0][0]
            last_node = solution['truck_route'][-1][1]
            if first_node != self.depot or last_node != self.depot:
                print(f"警告: 卡车路径未形成完整回路!")
                print(f"  起点: {first_node}, 终点: {last_node}")

        # ==================== 提取无人机任务 ====================
        y = self.vars['y']
        r_out = self.vars['r_out']
        r_ret = self.vars['r_ret']
        E_total = self.vars['E_total']
        w_drone = self.vars['w_drone']
        w_truck = self.vars['w_truck']
        v_out = self.vars['v_out']
        v_ret = self.vars['v_ret']
        lambda_minus = self.vars['lambda_minus']
        lambda_plus = self.vars['lambda_plus']

        for (i, j, k) in self.drone_trips:
            if y[i, j, k].X > 0.5:
                # 找到选择的速度
                speed_out = None
                speed_ret = None
                for v in self.V:
                    if v_out[i, j, k, v].X > 0.5:
                        speed_out = v
                    if v_ret[i, j, k, v].X > 0.5:
                        speed_ret = v

                mission = {
                    'launch_node': i,
                    'customer': k,
                    'recover_node': j,
                    'launch_lambda': lambda_minus[i, j, k].X,
                    'recover_lambda': lambda_plus[i, j, k].X,
                    'outbound_distance': r_out[i, j, k].X,
                    'return_distance': r_ret[i, j, k].X,
                    'outbound_speed': speed_out,
                    'return_speed': speed_ret,
                    'energy': E_total[i, j, k].X,
                    'drone_wait': w_drone[i, j, k].X,
                    'truck_wait': w_truck[i, j, k].X,
                }
                solution['drone_missions'].append(mission)
                solution['total_drone_dist'] += mission['outbound_distance'] + mission['return_distance']
                solution['total_energy'] += mission['energy']
                solution['total_wait_time'] += mission['drone_wait'] + mission['truck_wait']

        return solution

    def print_solution(self, solution: Dict):
        """打印解的详细信息"""
        if solution is None:
            print("没有可行解")
            return

        print("\n" + "=" * 70)
        print("求解结果详情")
        print("=" * 70)

        print(f"\n目标函数值: {solution['objective']:.4f}")
        print(f"卡车总行驶距离: {solution['total_truck_dist']:.2f}")
        print(f"无人机总飞行距离: {solution['total_drone_dist']:.2f}")
        print(f"总能耗: {solution['total_energy']:.2f} Wh")
        print(f"总等待时间: {solution['total_wait_time']:.2f}")

        # 打印完整卡车路径
        print(f"\n卡车路径 ({len(solution['truck_route'])} 段弧):")
        if solution['truck_route']:
            route_str = str(solution['truck_route'][0][0])  # 起点
            for (i, j) in solution['truck_route']:
                route_str += f" -> {j}"
            print(f"  {route_str}")

            # 验证是否形成回路
            start = solution['truck_route'][0][0]
            end = solution['truck_route'][-1][1]
            if start == 0 and end == 0:
                print(f"  ✓ 路径形成完整回路 (仓库 -> ... -> 仓库)")
            else:
                print(f"  ✗ 警告: 路径未形成完整回路 (起点={start}, 终点={end})")

            # 检测路径交叉
            crossings = self._detect_route_crossings(solution['truck_route'])
            if crossings:
                print(f"  ⚠ 警告: 检测到 {len(crossings)} 处路径交叉!")
                print(f"    这可能表示当前解不是最优解，建议:")
                print(f"    1. 增加求解时间 (time_limit)")
                print(f"    2. 减小 MIP Gap")
                for c in crossings[:3]:  # 只显示前3个
                    print(f"    - 弧 {c[0]} 与弧 {c[1]} 交叉")
            else:
                print(f"  ✓ 路径无交叉")
        else:
            print("  无卡车路径")

        # 打印卡车访问的节点
        truck_visited = set()
        for (i, j) in solution['truck_route']:
            if i != 0:
                truck_visited.add(i)
            if j != 0:
                truck_visited.add(j)
        print(f"  卡车访问的客户节点: {sorted(truck_visited)}")

        print(f"\n无人机任务 ({len(solution['drone_missions'])} 个):")
        for idx, mission in enumerate(solution['drone_missions'], 1):
            print(f"\n  任务 {idx}:")
            print(f"    服务客户: {mission['customer']}")
            print(f"    起飞节点: {mission['launch_node']} (λ={mission['launch_lambda']:.3f})")
            print(f"    回收节点: {mission['recover_node']} (λ={mission['recover_lambda']:.3f})")
            print(f"    去程: {mission['outbound_distance']:.2f}m @ {mission['outbound_speed']}m/s")
            print(f"    返程: {mission['return_distance']:.2f}m @ {mission['return_speed']}m/s")
            print(f"    能耗: {mission['energy']:.2f} Wh")
            if mission['drone_wait'] > 0.01:
                print(f"    无人机等待: {mission['drone_wait']:.2f}s")
            if mission['truck_wait'] > 0.01:
                print(f"    卡车等待: {mission['truck_wait']:.2f}s")

    def _detect_route_crossings(self, truck_route: List[Tuple[int, int]]) -> List[Tuple]:
        """
        检测卡车路径中的交叉

        Args:
            truck_route: 卡车路径弧列表

        Returns:
            交叉的弧对列表
        """
        def ccw(A, B, C):
            """判断三点是否逆时针排列"""
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def segments_intersect(A, B, C, D):
            """判断线段AB和CD是否相交（不包括端点重合的情况）"""
            # 如果共享端点，不算交叉
            if A == C or A == D or B == C or B == D:
                return False
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        crossings = []
        n = len(truck_route)

        for i in range(n):
            for j in range(i + 2, n):  # 跳过相邻弧
                arc1 = truck_route[i]
                arc2 = truck_route[j]

                # 获取弧的端点坐标
                A = self.coords[arc1[0]]
                B = self.coords[arc1[1]]
                C = self.coords[arc2[0]]
                D = self.coords[arc2[1]]

                if segments_intersect(A, B, C, D):
                    crossings.append((arc1, arc2))

        return crossings


def create_small_instance():
    """创建小规模测试实例"""
    node_coords = {
        0: (50, 50),   # 仓库
        1: (30, 70),
        2: (70, 80),
        3: (80, 40),
        4: (20, 30),
        5: (60, 20),
    }

    customers = [1, 2, 3, 4, 5]
    demands = {1: 1.0, 2: 1.5, 3: 2.0, 4: 1.0, 5: 1.5}
    time_windows = {c: (0, 1000) for c in customers}
    service_times = {c: 5 for c in customers}

    return node_coords, customers, demands, time_windows, service_times


def create_medium_instance():
    """创建中等规模测试实例"""
    np.random.seed(42)

    n_customers = 10

    # 随机生成节点坐标
    node_coords = {0: (50, 50)}  # 仓库
    for i in range(1, n_customers + 1):
        node_coords[i] = (np.random.uniform(10, 90), np.random.uniform(10, 90))

    customers = list(range(1, n_customers + 1))
    demands = {c: np.random.uniform(0.5, 3.0) for c in customers}
    time_windows = {c: (0, 2000) for c in customers}
    service_times = {c: np.random.uniform(3, 8) for c in customers}

    return node_coords, customers, demands, time_windows, service_times


def create_solomon_instance(num_customers: int = 10, seed: int = 42, filepath: str = None):
    """
    从Solomon算例创建实例

    Args:
        num_customers: 需要的客户数量
        seed: 随机种子
        filepath: 算例文件路径，None则使用内置C101

    Returns:
        node_coords, customers, demands, time_windows, service_times
    """
    instance = load_solomon_instance(
        filepath=filepath,
        num_customers=num_customers,
        random_select=True,
        seed=seed
    )

    return (
        instance['node_coords'],
        instance['customers'],
        instance['demands'],
        instance['time_windows'],
        instance['service_times']
    )


def visualize_solution(
    node_coords: Dict[int, Tuple[float, float]],
    solution: Dict,
    title: str = "enRTSP-DVS Solution",
    save_path: str = None,
    show_legend: bool = True,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    可视化求解结果

    Args:
        node_coords: 节点坐标字典 {node_id: (x, y)}
        solution: 求解结果字典
        title: 图标题
        save_path: 保存路径，None则不保存
        show_legend: 是否显示图例
        figsize: 图片大小
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    # 设置中文字体（如果系统支持）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=figsize)

    # 获取所有被卡车服务的客户和被无人机服务的客户
    drone_served_customers = set()
    for mission in solution.get('drone_missions', []):
        drone_served_customers.add(mission['customer'])

    truck_route = solution.get('truck_route', [])
    truck_visited = set()
    for (i, j) in truck_route:
        if i != 0:
            truck_visited.add(i)
        if j != 0:
            truck_visited.add(j)

    # 被卡车直接服务的客户（在卡车路径上但不是无人机服务的）
    truck_served_customers = truck_visited - drone_served_customers

    # ==================== 绘制节点 ====================
    # 仓库
    depot_coord = node_coords[0]
    ax.scatter(*depot_coord, c='red', s=300, marker='s', zorder=10,
               edgecolors='darkred', linewidths=2, label='Depot')
    ax.annotate('D', depot_coord, fontsize=14, ha='center', va='center',
                fontweight='bold', color='white')

    # 卡车服务的客户（蓝色）
    for c in truck_served_customers:
        coord = node_coords[c]
        ax.scatter(*coord, c='royalblue', s=200, marker='o', zorder=8,
                  edgecolors='darkblue', linewidths=1.5)
        ax.annotate(str(c), (coord[0]+1, coord[1]+1), fontsize=10,
                   fontweight='bold', color='darkblue')

    # 无人机服务的客户（橙色星形）
    for c in drone_served_customers:
        coord = node_coords[c]
        ax.scatter(*coord, c='orange', s=300, marker='*', zorder=9,
                  edgecolors='darkorange', linewidths=1.5)
        ax.annotate(str(c), (coord[0]+1, coord[1]+1), fontsize=10,
                   fontweight='bold', color='darkorange')

    # 未被服务的客户（灰色，如果有的话）
    all_customers = set(node_coords.keys()) - {0}
    unserved = all_customers - truck_served_customers - drone_served_customers
    for c in unserved:
        coord = node_coords[c]
        ax.scatter(*coord, c='lightgray', s=150, marker='o', zorder=5,
                  edgecolors='gray', linewidths=1)
        ax.annotate(str(c), (coord[0]+1, coord[1]+1), fontsize=9, color='gray')

    # ==================== 绘制卡车路径 ====================
    for (i, j) in truck_route:
        xi, yi = node_coords[i]
        xj, yj = node_coords[j]

        # 绘制卡车路径（绿色实线箭头）
        ax.annotate(
            '', xy=(xj, yj), xytext=(xi, yi),
            arrowprops=dict(
                arrowstyle='-|>',
                color='green',
                lw=2.5,
                mutation_scale=15
            ),
            zorder=3
        )

    # ==================== 绘制无人机任务 ====================
    # 使用不同颜色区分不同的无人机任务
    drone_colors = plt.cm.tab10(np.linspace(0, 1, max(len(solution.get('drone_missions', [])), 1) + 1))

    for idx, mission in enumerate(solution.get('drone_missions', [])):
        color = drone_colors[idx]

        launch_node = mission['launch_node']
        recover_node = mission['recover_node']
        customer = mission['customer']
        launch_lambda = mission.get('launch_lambda', 0)
        recover_lambda = mission.get('recover_lambda', 1)

        customer_coord = node_coords[customer]

        # ==================== 计算起飞点坐标 ====================
        # 起飞点在从 launch_node 出发的弧上
        # 查找 (launch_node, ?) 的弧
        launch_x, launch_y = node_coords[launch_node]  # 默认
        launch_arc_found = False

        for (i, j) in truck_route:
            if i == launch_node:
                # 找到起飞弧 (launch_node, j)
                xi, yi = node_coords[i]
                xj, yj = node_coords[j]
                launch_x = xi + launch_lambda * (xj - xi)
                launch_y = yi + launch_lambda * (yj - yi)
                launch_arc_found = True
                break

        # ==================== 计算回收点坐标 ====================
        # 回收点在到达 recover_node 的弧上
        # 查找 (?, recover_node) 的弧
        recover_x, recover_y = node_coords[recover_node]  # 默认
        recover_arc_found = False

        for (i, j) in truck_route:
            if j == recover_node:
                # 找到回收弧 (i, recover_node)
                xi, yi = node_coords[i]
                xj, yj = node_coords[j]
                recover_x = xi + recover_lambda * (xj - xi)
                recover_y = yi + recover_lambda * (yj - yi)
                recover_arc_found = True
                break

        # 绘制起飞点（三角形向上）
        ax.scatter(launch_x, launch_y, c=[color], s=150, marker='^', zorder=7,
                  edgecolors='black', linewidths=1.5)

        # 绘制回收点（三角形向下）
        ax.scatter(recover_x, recover_y, c=[color], s=150, marker='v', zorder=7,
                  edgecolors='black', linewidths=1.5)

        # 绘制无人机去程路径（虚线）
        ax.plot([launch_x, customer_coord[0]], [launch_y, customer_coord[1]],
               '--', color=color, lw=2, zorder=2, alpha=0.8)

        # 绘制无人机返程路径（虚线）
        ax.plot([customer_coord[0], recover_x], [customer_coord[1], recover_y],
               '--', color=color, lw=2, zorder=2, alpha=0.8)

        # 在路径上添加箭头表示方向
        # 去程箭头
        mid_out_x = (launch_x + customer_coord[0]) / 2
        mid_out_y = (launch_y + customer_coord[1]) / 2
        dx_out = customer_coord[0] - launch_x
        dy_out = customer_coord[1] - launch_y
        if abs(dx_out) > 0.1 or abs(dy_out) > 0.1:
            ax.annotate('', xy=(mid_out_x + dx_out*0.08, mid_out_y + dy_out*0.08),
                       xytext=(mid_out_x - dx_out*0.08, mid_out_y - dy_out*0.08),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                       zorder=2)

        # 返程箭头
        mid_ret_x = (customer_coord[0] + recover_x) / 2
        mid_ret_y = (customer_coord[1] + recover_y) / 2
        dx_ret = recover_x - customer_coord[0]
        dy_ret = recover_y - customer_coord[1]
        if abs(dx_ret) > 0.1 or abs(dy_ret) > 0.1:
            ax.annotate('', xy=(mid_ret_x + dx_ret*0.08, mid_ret_y + dy_ret*0.08),
                       xytext=(mid_ret_x - dx_ret*0.08, mid_ret_y - dy_ret*0.08),
                       arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                       zorder=2)

    # ==================== 添加图例 ====================
    if show_legend:
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markersize=15, label='Depot (仓库)', markeredgecolor='darkred'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='royalblue',
                   markersize=12, label='Truck-served Customer (卡车服务)', markeredgecolor='darkblue'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='orange',
                   markersize=15, label='Drone-served Customer (无人机服务)', markeredgecolor='darkorange'),
            Line2D([0], [0], color='green', lw=2.5, label='Truck Route (卡车路径)'),
            Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Drone Flight (无人机飞行)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                   markersize=10, label='Launch Point (起飞点)', markeredgecolor='black'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='gray',
                   markersize=10, label='Recovery Point (回收点)', markeredgecolor='black'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
                 framealpha=0.9, edgecolor='gray')

    # ==================== 添加统计信息框 ====================
    info_text = (
        f"Objective: {solution.get('objective', 'N/A'):.2f}\n"
        f"Truck Distance: {solution.get('total_truck_dist', 'N/A'):.2f}\n"
        f"Drone Distance: {solution.get('total_drone_dist', 'N/A'):.2f}\n"
        f"Total Energy: {solution.get('total_energy', 'N/A'):.2f} Wh\n"
        f"Drone Missions: {len(solution.get('drone_missions', []))}\n"
        f"Total Wait Time: {solution.get('total_wait_time', 'N/A'):.2f}s"
    )

    # 放置在右上角
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', horizontalalignment='right',
           bbox=props, family='monospace')

    # ==================== 设置图形属性 ====================
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')

    # 设置坐标轴范围（留一些边距）
    all_x = [coord[0] for coord in node_coords.values()]
    all_y = [coord[1] for coord in node_coords.values()]
    margin = 5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"\n图片已保存到: {save_path}")

    plt.show()

    return fig, ax


def visualize_drone_details(
    solution: Dict,
    figsize: Tuple[int, int] = (12, 6),
    save_path: str = None
):
    """
    可视化无人机任务详情（速度、能耗、等待时间）

    Args:
        solution: 求解结果字典
        figsize: 图片大小
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt

    missions = solution.get('drone_missions', [])
    if not missions:
        print("没有无人机任务可视化")
        return

    n_missions = len(missions)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 数据准备
    customers = [m['customer'] for m in missions]
    out_speeds = [m.get('outbound_speed', 0) for m in missions]
    ret_speeds = [m.get('return_speed', 0) for m in missions]
    energies = [m.get('energy', 0) for m in missions]
    out_dists = [m.get('outbound_dist', m.get('outbound_distance', 0)) for m in missions]
    ret_dists = [m.get('return_dist', m.get('return_distance', 0)) for m in missions]
    drone_waits = [m.get('drone_wait', 0) for m in missions]
    truck_waits = [m.get('truck_wait', 0) for m in missions]

    x = np.arange(n_missions)
    width = 0.35

    # 图1: 速度选择
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, out_speeds, width, label='Outbound (去程)', color='steelblue')
    bars2 = ax1.bar(x + width/2, ret_speeds, width, label='Return (返程)', color='coral')
    ax1.set_xlabel('Mission (任务)')
    ax1.set_ylabel('Speed (m/s)')
    ax1.set_title('Drone Speed Selection (无人机速度选择)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'C{c}' for c in customers])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 图2: 飞行距离
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width/2, out_dists, width, label='Outbound (去程)', color='steelblue')
    bars4 = ax2.bar(x + width/2, ret_dists, width, label='Return (返程)', color='coral')
    ax2.set_xlabel('Mission (任务)')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Flight Distance (飞行距离)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'C{c}' for c in customers])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 图3: 能耗
    ax3 = axes[1, 0]
    colors = plt.cm.Reds(np.linspace(0.3, 0.8, n_missions))
    bars5 = ax3.bar(x, energies, color=colors, edgecolor='darkred')
    ax3.set_xlabel('Mission (任务)')
    ax3.set_ylabel('Energy (Wh)')
    ax3.set_title('Energy Consumption (能耗)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'C{c}' for c in customers])
    ax3.grid(axis='y', alpha=0.3)
    # 添加数值标签
    for bar, val in zip(bars5, energies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # 图4: 等待时间
    ax4 = axes[1, 1]
    bars6 = ax4.bar(x - width/2, drone_waits, width, label='Drone Wait (无人机等待)', color='lightblue')
    bars7 = ax4.bar(x + width/2, truck_waits, width, label='Truck Wait (卡车等待)', color='lightgreen')
    ax4.set_xlabel('Mission (任务)')
    ax4.set_ylabel('Wait Time (s)')
    ax4.set_title('Synchronization Wait Time (同步等待时间)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'C{c}' for c in customers])
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('Drone Mission Details (无人机任务详情)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n详情图已保存到: {save_path}")

    plt.show()

    return fig


def main():
    """主函数"""
    print("=" * 70)
    print("弧中同步的速度可变无人机与卡车配送优化问题 (enRTSP-DVS)")
    print("SOCP + 分段线性化 版本")
    print("=" * 70)

    # ==================== 配置区域 ====================
    # 选择实例类型: "small", "medium", "solomon"
    INSTANCE_TYPE = "solomon"

    # Solomon算例配置
    SOLOMON_NUM_CUSTOMERS = 8   # 从Solomon算例中选择的客户数量
    SOLOMON_SEED = 2            # 随机种子（用于随机选择客户）
    SOLOMON_FILEPATH = "D:\研究生、\python\solomon-100\In\c101.txt"      # Solomon文件路径，None则使用内置C101

    # 求解器配置
    TIME_LIMIT = 600             # 求解时间限制（秒）- 增加到10分钟
    MIP_GAP = 0.01               # MIP Gap - 减小到1%以获得更优解

    # 无人机配置
    DRONE_SPEEDS = [
    12.0,   # 接近空载最大航程速度（14 m/s）
    15.0,   # 低速节能
    18.0,   # 中速平衡
    22.0,   # 接近中载最大航程速度
    25.0,   # 接近满载最大航程速度
    30.0,   # 高速（需要时）
] # 无人机速度选项 (m/s)
    TRUCK_SPEED = 10.0                       # 卡车速度 (m/s)
    BATTERY_CAPACITY = 500.0                 # 电池容量 (Wh)
    DRONE_MAX_LOAD = 50.0                    # 无人机最大载重 (kg)
    # ==================== 配置结束 ====================

    print(f"\n实例类型: {INSTANCE_TYPE}")

    if INSTANCE_TYPE == "small":
        print("使用小规模实例 (5个客户)")
        node_coords, customers, demands, time_windows, service_times = create_small_instance()

    elif INSTANCE_TYPE == "medium":
        print("使用中等规模实例 (10个客户)")
        node_coords, customers, demands, time_windows, service_times = create_medium_instance()

    elif INSTANCE_TYPE == "solomon":
        print(f"使用Solomon算例 ({SOLOMON_NUM_CUSTOMERS}个客户)")
        node_coords, customers, demands, time_windows, service_times = create_solomon_instance(
            num_customers=SOLOMON_NUM_CUSTOMERS,
            seed=SOLOMON_SEED,
            filepath=SOLOMON_FILEPATH
        )
        # ===== 修改1: 缩短服务时间 =====
        SERVICE_TIME_FACTOR = 0.1  # 90秒 → 9秒
        service_times = {k: v * SERVICE_TIME_FACTOR for k, v in service_times.items()}
        print(f"  - 服务时间已调整为原来的 {SERVICE_TIME_FACTOR * 100}%")

        # ===== 修改2: 放大坐标 =====
        COORD_SCALE = 10.0  # 坐标放大10倍
        node_coords = {k: (v[0] * COORD_SCALE, v[1] * COORD_SCALE)
                       for k, v in node_coords.items()}
        print(f"  - 坐标已放大 {COORD_SCALE} 倍")
        time_windows = {k: (v[0] * COORD_SCALE, v[1] * COORD_SCALE)
                        for k, v in time_windows.items()}
        print(f"  - 时间窗已放大 {COORD_SCALE} 倍")
    else:
        raise ValueError(f"未知实例类型: {INSTANCE_TYPE}")

    print(f"\n实例信息:")
    print(f"  - 仓库位置: {node_coords[0]}")
    print(f"  - 客户数量: {len(customers)}")
    print(f"  - 无人机速度选项: {DRONE_SPEEDS} m/s")
    print(f"  - 卡车速度: {TRUCK_SPEED} m/s")

    # 创建模型
    model = EnRTSPDVSModelSOCP(
        node_coords=node_coords,
        customers=customers,
        depot=0,
        demands=demands,
        time_windows=time_windows,
        service_times=service_times,
        drone_speeds=DRONE_SPEEDS,
        truck_speed=TRUCK_SPEED,
        battery_capacity=BATTERY_CAPACITY,
        drone_max_load=DRONE_MAX_LOAD,
        drone_empty_weight=2.0,
        truck_cost=1.0,
        drone_cost=0.5,
        sync_cost=0.1,
        wait_cost=0.5,
        safety_margin=0.1,
        num_pwl_segments=10,
    )

    # 构建模型
    model.build()

    # 求解模型
    model.solve(time_limit=TIME_LIMIT, gap=MIP_GAP, verbose=True)

    # 获取并打印解
    solution = model.get_solution()
    model.print_solution(solution)

    # ==================== 可视化 ====================
    if solution is not None:
        print("\n" + "=" * 70)
        print("生成可视化图表...")
        print("=" * 70)

        # 可视化路径
        visualize_solution(
            node_coords=node_coords,
            solution=solution,
            title=f"enRTSP-DVS Solution ({len(customers)} customers)",
            save_path="solution_route.png",  # 保存路径，设为None则不保存
            show_legend=True
        )

        # 可视化无人机任务详情
        if solution.get('drone_missions'):
            visualize_drone_details(
                solution=solution,
                save_path="solution_drone_details.png"  # 保存路径
            )

    return model, solution


def run_solomon_experiment(
    num_customers_list: List[int] = [5, 8, 10, 12, 15],
    seeds: List[int] = [42, 123, 456],
    filepath: str = None,
    time_limit: float = 300,
    gap: float = 0.05
):
    """
    运行多组Solomon算例实验

    Args:
        num_customers_list: 客户数量列表
        seeds: 随机种子列表
        filepath: 算例文件路径
        time_limit: 求解时间限制
        gap: MIP Gap

    Returns:
        实验结果列表
    """
    results = []

    for n_cust in num_customers_list:
        for seed in seeds:
            print(f"\n{'='*70}")
            print(f"实验: {n_cust}个客户, 种子={seed}")
            print('='*70)

            try:
                # 加载实例
                node_coords, customers, demands, time_windows, service_times = \
                    create_solomon_instance(n_cust, seed, filepath)
                
                # 创建模型
                model = EnRTSPDVSModelSOCP(
                    node_coords=node_coords,
                    customers=customers,
                    depot=0,
                    demands=demands,
                    time_windows=time_windows,
                    service_times=service_times,
                    drone_speeds=[5.0, 10.0, 15.0, 20.0],
                    truck_speed=10.0,
                    battery_capacity=500.0,
                    drone_max_load=50.0,
                )
                
                # 构建并求解
                model.build()
                model.solve(time_limit=time_limit, gap=gap, verbose=False)
                
                # 获取结果
                solution = model.get_solution()
                
                result = {
                    'num_customers': n_cust,
                    'seed': seed,
                    'objective': solution['objective'] if solution else None,
                    'truck_dist': solution['total_truck_dist'] if solution else None,
                    'drone_dist': solution['total_drone_dist'] if solution else None,
                    'energy': solution['total_energy'] if solution else None,
                    'num_drone_missions': len(solution['drone_missions']) if solution else 0,
                    'gap': model.model.MIPGap if model.model.SolCount > 0 else None,
                    'status': model.model.Status,
                }
                results.append(result)
                
                print(f"结果: 目标={result['objective']:.2f}, "
                      f"卡车={result['truck_dist']:.2f}, "
                      f"无人机={result['drone_dist']:.2f}")
                
            except Exception as e:
                print(f"错误: {e}")
                results.append({
                    'num_customers': n_cust,
                    'seed': seed,
                    'error': str(e)
                })
    
    return results


if __name__ == "__main__":
    model, solution = main()
    
    # 如果需要运行批量实验，取消下面的注释
    # results = run_solomon_experiment(
    #     num_customers_list=[5, 8, 10],
    #     seeds=[42, 123],
    #     time_limit=120
    # )

