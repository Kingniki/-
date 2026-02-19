"""
Solomon算例客户节点可视化工具

用于可视化Solomon标准算例文件中的客户节点分布
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def parse_solomon_file(filepath: str):
    """
    解析Solomon算例文件

    返回:
        instance_name: 实例名称
        depot: (x, y, demand, ready_time, due_date, service_time)
        customers: [(id, x, y, demand, ready_time, due_date, service_time), ...]
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    instance_name = lines[0].strip()

    # 解析车辆信息
    vehicle_number = 0
    vehicle_capacity = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("NUMBER") and "CAPACITY" in stripped:
            parts = lines[i + 1].split()
            if len(parts) >= 2:
                vehicle_number = int(parts[0])
                vehicle_capacity = int(parts[1])
            break

    # 解析节点数据
    depot = None
    customers = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 7:
            try:
                cust_no = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])

                if cust_no == 0:
                    depot = (x, y, demand, ready_time, due_date, service_time)
                else:
                    customers.append((cust_no, x, y, demand, ready_time, due_date, service_time))
            except ValueError:
                continue

    return {
        'name': instance_name,
        'vehicle_number': vehicle_number,
        'vehicle_capacity': vehicle_capacity,
        'depot': depot,
        'customers': customers
    }


def visualize_solomon_nodes(
    filepath: str,
    save_path: str = None,
    show_demand: bool = True,
    show_time_windows: bool = False,
    show_customer_id: bool = True,
    num_customers: int = None,
    figsize: tuple = (12, 10),
    title: str = None
):
    """
    可视化Solomon算例的客户节点

    参数:
        filepath: Solomon .txt 文件路径
        save_path: 图片保存路径 (None则自动生成)
        show_demand: 是否按需求大小调整节点大小
        show_time_windows: 是否显示时间窗信息
        show_customer_id: 是否显示客户编号
        num_customers: 只显示前N个客户 (None=全部)
        figsize: 图片大小
        title: 自定义标题
    """
    # 解析文件
    data = parse_solomon_file(filepath)

    instance_name = data['name']
    depot = data['depot']
    customers = data['customers']

    if num_customers is not None:
        customers = customers[:num_customers]

    print(f"Solomon算例: {instance_name}")
    print(f"仓库位置: ({depot[0]}, {depot[1]})")
    print(f"客户数量: {len(customers)}")
    print(f"车辆数量: {data['vehicle_number']}, 容量: {data['vehicle_capacity']}")

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 提取坐标
    depot_x, depot_y = depot[0], depot[1]
    cust_x = [c[1] for c in customers]
    cust_y = [c[2] for c in customers]
    cust_demands = [c[3] for c in customers]
    cust_ids = [c[0] for c in customers]

    # 计算节点大小
    if show_demand:
        max_demand = max(cust_demands) if cust_demands else 1
        min_size = 50
        max_size = 300
        sizes = [min_size + (d / max_demand) * (max_size - min_size) for d in cust_demands]
    else:
        sizes = [100] * len(customers)

    # 绘制仓库
    ax.scatter([depot_x], [depot_y], c='red', s=400, marker='s',
               zorder=5, edgecolors='black', linewidths=2, label='Depot (0)')
    ax.annotate('D', (depot_x, depot_y), fontsize=14, ha='center', va='center',
                fontweight='bold', color='white')

    # 绘制客户节点
    scatter = ax.scatter(cust_x, cust_y, c='steelblue', s=sizes, marker='o',
                        zorder=4, edgecolors='darkblue', linewidths=1,
                        alpha=0.7, label='Customers')

    # 显示客户编号
    if show_customer_id:
        for i, (x, y, cid) in enumerate(zip(cust_x, cust_y, cust_ids)):
            ax.annotate(str(cid), (x, y), fontsize=8, ha='center', va='center',
                       color='white', fontweight='bold')

    # 显示时间窗信息
    if show_time_windows:
        for c in customers[:min(10, len(customers))]:  # 只显示前10个避免太拥挤
            cid, x, y, demand, ready, due, service = c
            ax.annotate(f'[{int(ready)},{int(due)}]', (x, y + 2),
                       fontsize=6, ha='center', va='bottom', color='gray')

    # 计算坐标范围
    all_x = [depot_x] + cust_x
    all_y = [depot_y] + cust_y
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    padding = max(x_max - x_min, y_max - y_min) * 0.1

    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # 设置标题和标签
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Solomon Instance: {instance_name}\n'
                    f'Customers: {len(customers)}, '
                    f'Vehicles: {data["vehicle_number"]}, '
                    f'Capacity: {data["vehicle_capacity"]}',
                    fontsize=12, fontweight='bold')

    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)

    # 添加图例
    ax.legend(loc='upper right', fontsize=10)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # 添加统计信息文本框
    stats_text = (f"Total Customers: {len(customers)}\n"
                  f"Demand Range: [{int(min(cust_demands))}, {int(max(cust_demands))}]\n"
                  f"Total Demand: {int(sum(cust_demands))}\n"
                  f"Coord Range: X[{x_min:.0f},{x_max:.0f}] Y[{y_min:.0f},{y_max:.0f}]")

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存图片
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        save_dir = os.path.dirname(os.path.abspath(filepath))
        save_path = os.path.join(save_dir, f"{base_name}_nodes.png")

    if not save_path.endswith('.png'):
        save_path += '.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n图片已保存: {save_path}")

    plt.close(fig)
    return save_path


def visualize_with_clusters(
    filepath: str,
    save_path: str = None,
    num_customers: int = None,
    figsize: tuple = (12, 10)
):
    """
    按需求大小分组可视化（用颜色区分）
    """
    data = parse_solomon_file(filepath)

    depot = data['depot']
    customers = data['customers']

    if num_customers is not None:
        customers = customers[:num_customers]

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制仓库
    depot_x, depot_y = depot[0], depot[1]
    ax.scatter([depot_x], [depot_y], c='red', s=400, marker='s',
               zorder=5, edgecolors='black', linewidths=2, label='Depot')
    ax.annotate('D', (depot_x, depot_y), fontsize=14, ha='center', va='center',
                fontweight='bold', color='white')

    # 按需求分组
    demands = [c[3] for c in customers]
    max_demand = max(demands)

    # 分成3组: 小需求、中需求、大需求
    small = [(c[1], c[2], c[0]) for c in customers if c[3] <= max_demand / 3]
    medium = [(c[1], c[2], c[0]) for c in customers if max_demand / 3 < c[3] <= 2 * max_demand / 3]
    large = [(c[1], c[2], c[0]) for c in customers if c[3] > 2 * max_demand / 3]

    # 绘制各组
    if small:
        ax.scatter([p[0] for p in small], [p[1] for p in small],
                  c='lightgreen', s=80, marker='o', zorder=4,
                  edgecolors='green', linewidths=1,
                  label=f'Small Demand (n={len(small)})')
        for x, y, cid in small:
            ax.annotate(str(cid), (x, y), fontsize=7, ha='center', va='center')

    if medium:
        ax.scatter([p[0] for p in medium], [p[1] for p in medium],
                  c='orange', s=120, marker='o', zorder=4,
                  edgecolors='darkorange', linewidths=1,
                  label=f'Medium Demand (n={len(medium)})')
        for x, y, cid in medium:
            ax.annotate(str(cid), (x, y), fontsize=7, ha='center', va='center')

    if large:
        ax.scatter([p[0] for p in large], [p[1] for p in large],
                  c='tomato', s=180, marker='o', zorder=4,
                  edgecolors='darkred', linewidths=1,
                  label=f'Large Demand (n={len(large)})')
        for x, y, cid in large:
            ax.annotate(str(cid), (x, y), fontsize=7, ha='center', va='center')

    ax.set_title(f'Solomon Instance: {data["name"]} - Demand Distribution\n'
                f'Customers: {len(customers)}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Coordinate', fontsize=11)
    ax.set_ylabel('Y Coordinate', fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path is None:
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        save_dir = os.path.dirname(os.path.abspath(filepath))
        save_path = os.path.join(save_dir, f"{base_name}_demand_clusters.png")

    if not save_path.endswith('.png'):
        save_path += '.png'

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"图片已保存: {save_path}")

    plt.close(fig)
    return save_path


# ==================== 主函数 ====================

if __name__ == "__main__":
    import sys

    # 默认文件路径
    solomon_file = "c101.txt"

    # 检查命令行参数
    if len(sys.argv) > 1:
        solomon_file = sys.argv[1]

    # 检查文件是否存在
    if not os.path.exists(solomon_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, solomon_file)
        if os.path.exists(alt_path):
            solomon_file = alt_path
        else:
            print(f"错误: 找不到文件 '{solomon_file}'")
            sys.exit(1)

    print("=" * 60)
    print("Solomon算例节点可视化")
    print("=" * 60)

    # 基本可视化
    print("\n[1] 生成基本节点分布图...")
    visualize_solomon_nodes(
        solomon_file,
        show_demand=True,
        show_customer_id=True,
        show_time_windows=False
    )

    # 按需求分组可视化
    print("\n[2] 生成需求分组图...")
    visualize_with_clusters(solomon_file)

    print("\n完成!")
