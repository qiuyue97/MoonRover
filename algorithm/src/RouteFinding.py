import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import distance_transform_edt
import sys
import time
import json


folder_path = '../../mapset/'
file_dicts = [
    {
        "file": 'map.npy',
        "start": [10, -15],
        "goal": [-15, 15],
    },
    {
        "file": 'TestMap-3.npy',
        "start": [11, -14],
        "goal": [15, 15],
    },
    {
        "file": 'TestMap4.npy',
        "start": [-11, 18],
        "goal": [15, -15],
    },
    {
        "file": 'TestMap5.npy',
        "start": [-15, -20],
        "goal": [0, 10],
    },
]
json_path = 'results.json'


def map_handler(path, file_dicts, pun_threshold=20):
    map_set = []
    tri_map_set = []
    map_infos = []
    for file_dict in file_dicts:
        file_path = os.path.join(path, file_dict["file"])
        if os.path.exists(file_path):
            map_data = np.load(file_path)  # 加载地图数据
            start = file_dict["start"]
            goal = file_dict["goal"]
            map_set.append(map_data[:, :, 3])
            max_elevation = np.max(map_data[:, :, 3])
            min_elevation = np.min(map_data[:, :, 3])
            avg_elevation = np.mean(map_data[:, :, 3])
            if file_dict["start"] is not None and file_dict["goal"] is not None:
                np_start = (int(250 - start[1] * 10), int(250 + start[0] * 10))
                np_goal = (int(250 - goal[1] * 10), int(250 + goal[0] * 10))
                start_elevation = map_data[np_start[0], np_start[1], 3]
                goal_elevation = map_data[np_goal[0], np_goal[1], 3]
            else:
                np_start = None
                np_goal = None
                start_elevation = None
                goal_elevation = None
            map_info = {
                "file": file_dict["file"],
                "max_elevation": max_elevation,
                "min_elevation": min_elevation,
                "avg_elevation": avg_elevation,
                "start": start,
                "np_start": np_start,
                "start_elevation": start_elevation,
                "goal": goal,
                "np_goal": np_goal,
                "goal_elevation": goal_elevation,
                "obs_threshold": max(start_elevation, goal_elevation) + 1.3
            }
            map_infos.append(map_info)
            # 定义障碍物
            obstacles = np.where(map_data[:, :, 3] > map_info["obs_threshold"], True, False)
            # 计算到最近非障碍物点的距离
            distances = distance_transform_edt(~obstacles)
            # 定义惩罚点为距离小于惩罚阈值的点
            pun = (distances > 0) & (distances < pun_threshold)
            # 创建三值地图，障碍物点为2，危险点为1，其他为0
            tri_map = np.zeros_like(map_data[:, :, 3], dtype=int)
            tri_map[obstacles] = 2  # 障碍物点
            tri_map[pun] = 1  # 危险点
            tri_map_set.append(tri_map)
    return map_set, tri_map_set, map_infos


class AStar:
    def __init__(self, elevation_map, tri_map, start, goal):
        self.elevation_map = elevation_map / 1.417379975200399 * 10
        self.tri_map = tri_map
        self.start = start
        self.goal = goal
        self.nodes_expanded = 0  # 记录扩展的节点数
        self.total_nodes = np.product(tri_map.shape)  # 计算地图上的总节点数
        self.execution_time = None
        self.max_elevation_change = None
        self.total_elevation_change = None

    def update_progress(self):
        """更新并显示搜索进度，使用回车符覆盖之前的输出"""
        progress_percentage = self.nodes_expanded / self.total_nodes * 100
        sys.stdout.write(f"\r当前搜索进度: {progress_percentage:.2f}%")
        sys.stdout.flush()

    def g(self, current, neighbor):
        """从起点到当前节点的实际成本"""
        return np.sqrt((current[0] - neighbor[0]) ** 2 + (current[1] - neighbor[1]) ** 2)

    def h(self, current, goal):
        """从当前节点到终点的估计成本"""
        return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)

    def get_neighbors(self, node):
        """获取节点的邻居节点，只包括非障碍物点"""
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        neighbors = []
        for direction in directions:
            neighbor = (node[0] + direction[0], node[1] + direction[1])
            if 0 <= neighbor[0] < self.tri_map.shape[0] and 0 <= neighbor[1] < self.tri_map.shape[1]:
                if self.tri_map[neighbor[0], neighbor[1]] != 2:  # 排除障碍物点
                    neighbors.append(neighbor)
        return neighbors

    def find_path(self):
        """使用A*算法找到从起点到终点的路径"""
        open_set = []
        heapq.heappush(open_set, (0 + self.h(self.start, self.goal), 0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.h(self.start, self.goal)}

        while open_set:
            current = heapq.heappop(open_set)[2]

            # 更新并显示进度
            self.update_progress()

            if current == self.goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.g(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.h(neighbor, self.goal)
                    if neighbor not in [i[2] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], self.nodes_expanded, neighbor))
                        self.nodes_expanded += 1

        return None  # 如果没有找到路径，则返回None

    def reconstruct_path(self, came_from, current):
        """重建路径"""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.insert(0, current)
        return total_path

    def calculate_elevation_changes(self, path):
        max_elevation_change = 0
        total_elevation_change = 0
        for i in range(len(path) - 1):
            elevation_diff = abs(
                self.elevation_map[path[i + 1][0], path[i + 1][1]] - self.elevation_map[path[i][0], path[i][1]])
            total_elevation_change += elevation_diff
            if elevation_diff > max_elevation_change:
                max_elevation_change = elevation_diff
        self.max_elevation_change = max_elevation_change
        self.total_elevation_change = total_elevation_change


class Dijkstra(AStar):
    def h(self, current, goal):
        return 0


class BFS(AStar):
    def g(self, current, neighbor):
        return 0


class AStarWithH(AStar):
    def __init__(self, elevation_map, tri_map, start, goal, delta=20.0, lambda_val=2.0):
        super().__init__(elevation_map, tri_map, start, goal)
        self.delta = delta
        self.lambda_val = lambda_val

    def g(self, current, neighbor):
        """从起点到当前节点的实际成本"""
        h_current = self.elevation_map[current[0], current[1]]
        h_neighbor = self.elevation_map[neighbor[0], neighbor[1]]
        d_height = (h_neighbor - h_current) ** self.lambda_val
        return np.sqrt((current[0] - neighbor[0]) ** 2 + (current[1] - neighbor[1]) ** 2 + self.delta * d_height)

    def h(self, current, goal):
        """从当前节点到终点的估计成本"""
        d_flat = (goal[0] - current[0])**2 + (goal[1] - current[1])**2
        h_current = self.elevation_map[current[0], current[1]]
        h_goal = self.elevation_map[goal[0], goal[1]]
        d_height = (h_goal - h_current) ** self.lambda_val
        return np.sqrt(d_flat + self.delta * d_height)


class AStarWithPun(AStarWithH):
    def __init__(self, elevation_map, tri_map, start, goal, delta=20.0, lambda_val=2.0):
        super().__init__(elevation_map, tri_map, start, goal, delta, lambda_val)
        self.h_bar = np.mean(self.elevation_map)
        self.grad_x, self.grad_y = np.gradient(elevation_map)

    def o(self, current):
        """计算障碍物距离成本函数"""
        h_current = self.elevation_map[current[0], current[1]]
        grad_magnitude = np.sqrt(self.grad_x[current[0], current[1]]**2 + self.grad_y[current[0], current[1]]**2)
        max_grad_magnitude = np.max(np.sqrt(self.grad_x**2 + self.grad_y**2))
        return (h_current / self.h_bar) * max_grad_magnitude * grad_magnitude * 3000

    def find_path(self):
        """使用A*算法找到从起点到终点的路径"""
        open_set = []
        heapq.heappush(open_set, (0 + self.h(self.start, self.goal), 0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.h(self.start, self.goal)}

        while open_set:
            current = heapq.heappop(open_set)[2]

            # 更新并显示进度
            self.update_progress()

            if current == self.goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + self.g(current, neighbor)
                o_score = self.o(neighbor) if self.tri_map[neighbor[0], neighbor[1]] == 1 else 0

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.h(neighbor, self.goal) + o_score
                    if neighbor not in [i[2] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], self.nodes_expanded, neighbor))
                        self.nodes_expanded += 1

        return None  # 如果没有找到路径，则返回None


def plot_paths(elevation_map, tri_map, paths_infos, np_start, np_goal):
    """
    绘制和比较路径。
    """
    colors = ['red', 'green', 'blue', 'magenta', 'cyan']
    paths_infos_group1 = []
    paths_infos_group2 = []
    i = 0
    for path, label in paths_infos:
        if path is None:
            i += 1
            continue
        if label in ["AStar", "Dijkstra", "BFS"]:
            paths_infos_group1.append((path, colors[i], label))
        else:
            paths_infos_group2.append((path, colors[i], label))
        i += 1
    plt.figure(figsize=(20, 20))

    ax1 = plt.subplot(2, 2, 1)
    cmap1 = ListedColormap(['white', 'black'])
    bounds1 = [-0.5, 1.5, 2.5]
    norm1 = BoundaryNorm(bounds1, cmap1.N)
    ax1.imshow(tri_map, cmap=cmap1, norm=norm1)
    ax1.axis('on')
    ax1.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax1.set_xticks(np.arange(-0.5, 500, 25), minor=True)
    ax1.set_yticks(np.arange(-0.5, 500, 25), minor=True)
    ax1.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    for path, color, label in paths_infos_group1:
        ax1.plot([p[1] for p in path], [p[0] for p in path], color=color, label=label)
    ax1.scatter(np_start[1], np_start[0], color=['black'], zorder=5, label='Start')
    ax1.scatter(np_goal[1], np_goal[0], color=['yellow'], zorder=5, label='Goal')
    ax1.legend()
    ax1.set_title('Paths using AStar, Dijkstra, and BFS')

    ax2 = plt.subplot(2, 2, 2)
    cmap2 = ListedColormap(['white', 'gray', 'black'])
    bounds2 = [-0.5, 0.5, 1.5, 2.5]
    norm2 = BoundaryNorm(bounds2, cmap2.N)
    ax2.imshow(tri_map, cmap=cmap2, norm=norm2)
    ax2.axis('on')
    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax2.set_xticks(np.arange(-0.5, 500, 25), minor=True)
    ax2.set_yticks(np.arange(-0.5, 500, 25), minor=True)
    ax2.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    for path, color, label in paths_infos_group2:
        ax2.plot([p[1] for p in path], [p[0] for p in path], color=color, label=label)
    ax2.scatter(np_start[1], np_start[0], color=['black'], zorder=5, label='Start')
    ax2.scatter(np_goal[1], np_goal[0], color=['yellow'], zorder=5, label='Goal')
    ax2.legend()
    ax2.set_title('Paths using AStarWithH and AStarWithPun')

    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(elevation_map, cmap='viridis')
    ax3.axis('on')
    ax3.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax3.set_xticks(np.arange(-0.5, 500, 25), minor=True)
    ax3.set_yticks(np.arange(-0.5, 500, 25), minor=True)
    ax3.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    for path, color, label in paths_infos_group1:
        ax3.plot([p[1] for p in path], [p[0] for p in path], color=color, label=label)
    ax3.scatter(np_start[1], np_start[0], color=['black'], zorder=5, label='Start')
    ax3.scatter(np_goal[1], np_goal[0], color=['yellow'], zorder=5, label='Goal')
    ax3.legend()
    ax3.set_title('Paths using AStar, Dijkstra, and BFS')

    ax4 = plt.subplot(2, 2, 4)
    ax4.imshow(elevation_map, cmap='viridis')
    ax4.axis('on')
    ax4.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax4.set_xticks(np.arange(-0.5, 500, 25), minor=True)
    ax4.set_yticks(np.arange(-0.5, 500, 25), minor=True)
    ax4.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    for path, color, label in paths_infos_group2:
        ax4.plot([p[1] for p in path], [p[0] for p in path], color=color, label=label)
    ax4.scatter(np_start[1], np_start[0], color=['black'], zorder=5, label='Start')
    ax4.scatter(np_goal[1], np_goal[0], color=['yellow'], zorder=5, label='Goal')
    ax4.legend()
    ax4.set_title('Paths using AStarWithH and AStarWithPun')

    plt.savefig(f'./fig/map-{map_id}.png')
    # plt.show()


if __name__ == '__main__':
    ana_dict = []
    map_set, tri_map_set, map_infos = map_handler(folder_path, file_dicts, pun_threshold=20)
    map_id = 0
    for elevation_map, tri_map, map_info in zip(map_set, tri_map_set, map_infos):
        start = map_info["np_start"]
        goal = map_info["np_goal"]
        algorithms = {
            "AStar": AStar(elevation_map, tri_map, start, goal),
            "Dijkstra": Dijkstra(elevation_map, tri_map, start, goal),
            "BFS": BFS(elevation_map, tri_map, start, goal),
            "AStarWithH": AStarWithH(elevation_map, tri_map, start, goal, delta=20.0, lambda_val=2.0),
            "AStarWithPun": AStarWithPun(elevation_map, tri_map, start, goal, delta=20.0, lambda_val=2.0),
        }
        path_infos = []
        for name, algorithm in algorithms.items():
            print(f"\n当前地图ID：{map_id}\n开始查找：{name}")
            start_time = time.time()
            path = algorithm.find_path()
            end_time = time.time()
            execution_time = end_time - start_time
            if path is not None:
                algorithm.calculate_elevation_changes(path)
                print(f"\n路径规划长度: {len(path)}")
                print(f"计算时间: {execution_time:.4f}秒")
                print(f"扩展节点数: {algorithm.nodes_expanded}")
                print(f"路径最大高程差: {algorithm.max_elevation_change}")
                print(f"路径累计高程差: {algorithm.total_elevation_change}")
                ana_dict.append(
                    {
                        "地图编号": map_id,
                        "算法名称": name,
                        "路径规划长度": len(path),
                        "计算时间": execution_time,
                        "扩展节点数": algorithm.nodes_expanded,
                        "路径最大高程差": algorithm.max_elevation_change,
                        "路径累计高程差": algorithm.total_elevation_change,
                    }
                )
            else:
                print("\n未找到路径")
            path_infos.append((path, name))
        plot_paths(elevation_map, tri_map, path_infos, start, goal)
        map_id += 1
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(ana_dict, file, ensure_ascii=False, indent=4)
