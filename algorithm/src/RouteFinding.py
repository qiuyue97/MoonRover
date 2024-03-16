import os
import numpy as np
import heapq
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import sys


folder_path = '../../mapset/'


def map_handler(path, obs_threshold=2, pun_threshold=20):
    map_set = []
    for file in os.listdir(path):
        if file.endswith('.npy'):
            file_path = os.path.join(path, file)
            map_data = np.load(file_path)
            map_set.append(map_data)

    high_map = []
    tri_map = []
    for map in map_set:
        high_map.append(map[:, :, 3])
        # 定义障碍物
        obstacles = np.where(map[:, :, 3] > obs_threshold, True, False)
        # 计算到最近非障碍物点的距离
        distances = distance_transform_edt(~obstacles)
        # 定义惩罚点为距离小于惩罚阈值的点
        pun = (distances > 0) & (distances < pun_threshold)
        # 创建三值地图，障碍物点为2，危险点为1，其他为0
        tri_map_cur = np.zeros_like(map[:, :, 3], dtype=int)
        tri_map_cur[obstacles] = 2  # 障碍物点
        tri_map_cur[pun] = 1  # 危险点
        tri_map.append(tri_map_cur)

    return high_map, tri_map


class AStar:
    def __init__(self, elevation_map, tri_map, start, goal):
        self.elevation_map = elevation_map / 1.417379975200399 * 10
        self.tri_map = tri_map
        self.start = (int(250 - start[1] * 10), int(250 + start[0] * 10))
        self.goal = (int(250 - goal[1] * 10), int(250 + goal[0] * 10))
        self.nodes_expanded = 0  # 记录扩展的节点数
        self.total_nodes = np.product(tri_map.shape)  # 计算地图上的总节点数

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


if __name__ == '__main__':
    # 假设map_handler已正确返回高程地图和三值地图
    high_map, tri_map = map_handler(folder_path)
    elevation_map = high_map[0]
    tri_map = tri_map[0]
    start = [10, -15]  # 起始坐标
    goal = [-15, 15]   # 目标坐标

    # AStar
    print("开始查找：AStar")
    a_star = AStar(elevation_map, tri_map, start, goal)
    path_a_star = a_star.find_path()

    # Dijkstra
    print("\n开始查找：Dijkstra")
    dijkstra = Dijkstra(elevation_map, tri_map, start, goal)
    path_dijkstra = dijkstra.find_path()

    # BFS
    print("\n开始查找：BFS")
    bfs = BFS(elevation_map, tri_map, start, goal)
    path_bfs = bfs.find_path()

    # AStarWithH
    print("\n开始查找：AStarWithH")
    a_star_with_h = AStarWithH(elevation_map, tri_map, start, goal, delta=20.0, lambda_val=2.0)
    path_a_star_with_h = a_star_with_h.find_path()

    # AStarWithPun
    print("\n开始查找：AStarWithPun")
    a_star_with_pun = AStarWithPun(elevation_map, tri_map, start, goal, delta=20.0, lambda_val=2.0)
    path_a_star_with_pun = a_star_with_pun.find_path()

    # 绘制第一张图（AStar, Dijkstra, BFS）
    plt.figure(figsize=(10, 10))
    plt.imshow(elevation_map, cmap='terrain')
    plt.plot([p[1] for p in path_a_star], [p[0] for p in path_a_star], color='red', label='AStar')
    plt.plot([p[1] for p in path_dijkstra], [p[0] for p in path_dijkstra], color='green', label='Dijkstra')
    plt.plot([p[1] for p in path_bfs], [p[0] for p in path_bfs], color='blue', label='BFS')
    plt.scatter([a_star.start[1], a_star.goal[1]], [a_star.start[0], a_star.goal[0]], color=['black', 'yellow'])
    plt.legend()
    plt.title('Paths using AStar, Dijkstra, and BFS')
    plt.show()

    # 绘制第二张图（AStarWithH, AStarWithPun）
    plt.figure(figsize=(10, 10))
    plt.imshow(elevation_map, cmap='terrain')
    plt.plot([p[1] for p in path_a_star_with_h], [p[0] for p in path_a_star_with_h], color='magenta', label='AStarWithH')
    plt.plot([p[1] for p in path_a_star_with_pun], [p[0] for p in path_a_star_with_pun], color='cyan', label='AStarWithPun')
    plt.scatter([a_star.start[1], a_star.goal[1]], [a_star.start[0], a_star.goal[0]], color=['black', 'yellow'])
    plt.legend()
    plt.title('Paths using AStarWithH and AStarWithPun')
    plt.show()
