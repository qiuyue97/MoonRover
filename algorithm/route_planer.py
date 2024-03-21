import numpy as np
import heapq
import matplotlib.pyplot as plt


def get_neighbors(map_handled, node):
    # 返回节点的所有相邻节点
    i, j = node
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < map_handled.shape[0] and 0 <= ny < map_handled.shape[1]:
            neighbors.append((nx, ny))
    return neighbors


def calculate_distance(map_handled, node1, node2, obstacle_distances):
    # 返回从node1到node2的距离，但要考虑到节点到最近障碍物的距离
    i1, j1 = node1
    i2, j2 = node2
    dx = i1 - i2
    dy = j1 - j2
    dh = (map_handled[i2, j2] - map_handled[i1, j1] + 1) * 10
    dd = obstacle_distances[i2, j2]  # 新增：获取node2到最近障碍物的距离
    return np.sqrt(dh ** 2 + dx ** 2 + dy ** 2)  # 减去一个与dd成比例的值，使得dd更大的节点得到更小的距离


def calculate_obstacle_distances(map_handled):
    # 假设map_handled的值越大，表示越接近障碍物
    threshold = 2  # 设置一个阈值来判断什么算是“障碍物”

    obstacle_distances = np.full(map_handled.shape, np.inf)  # 初始化距离数组
    queue = []

    # 找到所有的障碍物，并设置它们的距离为0
    for i in range(map_handled.shape[0]):
        for j in range(map_handled.shape[1]):
            if map_handled[i, j] > threshold:
                queue.append((i, j))
                obstacle_distances[i, j] = 0

    # 广度优先搜索
    while queue:
        current_node = queue.pop(0)  # 取出一个节点
        i, j = current_node
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 遍历该节点的四个相邻节点
            ni, nj = i + di, j + dj
            if 0 <= ni < map_handled.shape[0] and 0 <= nj < map_handled.shape[1]:  # 检查边界
                new_distance = obstacle_distances[i, j] + 1
                if new_distance < obstacle_distances[ni, nj]:  # 如果新的距离比现有的距离小，就更新距离
                    obstacle_distances[ni, nj] = new_distance
                    queue.append((ni, nj))  # 将这个节点添加到队列中
    return obstacle_distances


def dijkstra(map_handled, start, end):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {(i, j): float('infinity') for i in range(map_handled.shape[0]) for j in range(map_handled.shape[1])}
    distances[start] = 0
    path = {}
    visited = set()  # 创建一个集合来保存已经访问过的节点
    obstacle_distances = calculate_obstacle_distances(map_handled)
    total_nodes = map_handled.shape[0] * map_handled.shape[1]  # 计算总的节点数量
    last_printed_progress = -2.0  # 记录上一次打印的进度
    print("正在寻路...")
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_node == end:  # 添加了检查当前节点是否为结束节点的语句
            break
        if current_node in visited:  # 如果这个节点已经被访问过，就跳过
            continue
        visited.add(current_node)  # 标记这个节点已经被访问过
        progress = len(visited) / total_nodes * 100  # 计算进度
        if progress - last_printed_progress >= 10.0:  # 如果进度增加了至少10%
            print(f"已完成寻路进度：{progress:.2f}%")  # 打印进度，保留两位小数
            last_printed_progress = progress  # 更新上一次打印的进度
        for neighbor in get_neighbors(map_handled, current_node):
            distance = current_distance + calculate_distance(map_handled, current_node, neighbor, obstacle_distances)
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                path[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))
    # 使用路径信息重建最短路径
    current_node = end
    shortest_path = []
    while current_node != start:
        shortest_path.append(current_node)
        current_node = path[current_node]
    shortest_path.append(start)
    shortest_path.reverse()
    print("已得到最优路径。")
    return shortest_path


def save_visualized_path(map_handled, shortest_path):
    plt.figure(figsize=(10, 10))

    # 创建一个颜色图，显示地图的高度
    plt.imshow(map_handled, cmap='terrain')

    # 提取路径中的x和y坐标
    path_x = [node[0] for node in shortest_path]
    path_y = [node[1] for node in shortest_path]
    start = shortest_path[0]
    end = shortest_path[-1]

    # 在地图上标记出路径
    plt.plot(path_y, path_x, color='red')  # 注意我们这里交换了x和y，因为matplotlib的imshow函数和numpy数组的索引不同

    # 标记出起点和终点
    start_x, start_y = start[::-1]
    start_x = start_x / 10 - 25
    start_y = 25 - start_y / 10
    plt.scatter(*start[::-1], color='green')
    plt.scatter(*end[::-1], color='blue')

    # 显示/储存图像
    # plt.show()
    plt.savefig(f'./algorithm/temp/img/route_plan/path_from_{start_x:.1f}_{start_y:.1f}.png')


def path_coords_shift(shortest_path):
    new_path = []
    for node in shortest_path:
        x = node[1] / 10 - 25
        y = 25 - node[0] / 10
        new_node = (x, y)
        new_path.append(new_node)
    return new_path


def Dijkstra_main(car_status, area_B):
    map_handled = np.load('./algorithm/temp/map_handled.npy')
    # map_handled = np.load('./temp/map_handled.npy') # 测试用
    print('路径规划算法已加载地图。')
    start = (int(250 - car_status[1][1] * 10), int(250 + car_status[1][0] * 10))
    end = (int(250 - area_B[:2][1] * 10), int(250 + area_B[:2][0] * 10))
    # 对你的地图调用dijkstra函数
    shortest_path = dijkstra(map_handled, start, end)
    save_visualized_path(map_handled, shortest_path)
    return path_coords_shift(shortest_path)


def action_judge(rest_path, car_status, scaling_ratio=1):
    actions = []
    if len(rest_path) > 0:
        now_pos = car_status[1]
        end_node = np.array(rest_path[-1]).astype(np.float64)
        if len(rest_path) > scaling_ratio:
            next_node = np.array(rest_path[scaling_ratio]).astype(np.float64)
        else:
            next_node = end_node
        target_distance = np.sqrt(np.sum((next_node - now_pos)**2))
        if target_distance < 1 / 11:
            return rest_path[scaling_ratio:], actions
        # 计算从now_pos指向next_node的向量
        vector = next_node - now_pos
        target_direction = - np.arctan2(vector[0], vector[1])
        # 计算转向和直行的次数
        direction_diff = np.arccos(np.cos(car_status[0]) * np.cos(target_direction) + np.sin(car_status[0]) * np.sin(target_direction))
        turn_times = int(np.ceil(direction_diff / np.radians(18.95)))
        action = 'left' if car_status[0] < target_direction else 'right'
        for _ in range(turn_times):
            actions.append(action)
        move_times = int(np.ceil(target_distance * 11))
        actions.extend(['ahead'] * move_times)
        return rest_path[scaling_ratio:], actions
