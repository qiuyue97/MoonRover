import numpy as np
import heapq
import matplotlib.pyplot as plt

moon_map = np.load('../../map.npy')
np.set_printoptions(suppress=True)
map_handled = moon_map[:, :, 3]
threshold = 2.5
binary_map = np.where(map_handled > threshold, map_handled.max(), map_handled.min())


def Dijkstra_main(map_handled, car_status=[0, [10, -15]], area_B=[-15, 15, 10, 10]):
    print('路径规划算法已加载地图。')
    start = (int(250 - car_status[1][1] * 10), int(250 + car_status[1][0] * 10))
    end = (int(250 - area_B[:2][1] * 10), int(250 + area_B[:2][0] * 10))
    # 对你的地图调用dijkstra函数
    _, shortest_path = dijkstra(map_handled, start, end)
    return shortest_path


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
    return distances, shortest_path


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


def get_neighbors(map_handled, node):
    # 返回节点的所有相邻节点
    i, j = node
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < map_handled.shape[0] and 0 <= ny < map_handled.shape[1]:
            neighbors.append((nx, ny))
    return neighbors


def fig_printer(map_handled, binary_map, shortest_path_full, shortest_path_bin):

    # 提取路径中的x和y坐标
    path_x = [node[0] for node in shortest_path_full]
    path_y = [node[1] for node in shortest_path_full]
    path_x_bin = [node[0] for node in shortest_path_bin]
    path_y_bin = [node[1] for node in shortest_path_bin]
    start = shortest_path_full[0]
    end = shortest_path_full[-1]

    plt.figure(figsize=(20, 10))
    # 创建一个颜色图，显示二元地图
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(binary_map, cmap='gray_r')
    ax1.axis('on')
    ax1.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax1.set_xticks(np.arange(-0.5, 500, 25), minor=True)
    ax1.set_yticks(np.arange(-0.5, 500, 25), minor=True)
    ax1.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax1.plot(path_y_bin, path_x_bin, color='red')
    ax1.scatter(*start[::-1], color='green')
    ax1.scatter(*end[::-1], color='blue')

    # 创建一个颜色图，显示地图的高度
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(map_handled, cmap='viridis')
    ax2.axis('on')
    ax2.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    ax2.set_xticks(np.arange(-0.5, 500, 25), minor=True)
    ax2.set_yticks(np.arange(-0.5, 500, 25), minor=True)
    ax2.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax2.plot(path_y, path_x, color='red')
    ax2.scatter(*start[::-1], color='green')
    ax2.scatter(*end[::-1], color='blue')

    plt.show()


def Astar_main(binary_map, car_status=[0, [10, -15]], area_B=[-15, 15, 10, 10]):
    print('A* 路径规划算法已加载地图。')
    start = (int(250 - car_status[1][1] * 10), int(250 + car_status[1][0] * 10))
    end = (int(250 - area_B[:2][1] * 10), int(250 + area_B[:2][0] * 10))
    _, shortest_path = astar(binary_map, start, end)
    return shortest_path


def get_neighbors_astar(map_handled, node, binary_map):
    # 返回节点的所有非障碍物相邻节点，专为A*算法设计
    i, j = node
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < binary_map.shape[0] and 0 <= ny < binary_map.shape[1] and binary_map[nx, ny] == binary_map.min():
            neighbors.append((nx, ny))
    return neighbors


def heuristic(node1, node2):
    # 使用欧几里得距离作为启发函数
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def astar(binary_map, start, end):
    queue = []
    heapq.heappush(queue,
                   (0 + heuristic(start, end), 0, start))  # (estimated_total_cost, current_path_cost, current_node)
    path = {}
    distances = {start: 0}
    visited = set()

    total_nodes = binary_map.size  # 获取总节点数，用于计算进度
    last_printed_progress = -1  # 初始化最后打印的进度
    print("正在寻路...")

    while queue:
        estimated_total_cost, current_distance, current_node = heapq.heappop(queue)
        if current_node in visited:
            continue

        visited.add(current_node)

        # 打印进度更新
        progress = len(visited) / total_nodes * 100  # 计算当前进度百分比
        if progress - last_printed_progress >= 10:  # 每当进度增加至少10%，打印一次进度
            print(f"寻路进度: {progress:.2f}%")
            last_printed_progress = progress

        if current_node == end:
            break

        for neighbor in get_neighbors_astar(binary_map, current_node, binary_map):
            if neighbor in visited:
                continue

            tentative_g_score = distances[current_node] + calculate_distance_astar(binary_map, current_node, neighbor)
            if tentative_g_score < distances.get(neighbor, float('infinity')):
                path[neighbor] = current_node
                distances[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, end)
                heapq.heappush(queue, (f_score, tentative_g_score, neighbor))

    # 重建路径
    current_node = end
    shortest_path = []
    while current_node != start:
        shortest_path.append(current_node)
        current_node = path.get(current_node)
        if current_node is None:  # 如果找不到路径
            return float('infinity'), []
    shortest_path.append(start)
    shortest_path.reverse()
    print("已得到最优路径。")
    return distances[end], shortest_path


def calculate_distance_astar(map_handled, node1, node2):
    # Simplified distance calculation for A*; ignores height difference for this example
    return np.linalg.norm(np.array(node1) - np.array(node2))


shortest_path_full = Dijkstra_main(map_handled)
shortest_path_bin = Astar_main(binary_map)
fig_printer(map_handled, binary_map, shortest_path_full, shortest_path_bin)