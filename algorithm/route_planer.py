import numpy as np
import heapq
import matplotlib.pyplot as plt

def moon_rover_simulate(actions, pos_A):
    for now_step in range(1, len(actions) + 2):
        if now_step == 1:
            car_status = np.array([0, np.array(pos_A).astype(np.float64)], dtype=object)
            np.save('./temp/car_status_simulate', car_status)
        else:
            car_status = np.load('./temp/car_status_simulate.npy', allow_pickle=True)
        if now_step <= len(actions):
            if actions[int(now_step - 1)] == 'left':
                car_status[0] += np.radians(18.95)
                np.save('./temp/car_status_simulate', car_status)
                print(f'预计需要{len(actions)}步，现在是第{now_step}步，执行左转命令，现在小车与Y轴夹角为{np.degrees(car_status[0]):.2f}度，'
                      f'现在小车位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
            elif actions[int(now_step - 1)] == 'right':
                car_status[0] -= np.radians(18.95)
                np.save('./temp/car_status_simulate', car_status)
                print(f'预计需要{len(actions)}步，现在是第{now_step}步，执行右转命令，现在小车与Y轴夹角为{np.degrees(car_status[0]):.2f}度，'
                      f'现在小车位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
            elif actions[int(now_step - 1)] == 'ahead':
                car_status[1] += (1 / 11) * np.array([-np.sin(car_status[0]), np.cos(car_status[0])])
                np.save('./temp/car_status_simulate', car_status)
                print(f'预计需要{len(actions)}步，现在是第{now_step}步，执行直行命令，现在小车与Y轴夹角为{np.degrees(car_status[0]):.2f}度，'
                      f'现在小车位于({car_status[1][0]:.4f}, {car_status[1][1]:.4f})。')
        else:
            print('done!')

# 根据你的需求，定义一个函数来获取节点的相邻节点
def get_neighbors(map_handled, node):
    # 返回节点的所有相邻节点
    i, j = node
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = i + dx, j + dy
        if 0 <= nx < map_handled.shape[0] and 0 <= ny < map_handled.shape[1]:
            neighbors.append((nx, ny))
    return neighbors

# 定义一个函数来计算从一个节点到另一个节点的距离（权重）
# def calculate_distance(map_handled, node1, node2, obstacle_distances):
#     # 返回从node1到node2的距离
#     i1, j1 = node1
#     i2, j2 = node2
#     dx = i2 - i1
#     dy = j2 - j1
#     dh = (map_handled[i2, j2] - map_handled[i1, j1] + 1) * 10
#     dd = obstacle_distances[i2, j2]  # 新增：获取node2到最近障碍物的距离
#     return dh ** 6 - dd * 10

# def calculate_obstacle_distances(map_handled, start):
#     # 获取start点的值
#     start_value = map_handled[start[0], start[1]]
#     threshold = 0.3
#     # 初始化距离数组
#     obstacle_distances = np.full(map_handled.shape, np.inf)
#     queue = []
#     # 找到所有的障碍物，并设置它们的距离为0
#     for i in range(map_handled.shape[0]):
#         for j in range(map_handled.shape[1]):
#             if abs(map_handled[i, j] - start_value) > threshold:
#                 queue.append((i, j))
#                 obstacle_distances[i, j] = 0
#     # 广度优先搜索
#     while queue:
#         current_node = queue.pop(0)  # 取出一个节点
#         i, j = current_node
#         for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 遍历该节点的四个相邻节点
#             ni, nj = i + di, j + dj
#             if 0 <= ni < map_handled.shape[0] and 0 <= nj < map_handled.shape[1]:  # 检查边界
#                 new_distance = obstacle_distances[i, j] + 1 + np.random.rand() * 0.001
#                 if new_distance < obstacle_distances[ni, nj]:  # 如果新的距离比现有的距离小，就更新距离
#                     obstacle_distances[ni, nj] = new_distance
#                     queue.append((ni, nj))  # 将这个节点添加到队列中
#     return obstacle_distances

# Dijkstra算法的实现

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
    return distances, shortest_path

def create_action_list(shortest_path, current_direction, scaling_ratio=1):
    print("正在生成行动表...")
    actions = []
    now_pos = np.array(shortest_path[0]).astype(np.float64)
    times = int(len(shortest_path) / scaling_ratio)
    end_node = np.array(shortest_path[-1]).astype(np.float64)
    for i in range(1, times):
        # get current node and next node
        if i != times - 1:
            next_node = np.array(shortest_path[i * scaling_ratio]).astype(np.float64)
        else:
            next_node = end_node
        target_distance = np.sqrt(np.sum((next_node - now_pos)**2))
        if target_distance < 1 / 11:
            continue
        # 计算从now_pos指向next_node的向量
        vector = next_node - now_pos
        target_direction = - np.arctan2(vector[0], vector[1])
        # adjust current_direction to target_direction
        while np.arccos(np.cos(current_direction) * np.cos(target_direction) + np.sin(current_direction) * np.sin(target_direction)) > np.radians(18.95):
            if current_direction < target_direction:
                current_direction += np.radians(18.95)
                actions.append('left')
            else:
                current_direction -= np.radians(18.95)
                actions.append('right')
        # move forward according to the target_distance
        while round(target_distance, 10) >= 1 / 11:
            target_distance -= 1 / 11
            now_pos += (1 / 11) * np.array([-np.sin(current_direction), np.cos(current_direction)])
            actions.append('ahead')
    print(f'行动表已生成，还需{len(actions)}步到达目标区域中心点。')
    return actions

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

def save_visualized_actions(map_handled, actions, start, current_direction):
    plt.figure(figsize=(10,10))
    plt.imshow(map_handled, cmap='terrain')

    # 设置起始位置和方向
    current_position = np.array((start[1], start[0]), dtype=float)
    current_direction += np.radians(90)  # Assume we start heading to the up

    for action in actions:
        if action == 'ahead':
            # Calculate the displacement in x and y direction
            dx = np.cos(current_direction) * 10 / 11
            dy = - np.sin(current_direction) * 10 / 11
            color = 'blue'
        elif action == 'left':
            current_direction += np.radians(18.95)  # Turn left
            dx = dy = 0
            color = 'red'
        elif action == 'right':
            current_direction -= np.radians(18.95)  # Turn right
            dx = dy = 0
            color = 'red'

        # Update current_position
        displacement = np.array([dx, dy])
        new_position = current_position + displacement

        # Plot the arrow for this step
        if dx != 0 or dy != 0:  # Only draw an arrow if there's displacement
            plt.arrow(current_position[0], current_position[1], dx, dy, head_width=1.5, head_length=2, fc=color, ec=color)
        else:  # If no displacement, draw a point
            plt.plot(current_position[0], current_position[1], 'o', color=color)
        current_position = new_position

    # Plot start and end points
    plt.plot(start[1], start[0], 'go')  # Start point in green
    plt.plot(current_position[1], current_position[0], 'ro')  # End point in red

    # Show/save image
    start_x = start[1] / 10 - 25
    start_y = 25 - start[0] / 10
    # plt.show()
    plt.savefig(f'./algorithm/temp/img/route_plan/actions_from_{start_x:.1f}_{start_y:.1f}.png')

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
    distances, shortest_path = dijkstra(map_handled, start, end)
    save_visualized_path(map_handled, shortest_path)
    actions = create_action_list(path_coords_shift(shortest_path), car_status[0], scaling_ratio=25)
    save_visualized_actions(map_handled, actions, start, car_status[0])
    # moon_rover_simulate(actions, pos_A)
    return actions

def Baka_main(car_status, area_B):
    center_B = area_B[:2]
    dx = center_B[0] - car_status[1][0]
    dy = center_B[1] - car_status[1][1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    walk_steps = int(np.ceil(distance / (1 / 11)))
    target_angle = np.degrees(np.arctan2(dy, dx))
    # 计算小车需要转动的次数和方向
    turn_angle = 18.95  # 一次转动18.95°
    diff_angle = target_angle - (90 + np.degrees(car_status[0]))
    diff_angle = (diff_angle + 180) % 360 - 180  # 将角度标准化到[-180, 180]区间
    if diff_angle < 0:
        direction = "right"  # 如果差角小于0，那么应该向右转
        diff_angle = np.abs(diff_angle)
    else:
        direction = "left"  # 如果差角大于等于0，那么应该向左转
    turns = int(diff_angle / turn_angle)
    remainder = diff_angle % turn_angle
    if remainder > turn_angle / 2:
        turns += 1  # 如果剩余角度大于转动角度的一半，那么应该多转一次
    actions = []
    for i in range(turns):
        actions.append(direction)
    for i in range(walk_steps):
        actions.append('ahead')
    return actions

# Dijkstra_main([0, [10, -15]], [-15, 15, 10, 10])