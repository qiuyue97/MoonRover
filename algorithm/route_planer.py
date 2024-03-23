import numpy as np
import matplotlib.pyplot as plt
from algorithm.src.RouteFinding import AStar, Dijkstra, BFS, AStarWithH, AStarWithPun
from scipy.ndimage import distance_transform_edt


def tri_maper(map_handled, start, goal, pun_threshold=20):
    np_start = (int(250 - start[1] * 10), int(250 + start[0] * 10))
    np_goal = (int(250 - goal[1] * 10), int(250 + goal[0] * 10))
    start_elevation = map_handled[np_start[0], np_start[1]]
    goal_elevation = map_handled[np_goal[0], np_goal[1]]
    obs_threshold = max(start_elevation, goal_elevation) + 1.3
    # 定义障碍物
    obstacles = np.where(map_handled > obs_threshold, True, False)
    # 计算到最近非障碍物点的距离
    distances = distance_transform_edt(~obstacles)
    # 定义惩罚点为距离小于惩罚阈值的点
    pun = (distances > 0) & (distances < pun_threshold)
    # 创建三值地图，障碍物点为2，危险点为1，其他为0
    tri_map = np.zeros_like(map_handled, dtype=int)
    tri_map[obstacles] = 2  # 障碍物点
    tri_map[pun] = 1  # 危险点
    return tri_map, np_start, np_goal


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


def planer_main(start, goal, alg_name):
    map_handled = np.load('./algorithm/temp/map_handled.npy')
    # map_handled = np.load('./temp/map_handled.npy') # 测试用
    print('路径规划算法已加载地图。')
    # 计算三元地图
    tri_map, np_start, np_goal = tri_maper(map_handled, start, goal)
    # 对地图调用寻路算法
    alg_dict = {
            "AStar": AStar(map_handled, tri_map, np_start, np_goal),
            "Dijkstra": Dijkstra(map_handled, tri_map, np_start, np_goal),
            "BFS": BFS(map_handled, tri_map, np_start, np_goal),
            "AStarWithH": AStarWithH(map_handled, tri_map, np_start, np_goal, delta=20.0, lambda_val=2.0),
            "AStarWithPun": AStarWithPun(map_handled, tri_map, np_start, np_goal, delta=20.0, lambda_val=2.0),
        }
    if alg_name in alg_dict.keys():
        algorithm = alg_dict[alg_name]
        print(f'已选择“{alg_name}”作为寻路算法')
    else:
        raise NameError
    shortest_path = algorithm.find_path()
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
        direction_diff = (target_direction - car_status[0] + np.pi) % (2 * np.pi) - np.pi
        turn_times = int(np.ceil(abs(direction_diff) / np.radians(18.95)))
        action = 'left' if direction_diff > 0 else 'right'
        for _ in range(turn_times):
            actions.append(action)
        move_times = int(np.ceil(target_distance * 11))
        actions.extend(['ahead'] * move_times)
        return rest_path[scaling_ratio:], actions
