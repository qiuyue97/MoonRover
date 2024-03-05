import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import datetime
import os
import shutil

def point_clustering(tar_pos):
    # 移除包含NaN的点
    tar_pos = [point for point in tar_pos if not np.isnan(point).any()]
    # 将tar_pos分为5段
    tar_pos_segments = np.array_split(tar_pos, 5)
    # 用于储存每一段中最大聚类的中心点
    segment_centroids = []
    for segment in tar_pos_segments[1:]:
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=2, min_samples=3).fit(segment)
        # 找出所有的聚类（忽略噪声点）
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)
        centroids = []
        cluster_sizes = []
        for label in unique_labels:
            points_in_cluster = [segment[i] for i in range(len(segment)) if labels[i] == label]
            centroid = np.mean(points_in_cluster, axis=0)
            centroids.append(list(centroid))
            cluster_sizes.append(len(points_in_cluster))
        # 找出最大的三个聚类并添加其中心点到结果中
        if cluster_sizes:
            max_centroids = sorted(zip(centroids, cluster_sizes), key=lambda x: x[1], reverse=True)[:3]
            segment_centroids.extend([centroid for centroid, _ in max_centroids])
    # 从所有找出的聚类中，选出最大的五个
    top_centroids = sorted(segment_centroids, key=lambda x: x[1], reverse=True)[:5]
    # 如果top_centroids中有相互之间距离小于1的点，那么删除聚类较小的那一个点
    while True:
        # 计算所有点的距离
        dists = distance.cdist(top_centroids, top_centroids)
        # 查找距离小于1的点对
        close_points = [(i, j) for i in range(len(dists)) for j in range(i + 1, len(dists)) if dists[i][j] < 3]
        if not close_points:
            # 没有找到距离小于1的点对，结束循环
            break
        # 删除聚类较小的那一个点
        i, j = close_points[0]
        if top_centroids[i][1] > top_centroids[j][1]:  # Corrected here
            top_centroids.pop(j)
        else:
            top_centroids.pop(i)
        # 如果还有剩余的点，添加下一个点
        if len(top_centroids) < 5 and segment_centroids:
            top_centroids.append(segment_centroids.pop(0))
    return top_centroids

def find_closest_point(target_point):
    # 五个兴趣点实际坐标
    points = np.array([[-7.06595, 14.7814], [-16.7138, 20.4243], [-22.0564, 14.6231], [-8.86782, 20.5962], [-13.1897, 9.45819]])
    # 计算目标点和每个点的距离
    distances = np.sqrt(np.sum((points - target_point)**2, axis=1))
    # 找到最近的点和距离
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]
    return min_distance_index, min_distance

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# points = np.load('./real_map.npy')
# elevation = np.load('./map_handled.npy')
# print(elevation.shape)
#
# # 计算x、y、z坐标的最小值和最大值
# x_min, z_min, y_min = np.min(points, axis=0)
# x_max, z_max, y_max = np.max(points, axis=0)
# print(f'x: min={x_min}, max={x_max}, x shape is: {x_max - x_min}')
# print(f'y: min={y_min}, max={y_max}, y shape is: {y_max - y_min}')
# print(f'z: min={z_min}, max={z_max}, z shape is: {z_max - z_min}')
#
# # 计算先验地图中高程的最小值和最大值
# e_min = np.min(elevation[0])
# e_max = np.max(elevation[0])
# print(f'e: min={e_min}, max={e_max}, e shape is: {e_max - e_min}')
# print(f'高程地图的数据应该除以{(e_max - e_min) / (z_max - z_min)}才能变成以m为单位。')

# # 创建一个PointCloud对象
# pcd = o3d.geometry.PointCloud()
#
# # 将坐标添加到PointCloud对象中
# pcd.points = o3d.utility.Vector3dVector(points)
#
# # 可视化点云
# o3d.visualization.draw_geometries([pcd])
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# moon_map = np.load('map.npy')
# map_np = moon_map[:, :, 3] / 1.417379975200399
# np.save('map_np', map_np)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# tar = np.array([-15, 15])
# now = np.array([-14.61865831, 14.98553827])
# distances = np.sqrt(np.sum((tar - now)**2))
# print(distances)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# target_pos = [[-16.704171486252157, 10.23946787762889], [-18.177401720334355, 17.209447349706153]]
# for tar in target_pos:
#     min_distance_index, min_distance = find_closest_point(tar)
#     print(f"Closest point index: {min_distance_index + 1}, Distance: {min_distance}")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 创建点集

def result_saver(real_pos):
    counter = [0, 0, 0, 0, 0]
    points = np.load('./algorithm/temp/tar_all.npy')
    target_pos = point_clustering(points)
    for tar in target_pos:
        min_distance_index, min_distance = find_closest_point(tar)
        print(f"Closest point index: {min_distance_index + 1}, Distance: {min_distance}")
        if min_distance <= 2:
            counter[min_distance_index] = 1
    # 兴趣目标所在点
    additional_points = [[-7.06595, 14.7814], [-16.7138, 20.4243], [-22.0564, 14.6231], [-8.86782, 20.5962], [-13.1897, 9.45819]]
    # 生成一个颜色映射
    color_range = np.linspace(0, 1, len(points))
    colors = plt.cm.Blues(color_range)  # 使用Blues颜色映射
    plt.figure(figsize=(6, 6))
    # 将坐标轴的范围设置为-25到25
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # 遍历点集，画出每一个点，颜色按照序列号渐变
    for i in range(len(points)):
        plt.scatter(*points[i], color=colors[i])
    # 添加target_pos中的所有点，颜色为深红色
    for target in target_pos:
        plt.scatter(*target, color='darkred')
    # 添加兴趣目标所在点，颜色为金黄色
    for point in additional_points:
        plt.scatter(*point, color='gold')
    # 添加小车实际位置点，颜色为紫色
    plt.scatter(*real_pos, color='purple')
    # 添加小车应在位置点，颜色为绿色
    plt.scatter(*[-15, 15], color='green')
    # 创建一个用于colorbar的归一化对象
    norm = plt.Normalize(color_range.min(), color_range.max())
    # 创建colorbar，使用Blues颜色映射
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    plt.colorbar(sm)
    # plt.show()
    # 获取当前时间
    now = datetime.datetime.now()
    # 格式化为月_日_时_分
    time_str = now.strftime("%m_%d_%H_%M")
    # 指定保存路径和文件名
    save_path = os.path.join("./algorithm/result", f"{time_str}.png")
    plt.savefig(save_path)
    # 打开 log.txt 文件并添加一行文字
    with open('./algorithm/result/log_2_3.txt', 'a', encoding='utf-8') as log_file:
        log_file.write(f'{now.strftime("%m-%d %H:%M")}小车最终落于{real_pos},与目标区域中心相距{distance.euclidean(real_pos, [-15, 15])}米，发现的兴趣点：{counter}。\n')
    shutil.copyfile('./algorithm/temp/tar_all.npy', f'./algorithm/result/tar/{now.strftime("%m_%d_%H_%M")}.npy')
    # 指定要删除的目录
    dir_path = "./algorithm/temp"
    # 使用shutil.rmtree删除指定目录及其所有内容
    shutil.rmtree(dir_path, ignore_errors=True)
