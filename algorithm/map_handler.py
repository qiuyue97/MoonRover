import numpy as np
import open3d as o3d

def map_handler(moon_map):
    # 关闭np科学计数法
    np.set_printoptions(suppress=True)
    # 加载地形数据
    print('开始处理地图数据...')
    rows, cols, _ = moon_map.shape
    # 储存高程矩阵
    np.save('./algorithm/temp/map_handled', moon_map[:, :, 3])
    # 确定x轴最小坐标和y轴最大坐标
    x_min = np.ceil(cols / 2) - 1
    y_max = np.ceil(rows / 2) - 1
    # 将x、y坐标矩阵和z坐标合并为一个三维坐标矩阵（高程信息单位暂时目测为0.5m, x、y坐标单位确定为dm）
    x_coords, y_coords = np.meshgrid(np.arange(cols) - x_min, y_max - np.arange(rows))
    coordinates = np.stack((x_coords / 10, y_coords / 10, moon_map[:, :, 3] / 1.417379975200399), axis=-1).reshape(-1, 3)
    # 提取RGB信息
    colors = moon_map[:, :, :3].reshape(-1, 3)
    # 使用o3d处理点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinates)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 计算每个点的法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # Create a new PointCloud with only positions and normals
    pcd_without_colors = o3d.geometry.PointCloud()
    pcd_without_colors.points = pcd.points
    pcd_without_colors.normals = pcd.normals
    # 储存ply文件
    o3d.io.write_point_cloud(filename='./algorithm/temp/map_handled.ply', pointcloud=pcd, write_ascii=True)
    # o3d.visualization.draw_geometries([pcd], window_name='PointCloud Visualization')
    print('地图数据处理完毕。')