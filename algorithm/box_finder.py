import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

def d_img_viewer(img_depth):
    # 将深度图像中的 inf 替换为 nan
    img_depth[img_depth == np.inf] = np.nan
    # 计算有效深度的最小值和最大值
    min_valid_depth = np.nanmin(img_depth)
    max_valid_depth = np.nanmax(img_depth)
    # 将深度数据缩放到0-255的范围内
    scaled_depth_image = 255 * (img_depth - min_valid_depth) / (max_valid_depth - min_valid_depth)
    # 将 nan 值设置为0
    scaled_depth_image[np.isnan(scaled_depth_image)] = 0
    # 转换为 uint8 类型
    scaled_depth_image = scaled_depth_image.astype(np.uint8)
    return scaled_depth_image

def rgbd_saver(rgb_image, d_image, now_step, camera):
    cv2.imwrite(f'./algorithm/temp/img/interest_target/rgb_{camera}/{str(now_step)}.png', rgb_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f'./algorithm/temp/img/interest_target/d_{camera}/{str(now_step)}.png', d_img_viewer(d_image), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    np.save(f'./algorithm/temp/img/interest_target/d_{camera}/{str(now_step)}', d_image)

def get_camera_coordinates(u, v, z):
    # 相机参数
    width = 1280
    height = 720
    fieldOfView = 1.57  # 视场角，单位：弧度
    # 计算焦距
    fx = fy = 0.5 * width / np.tan(0.5 * fieldOfView)
    # 计算主点
    cx = width / 2
    cy = height / 2
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z
    return [X, Y, Z]

def coords_shift(object_coords_camera, vehicle_angle, now_pos, camera):
    # 偏移量
    offset = np.array([0, 0.2, 0])
    translation_vector = np.array([0, 0, 0.1])  # 考虑到相机的安装高度

    # Step 1: 将物体坐标从相机坐标系转换到小车坐标系（车头方向为y轴正向，垂直地面向上方向为z轴正向）
    if camera == 'f':
        rotation_angle = np.radians(110)
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [0, np.sin(rotation_angle), np.cos(rotation_angle)]])
        object_coords_vehicle = np.dot(rotation_matrix, object_coords_camera) + translation_vector + offset  # 加上偏移量
    elif camera == 'b':
        rotation_angle = np.radians(70)
        rotation_matrix = np.array([[-1, 0, 0],
                                    [0, -np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [0, -np.sin(rotation_angle), np.cos(rotation_angle)]])
        object_coords_vehicle = np.dot(rotation_matrix, object_coords_camera) + translation_vector - offset  # 加上偏移量
    else:
        raise ValueError("无效的相机类型。预期为 'f' 或 'b'。")
    # Step 2: 将物体坐标从小车坐标系转换到地图坐标系
    rotation_matrix2 = np.array([[np.cos(vehicle_angle), np.sin(vehicle_angle)],
                                [-np.sin(vehicle_angle), np.cos(vehicle_angle)]])
    object_coords_map = np.dot(rotation_matrix2, object_coords_vehicle[:2]) + np.array([now_pos[0], now_pos[1]])  # 只考虑二维平面的旋转
    return object_coords_map

def finder_main(img_rgb, img_depth, car_status, now_step, camera):
    all_coordinates = []
    # 对原始RGBD数据预处理
    if img_rgb is not None:
        img_rgb = cv2.cvtColor(np.frombuffer(img_rgb, np.uint8).reshape((720, 1280, 4)), cv2.COLOR_BGRA2BGR)
        img_depth = np.reshape(img_depth, (720, 1280))
        # 设定橙色阈值
        lower_orange = np.array([20, 32, 100])
        upper_orange = np.array([80, 111, 180])
        # 根据阈值构建掩模
        mask = cv2.inRange(img_rgb, lower_orange, upper_orange)
        # 掩模中有足够多像素点时寻找兴趣目标
        if np.count_nonzero(mask) > 100:
            # 对原图像和掩模进行位运算
            res = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
            # 轮廓检测
            num_labels, labels = cv2.connectedComponents(mask)
            if num_labels > 1:
                rgbd_saver(img_rgb, img_depth, now_step, camera)
                # print(f'{now_step}步发现连通区域，mask中有{np.count_nonzero(mask)}个像素点')
                for label in range(1, num_labels):  # Start from 1 because 0 is background
                    component = np.zeros(mask.shape, dtype=np.uint8)
                    component[labels == label] = 255
                    # 计算每个轮廓的中心点
                    M = cv2.moments(component)
                    if M["m00"] > 10000:  # Ignore small components
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # 在结果图像上标记中心点
                        cv2.circle(res, (cX, cY), 5, (255, 255, 255), -1)
                        # print('点深为：', img_depth[cY, cX])
                        C_coordinates = get_camera_coordinates(cX, cY, img_depth[cY, cX])
                        all_coordinates.append(coords_shift(C_coordinates, car_status[0], car_status[1], camera))
                cv2.imwrite(f'./algorithm/temp/img/interest_target/res/{camera}_{str(now_step)}.png', res,
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # # 可视化
                # cv2.imshow('image', img_rgb)
                # cv2.imshow('res', res)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
    return all_coordinates  # 返回所有兴趣目标的地图坐标

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
        # 查找距离小于3的点对
        close_points = [(i, j) for i in range(len(dists)) for j in range(i + 1, len(dists)) if dists[i][j] < 3]
        if not close_points:
            # 没有找到距离小于3的点对，结束循环
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