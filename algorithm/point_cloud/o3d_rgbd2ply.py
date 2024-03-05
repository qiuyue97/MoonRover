import open3d as o3d
import numpy as np

# 计算相机内参
width = 1280
height = 720
fieldOfView = 1.57  # 视场角，单位：弧度
# 计算焦距
fx = fy = 0.5 * height / np.tan(0.5 * fieldOfView)
# 计算主点
cx = width / 2
cy = height / 2
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# # 计算外参
# # Webots rotation parameter
# axis = np.array([0.17108597210580317, -0.9702878418023425, 0.1710879721054771])
# angle = 1.60096
# # Webots translation parameter
# translation = np.array([0.1895, 0, 0])
# # Convert axis-angle to rotation matrix
# axis = axis / np.linalg.norm(axis)  # normalize axis
# K = np.array([
#     [0, -axis[2], axis[1]],
#     [axis[2], 0, -axis[0]],
#     [-axis[1], axis[0], 0]
# ])  # cross product matrix of axis
# R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K  # Rodrigues' rotation formula
# # Create extrinsic matrix
# extrinsic = np.eye(4)
# extrinsic[:3, :3] = R
# extrinsic[:3, 3] = translation

def rgbd2ply(path, img_name, Camera='f'):
    global intrinsic
    deepth_path = path + '/' + 'd_' + Camera + '/' + str(img_name) + '.npy'
    rgb_path = path + '/' + 'rgb_' + Camera + '/' + str(img_name) + '.png'
    # 分别读取并处理RGB及深度图像信息
    depth_np = np.load(deepth_path)
    depth_np[np.isinf(depth_np)] = np.nan
    depth_np = (depth_np * 1000)
    depth_np[np.isnan(depth_np)] = np.iinfo(np.uint16).max
    depth_np = np.ascontiguousarray(depth_np.astype(np.uint16))
    img_d = o3d.geometry.Image(depth_np)
    img_rgb = o3d.geometry.Image(o3d.io.read_image(rgb_path))
    RGBDImage = o3d.geometry.RGBDImage.create_from_color_and_depth(img_rgb, img_d, depth_scale=1000.0, depth_trunc=6.1, convert_rgb_to_intensity=False)
    # 创建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(RGBDImage, intrinsic)
    # 降采样并计算每个点的法向量
    voxel_size = 0.05
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    # Create a new PointCloud with only positions and normals
    pcd_without_colors = o3d.geometry.PointCloud()
    pcd_without_colors.points = pcd.points
    pcd_without_colors.normals = pcd.normals
    # Save to a .ply file
    o3d.io.write_point_cloud(f"{Camera}_{img_name}.ply", pcd, write_ascii=True)
    o3d.visualization.draw_geometries([pcd])
    return pcd_without_colors

path = '../temp/img'
rgbd2ply(path, 6, Camera='b')

# # 创建一个空的全局点云
# global_pcd = rgbd2ply(path, 1, 'b')
# # 初始化变换矩阵
# transformation = np.identity(4)
# for i in range(1, 5):
#     source = rgbd2ply(path, (i + 1), 'b')
#     target = global_pcd
#     # 运行ICP算法来估计source到target的变换矩阵
#     result_icp = o3d.pipelines.registration.registration_icp(
#         source, target, max_correspondence_distance=0.05,
#         init=transformation,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )
#     # 更新变换矩阵
#     transformation = result_icp.transformation
#     # 将对齐后的源点云添加到全局点云
#     source.transform(transformation)
#     global_pcd += source
# # 保存全局点云为.ply文件
# # o3d.io.write_point_cloud("global_pointcloud.ply", global_pcd)
# o3d.visualization.draw_geometries([global_pcd])