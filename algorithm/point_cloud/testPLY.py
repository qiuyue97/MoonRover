import numpy as np
import open3d as o3d

def create_point_cloud(depth, K_inv):
    h, w = depth.shape
    y, x = np.indices((h, w))
    normalized_points = np.dot(K_inv, np.stack([x, y, np.ones_like(x)], axis=0).reshape(3, -1))
    points_3d = normalized_points * depth.reshape(1, -1)
    return points_3d

# Load your depth map
Camera = 'b'
img_name = 4
depth = np.load(f'../temp/img/d_{Camera}/{str(img_name)}.npy')  # Load depth information from .npy file
depth[depth == np.inf] = 0

# Define your camera matrix K
width = 1280
height = 720
fieldOfView = 1.57  # field of view in radians
fx = fy = 0.5 * height / np.tan(0.5 * fieldOfView)
cx = width / 2
cy = height / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K_inv = np.linalg.inv(K)

# Create the point cloud
points_3d = create_point_cloud(depth, K_inv)

# Convert to Open3D PointCloud object for visualization
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d.T)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

# 降采样并计算每个点的法向量
voxel_size = 0.05
pcd = pcd.voxel_down_sample(voxel_size)
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))

o3d.io.write_point_cloud(f"{Camera}_{img_name}_self.ply", pcd, write_ascii=True)