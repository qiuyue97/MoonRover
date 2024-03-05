import open3d as o3d
# visualization of point clouds.
pcd = o3d.io.read_point_cloud('b_4_widthPCTrans.ply')
# pcd = o3d.io.read_point_cloud('./data/parasaurolophus_6700.ply')
scene = o3d.io.read_point_cloud('map_handled.ply')

# red[1,0,0],green[0,1,0],blue[0,0,1]
# pcd.paint_uniform_color([1, 0, 0])
# scene.paint_uniform_color([0, 0, 1])

# o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd, scene], mesh_show_wireframe=True)
# o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True)