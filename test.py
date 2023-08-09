import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('D:/project/paper_flatten/data/yaguan/data_Gray/a_mess/case_002/result/result.ply')
# pcd2 = o3d.io.read_point_cloud('D:/project/paper_flatten/data/yaguan/data_RGB/a_mess/case_003/case_003.ply')
# # pcd = o3d.io.read_point_cloud('D:/project/paper_flatten/code/integrated_code/ply/DenseCloud.ply')
# pcd2 = o3d.io.read_point_cloud('D:/project/paper_flatten/code/integrated_code/temp/interpcd_downsample.ply')
# print(pcd)
print(pcd)
o3d.visualization.draw_geometries([pcd],mesh_show_wireframe=True,mesh_show_back_face=True)
    
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 3*avg_dist

# pcd.estimate_normals()
# bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))

# dec_mesh = bpa_mesh.simplify_quadric_decimation(100000)

# dec_mesh.remove_degenerate_triangles()
# dec_mesh.remove_duplicated_triangles()
# dec_mesh.remove_duplicated_vertices()
# dec_mesh.remove_non_manifold_edges()

# o3d.io.write_triangle_mesh('D:/project/paper_flatten/data/yaguan/data_RGB/a_mess/case_004/case_004_1.ply', dec_mesh)

# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pcd)
# vis.run()
# vis.destroy_window()

# pcd2 = o3d.io.read_point_cloud('D:/project/paper_flatten/code/integrated_code/temp/interpcd_t.ply')
# print(pcd2)
# box = pcd2.get_axis_aligned_bounding_box()
# boxMin = box.get_min_bound()
# boxMax = box.get_max_bound()
# diagonal = np.sum(np.power(boxMax-boxMin, 2))**0.5
# downpcd, nda, intv = pcd2.voxel_down_sample_and_trace(voxel_size=diagonal/1000, min_bound=boxMin, max_bound=boxMax)

# nda1 = []
# for i in range(nda.shape[0]):
#     for j in range(8):
#         if nda[i, j] != -1:
#             nda1.append(nda[i, j])
# nda1 = np.sort(np.asarray(nda1))
# print(nda1)

# intv1 = []
# for i in range(len(intv)):
#     inv = np.asarray(intv[i])
#     for j in range(len(inv)):
#         intv1.append(inv[j])
# intv1 = np.sort(np.asarray(intv1))
# print(intv1)

# print(nda1.shape, intv1.shape)
