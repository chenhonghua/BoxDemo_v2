import open3d as o3d
import numpy as np

# visualize the scene segmentation result
# cluster_planes: a list of planes, in which each element (plane) is a np.array in N*6.
# N is the number of points for that plane
def visualize_scene(cluster_planes):
    viz_pts = []

    viz_transformation = np.identity(4, np.float32)
    viz_transformation[0, 0] = -1.0
    viz_transformation[2, 2] = -1.0
    for plane in cluster_planes:
        viz_pt = o3d.geometry.PointCloud()
        viz_pt.points = o3d.utility.Vector3dVector(plane[:,0:3])
        viz_pt.colors = o3d.utility.Vector3dVector(plane[:,3:])

        # rotate the scene for better visualization
        viz_pt.transform(viz_transformation)
        viz_pts.append(viz_pt)
    
    o3d.visualization.draw_geometries(viz_pts)

def visualize_scene_with_optimal_plane(cluster_planes, grasp_id):
    viz_pts = []

    viz_transformation = np.identity(4, np.float32)
    viz_transformation[0, 0] = -1.0
    viz_transformation[2, 2] = -1.0
    for plane in cluster_planes:
        viz_pt = o3d.geometry.PointCloud()
        viz_pt.points = o3d.utility.Vector3dVector(plane[:,0:3])
        viz_pt.colors = o3d.utility.Vector3dVector(plane[:,3:])

        # rotate the scene for better visualization
        viz_pt.transform(viz_transformation)
        viz_pts.append(viz_pt)
    
    # highlight the selected plane with red color
    viz_pts[grasp_id].paint_uniform_color((1, 0, 0))
    o3d.visualization.draw_geometries(viz_pts)

def visualize_scene_with_pose(cluster_planes, grasp_id, R, t):
    viz_pts = []

    viz_transformation = np.identity(4, np.float32)
    viz_transformation[0, 0] = -1.0
    viz_transformation[2, 2] = -1.0
    for plane in cluster_planes:
        viz_pt = o3d.geometry.PointCloud()
        viz_pt.points = o3d.utility.Vector3dVector(plane[:, 0:3])
        viz_pt.colors = o3d.utility.Vector3dVector(plane[:, 3:])

        # rotate the scene for better visualization
        viz_pt.transform(viz_transformation)
        viz_pts.append(viz_pt)
    
    # highlight the selected plane with red color
    viz_pts[grasp_id].paint_uniform_color((1, 0, 0))

    transformation = np.identity(4, np.float32)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    pose_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1000, origin=[0, 0, 0])
    pose_axis_pcd.transform(transformation)

    # also need to rotate the axis for better visualization
    pose_axis_pcd.transform(viz_transformation)

    o3d.visualization.draw_geometries(viz_pts + [pose_axis_pcd])

# raw_pts: o3d.geometry.PointCloud
# selected_box_pts: np.array
def visualize_pose_in_raw_pts(raw_pts, selected_box_pts, R, t):
    viz_transformation = np.identity(4, np.float32)
    viz_transformation[0, 0] = -1.0
    viz_transformation[2, 2] = -1.0

    # visualize the selected box with red color
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(selected_box_pts)
    box.paint_uniform_color((1, 0, 0))

    transformation = np.identity(4, np.float32)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    pose_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1000, origin=[0, 0, 0])
    pose_axis_pcd.transform(transformation)

    # rotate the scene for better visualization
    box.transform(viz_transformation)
    raw_pts.transform(viz_transformation)
    pose_axis_pcd.transform(viz_transformation)

    o3d.visualization.draw_geometries([box] + [raw_pts] + [pose_axis_pcd])

# render overlap region in the raw scene
# raw_pts: o3d.geometry.PointCloud
# selected_box_pts: np.array
def visualize_pose_in_raw_pts_one_box(raw_pts, selected_box_pts, R, t, overlap, grasper_box, lines):
    viz_transformation = np.identity(4, np.float32)
    viz_transformation[0, 0] = -1.0
    viz_transformation[2, 2] = -1.0

    # visualize the selected box with red color
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(selected_box_pts)
    box.paint_uniform_color((1, 0, 0))

    transformation = np.identity(4, np.float32)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    pose_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    pose_axis_pcd.transform(transformation)

    # rotate the scene for better visualization
    box.transform(viz_transformation)
    raw_pts.transform(viz_transformation)
    pose_axis_pcd.transform(viz_transformation)

    # grasp box
    grasper_box_pc = o3d.geometry.PointCloud()
    grasper_box_pc.points = o3d.utility.Vector3dVector(grasper_box)
    grasper_box_pc.transform(viz_transformation)
    grasper_box = np.asarray(grasper_box_pc.points)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grasper_box),
        lines=o3d.utility.Vector2iVector(lines),
    )

    # overlap region
    overlap.transform(viz_transformation)
    overlap.paint_uniform_color([1, 0.706, 0])

    o3d.visualization.draw_geometries([box] + [raw_pts] + [overlap] + [pose_axis_pcd] + [line_set])

# render gripper box
# raw_pts: o3d.geometry.PointCloud
# selected_box_pts: np.array
def visualize_pose_in_raw_pts_two_box(raw_pts, selected_box_pts, R, t, grasper_box, lines, box_range, box_range_lines):
    viz_transformation = np.identity(4, np.float32)
    viz_transformation[0, 0] = -1.0
    viz_transformation[2, 2] = -1.0

    # visualize the selected box with red color
    box = o3d.geometry.PointCloud()
    box.points = o3d.utility.Vector3dVector(selected_box_pts)
    box.paint_uniform_color((1, 0, 0))

    transformation = np.identity(4, np.float32)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    pose_axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    pose_axis_pcd.transform(transformation)

    # rotate the scene for better visualization
    box.transform(viz_transformation)
    raw_pts.transform(viz_transformation)
    pose_axis_pcd.transform(viz_transformation)

    # grasp box
    grasper_box_pc = o3d.geometry.PointCloud()
    grasper_box_pc.points = o3d.utility.Vector3dVector(grasper_box)
    grasper_box_pc.transform(viz_transformation)
    grasper_box = np.asarray(grasper_box_pc.points)
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(grasper_box),
        lines=o3d.utility.Vector2iVector(lines),
    )

    # box_range
    box_pc = o3d.geometry.PointCloud()
    box_pc.points = o3d.utility.Vector3dVector(box_range)
    box_pc.transform(viz_transformation)
    box_range = np.asarray(box_pc.points)
    colors = [[1, 0, 0] for i in range(len(box_range_lines))]
    line_set2 = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(box_range),
        lines=o3d.utility.Vector2iVector(box_range_lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([box] + [raw_pts] + [pose_axis_pcd] + [line_set] + [line_set2])