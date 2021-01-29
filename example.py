import pcl
from box_io import load_ply
from visualize import *
from segment import plane_extract
from grasp_selection import grasp_selection
from pose import pose_estimation
from config import parser
from collision import grasper_collision_test, box_collision_test

import time

params = parser.parse_args()
if __name__ == '__main__':
    ply_path = './data/20210116/All/2021_01_16_14_55_34_818_points.ply'
    pts = load_ply(ply_path)

    start_time = time.time()
    cluster_planes, cloud_ds = plane_extract(pts, params)
    # visualize_scene(cluster_planes)

    grasp_ids = grasp_selection(cluster_planes, params)
    # grasp_ids = list(range(len(cluster_planes)))
    # visualize_scene_with_optimal_plane(cluster_planes, grasp_id)
    end_time1 = time.time()
    print('time cost for plane segmentation:', end_time1 - start_time)

    grasper_box = []
    box_lines = []
    box_range = []
    box_range_lines = []
    box_is_collision = 0
    box_graspable = 1
    grasp_id = 0
    if (len(grasp_ids) == 1):
        grasp_id = 0
    else:
        for idx in range(len(grasp_ids)):
            cluster_id = grasp_ids[idx]
            R, t, pose, rec_len, rec_wid = pose_estimation(cluster_planes[cluster_id])
            # box collision detection
            box_range, box_range_lines, points_inbox, box_is_collision, overlap_points = box_collision_test(cloud_ds, R, t, rec_len, rec_wid, params)
            if(box_is_collision != 1):
                # grasper collision detection
                grasper_box, box_lines, points_inbox, box_graspable, overlap_points = grasper_collision_test(cloud_ds, R, t, params)
                if (box_graspable == 1):
                    grasp_id = cluster_id
                    box_is_collision = 0
                    break
            else:
                continue
            # visualize_scene_with_pose(cluster_planes, cluster_id, R, t)

    end_time2 = time.time()
    print('time cost for pose estimation:', end_time2 - end_time1)

    scene_pts = o3d.io.read_point_cloud(ply_path)
    # scene_pts = o3d.geometry.PointCloud()
    # scene_pts.points = o3d.utility.Vector3dVector(cloud_ds)
    if(box_graspable == 0):
        overlap = o3d.geometry.PointCloud()
        overlap.points = o3d.utility.Vector3dVector(overlap_points)
        visualize_pose_in_raw_pts_one_box(scene_pts, cluster_planes[grasp_id][:, 0:3], R, t, overlap, grasper_box, box_lines)
    else:
        visualize_pose_in_raw_pts_two_box(scene_pts, cluster_planes[grasp_id][:, 0:3], R, t, grasper_box, box_lines, box_range, box_range_lines)