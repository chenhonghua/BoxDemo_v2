import numpy as np
import cv2

# compute the grasp score for a single plane
def compute_grasp_scores(single_plane, resolution, filter_z_max):
    # 1. pre-process
    points = single_plane[:, 0:3]
    data_mean = np.mean(points, axis=0)
    points = points - data_mean  # normalization

    # 2. PCA analysis
    H = np.dot(points.T, points)
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)   # H = U S V

    # 3. local projection
    points2 = np.dot(points, eigenvectors)

    # 4. bounding box computation
    min_x = min(points2[:, 0])
    min_y = min(points2[:, 1])
    max_x = max(points2[:, 0])
    max_y = max(points2[:, 1])
    rect_area = (max_x - min_x) * (max_y - min_y)

    # 5. score computation
    score_area = resolution*resolution*points.shape[0]/rect_area  #  area ratio, resolution*resolution approximates to unit area
    score_zval = -data_mean[2]  # / filter_z_max
    score_pnum = points.shape[0]

    scores = [score_area, score_zval, score_pnum]

    return scores

# compute the grasp score for a single plane by a easy scheme
def get_grasp_plane_id_hard (plane_scores):
    plane_scores = np.array(plane_scores)

    # 1. sort by score_pnum
    plane_scores1 = plane_scores[:, 2]
    top_k1_idx = plane_scores1.argsort()[::-1]
    grasp_plane_ids = top_k1_idx

    # 2. sort by score_area_ratio
    plane_scores2 = plane_scores[top_k1_idx, 0]
    top_k1_idx2 = plane_scores2.argsort()[::-1]
    top_k1_idx3 = top_k1_idx[top_k1_idx2]
    grasp_plane_ids = grasp_plane_ids[top_k1_idx2]

    # 3. sort by score_zval
    plane_scores3 = plane_scores[top_k1_idx3, 1]
    top_k1_idx4 = plane_scores3.argsort()[::-1]
    grasp_plane_ids = grasp_plane_ids[top_k1_idx4]

    return grasp_plane_ids

# compute three grasp proposal according to the plane height
def get_grasp_proposals(plane_scores):
    plane_scores = np.array(plane_scores)
    # 1. select 3 planes with max size
    plane_scores = plane_scores[:, 0]
    top_3_idx = plane_scores.argsort()[::-1][0:3]
    return top_3_idx

# select the optimal plane for robot grasping
def grasp_selection(cluster_planes, params):
    # 1. compute the grasp score for each plane
    plane_scores = []
    resolution = params.downsample_resolution
    filter_z_max = params.filter_z_max
    for plane in cluster_planes:
        plane_scores.append(compute_grasp_scores(plane, resolution, filter_z_max))
    
    # 2. select the plane with the highest grasp score (plane size/height/occlusion)
    if(len(plane_scores) >=3 ):
        grasp_plane_ids = get_grasp_plane_id_hard(plane_scores)
    else:
        grasp_plane_ids = [0]

    return grasp_plane_ids
