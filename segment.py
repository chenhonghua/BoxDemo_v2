import pcl
import numpy as np
from config import parser

# function for filtering the points near to certain plane
def removing_plane_bg(cloud, plane_paras):
    np_pts = cloud.to_array()
    np_pts_vec = np_pts - plane_paras[3:6]
    p2p_dist = np.dot(np_pts_vec, plane_paras[0:3])
    outlier_idxs = np.where(p2p_dist > 1.4)
    outlier_pts = np_pts[outlier_idxs]
    outlier_pts = outlier_pts.astype(np.float32)
    cloud_out = pcl.PointCloud()
    cloud_out.from_array(outlier_pts)
    return cloud_out

# function for scene plane segmentation
# input_pts: np.array (N*3)
# params: parameter definition in config.py
# return: scene planes
def plane_extract(input_pts, params):
    # 1. transform input_pts to pcl.PointCloud
    cloud_in = pcl.PointCloud()
    cloud_in.from_array(input_pts)
    # print('input successful...')

    # 2. Uniform Downsampling
    uni_down_Filter = cloud_in.make_voxel_grid_filter()
    resolution = params.downsample_resolution
    uni_down_Filter.set_leaf_size(resolution, resolution, resolution)
    cloud_ds = uni_down_Filter.filter()
    # np.savetxt("ds.txt", cloud_ds.to_array())

    # 3. Filtering by Z value
    val_cond = cloud_ds.make_ConditionAnd()
    filter_z_max = params.filter_z_max
    filter_z_min = params.filter_z_min
    val_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.GT, filter_z_max)
    val_cond.add_Comparison2('z', pcl.CythonCompareOp_Type.LT, filter_z_min)
    val_cond_Filter = cloud_ds.make_ConditionalRemoval(val_cond)
    cloud_ds_zf = val_cond_Filter.filter()

    # 4. Filtering by plane_para
    cloud_ds_zf = removing_plane_bg(cloud_ds_zf, params.filter_plane1_para)
    cloud_ds_zf = removing_plane_bg(cloud_ds_zf, params.filter_plane2_para)
    # np.savetxt("fz.txt", cloud_ds_zf.to_array())

    # 4. filter outlier
    sor_Filter = cloud_ds_zf.make_statistical_outlier_filter()
    sor_Filter.set_mean_k(20)
    sor_Filter.set_std_dev_mul_thresh(1.0)
    cloud_ds_zf_sor = sor_Filter.filter()
    # np.savetxt("filter.txt", cloud_ds_zf_sor.to_array())

    # 5. Segmentation based on region growing
    tree = cloud_ds_zf_sor.make_kdtree()
    segment = cloud_ds_zf_sor.make_RegionGrowing(ksearch=15)
    seg_min_number = params.seg_min_number
    seg_max_number = params.seg_max_number
    segment.set_MinClusterSize(seg_min_number)
    segment.set_MaxClusterSize(seg_max_number)
    segment.set_NumberOfNeighbours(15)
    seg_smoothness = params.seg_smoothness
    seg_curvature = params.seg_curvature
    segment.set_SmoothnessThreshold(seg_smoothness)
    segment.set_CurvatureThreshold(seg_curvature)
    segment.set_SearchMethod(tree)
    cluster_indices = segment.Extract()

    # 6. Save all planes
    cluster_planes = []
    for j, indices in enumerate(cluster_indices):
        points = np.zeros((len(indices), 6), dtype=np.float32)
        color = np.random.rand(3)  # generate a random color for each plane cluster
        for i, indice in enumerate(indices):
            points[i][0] = cloud_ds_zf_sor[indice][0]
            points[i][1] = cloud_ds_zf_sor[indice][1]
            points[i][2] = cloud_ds_zf_sor[indice][2]
            points[i][3] = color[0]
            points[i][4] = color[1]
            points[i][5] = color[2]
        cluster_planes.append(points)
    
    return cluster_planes, cloud_ds.to_array()