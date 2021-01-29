import argparse

parser = argparse.ArgumentParser()

# segment relevant
parser.add_argument('--downsample_resolution', type=float, default=5.0, help='the resolution for pcl voxel grid filter')
parser.add_argument('--filter_z_min', type=float, default=800.0) # 1354 for 11.30 data  1390 for 1231 data
parser.add_argument('--filter_z_max', type=float, default=600.0) # 1000 for 11.30 data  1200 for 1231 data

# parser.add_argument('--downsample_resolution', type=float, default=0.005, help='the resolution for pcl voxel grid filter')
# parser.add_argument('--filter_z_min', type=float, default=1.75) # 1354 for 11.30 data
# parser.add_argument('--filter_z_max', type=float, default=1.4) # 1000 for 11.30 data
parser.add_argument('--seg_min_number', type=int, default=200)
parser.add_argument('--seg_max_number', type=int, default=10000)
parser.add_argument('--seg_smoothness', type=float, default=0.06)
parser.add_argument('--seg_curvature', type=int, default=0.4)
parser.add_argument('--filter_plane1_para', type=list,
                    default=[0.0226342, -0.663957, -0.747428, -304.156158, 102.591934, 744.129944])
parser.add_argument('--filter_plane2_para', type=list,
                    default=[-0.998418, 0.002367, -0.056178, 122.822815, 37.618717, 767.435303])
parser.add_argument('--grasper_size_para', type=list,
                    default=[0.28*1000, 0.2*1000, 0.16*1000, 0.115*1000])  # grasper_box_x/y/z/Sucker length
