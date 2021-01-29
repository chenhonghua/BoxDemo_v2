import open3d as o3d
import numpy as np

# load .ply and transform it into numpy array
def load_ply(ply_path):
    pts = o3d.io.read_point_cloud(ply_path)

    # N * 3
    np_pts = np.array(pts.points).astype(np.float32)

    return np_pts