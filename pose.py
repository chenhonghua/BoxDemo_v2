import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as Rot

# estimated the 6d pose for the selected plane
def pose_estimation(selected_plane):
    # 1. pre-process
    plane_pts = selected_plane[:, 0:3]
    pts_center = np.mean(plane_pts, axis=0)
    normalized_pts = plane_pts - pts_center  # normalization

    # 2. PCA analysis
    H = np.dot(normalized_pts.T, normalized_pts)
    eigenvectors, eigenvalues, eigenvectors_t = np.linalg.svd(H)   # H = U S V

    # 3. local projection
    projected_pts = np.dot(normalized_pts, eigenvectors)
    projected_pts = projected_pts.T
    np.savetxt("ds.txt", projected_pts.T)

    projected_pts = projected_pts[:2, :]

    # 4. fit the obb of the projected point clouds
    rec = cv2.minAreaRect(projected_pts.T.astype(np.float32))
    normal = eigenvectors[:, 2]

    # 5. represent points in desired object coordinate system
    # z
    pts = plane_pts.T
    n = normal[:3].reshape(1, 3)
    depths = n @ pts
    average_depth = np.mean(depths)
    zs = depths - average_depth

    # x & y
    angle = -rec[2] * math.pi / 180.0
    rotation_matrix = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], np.float32)
    projected_pts = projected_pts - np.array([rec[0][0], rec[0][1]]).reshape(2, 1)
    projected_pts = rotation_matrix @ projected_pts

    object_frame_pts = np.vstack([projected_pts, zs])
    object_center = np.mean(object_frame_pts, axis=1).reshape(3, 1)
    object_frame_pts = object_frame_pts - object_center

    # 6. represent points in camera frame
    camera_frame_pts = pts - pts_center.reshape(3, 1)

    # 7. compute the rotation matrix
    M = np.dot(object_frame_pts, camera_frame_pts.T)
    u, sigma, vt = np.linalg.svd(M)
    R = np.dot(vt.T, u.T)

    # 8. check the R, right-hand or left-hand?
    axis_x = R[:, 0]
    axis_y = R[:, 1]
    axis_z = R[:, 2]
    # if np.dot(np.cross(axis_x, axis_y), axis_z) < 0, indicates the determined object fraome is left-hand
    # revise it to right-hand
    if np.dot(np.cross(axis_x, axis_y), axis_z) < 0:
        zs = - zs
        object_frame_pts = np.vstack([projected_pts, zs])
        object_center = np.mean(object_frame_pts, axis=1).reshape(3, 1)
        object_frame_pts = object_frame_pts - object_center

        M = np.dot(object_frame_pts, camera_frame_pts.T)
        u, sigma, vt = np.linalg.svd(M)
        R = np.dot(vt.T, u.T)
    
    # 9. update the translation
    t = pts_center.reshape(3, 1) - R @ object_center
    t = t.flatten()

    # 10. transformation matrix to quaternion
    r = Rot.from_matrix(R)
    q = r.as_quat()

    # 11. pose == [x y z rx ry rz rw]
    pose = [t[0], t[1], t[2], q[0], q[1], q[2], q[3]]

    # 12
    rec_len = max(normalized_pts.dot(axis_x))*2
    rec_wid = max(normalized_pts.dot(axis_y))*2


    return R, t, pose, rec_len, rec_wid