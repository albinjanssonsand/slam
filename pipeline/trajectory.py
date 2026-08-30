"""
Trajectory export shared by pipeline entry points that need to write
estimated keyframe poses to disk for ground-truth scoring (e.g. with
evo_ape/evo_rpe against a TUM RGB-D sequence's groundtruth.txt).
"""

import os

from scipy.spatial.transform import Rotation

from pipeline.pose import camera_center


def write_tum_trajectory(path, keyframe_poses):
    """
    Writes each keyframe's pose as one TUM-format line: "timestamp tx ty tz
    qx qy qz qw" - the format TUM's own tools and evo (evo_ape/evo_rpe)
    expect for both ground truth and estimated trajectories.

    keyframe_poses entries expose .R/.t/.timestamp (see
    pipeline.mapping.KeyframePose) in this pipeline's world-to-camera
    convention (X_cam = R @ X_world + t; see pose.py). TUM expects the
    inverse: camera position and orientation *in the world frame*. Position
    reuses pose.camera_center (-R.T @ t), the same inversion already used
    for plotting; orientation is R.T (rotation from camera frame to world
    frame), converted to a quaternion via scipy - Rotation.as_quat() returns
    (x, y, z, w), which is already TUM's column order.
    """
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w") as f:
        for pose in keyframe_poses:
            position = camera_center(pose.R, pose.t).ravel()
            qx, qy, qz, qw = Rotation.from_matrix(pose.R.T).as_quat()
            f.write(
                f"{pose.timestamp:.6f} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )
