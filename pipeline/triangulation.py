"""
Triangulate 3D points from two keyframe poses and their matched image points.
"""

import cv2
import numpy as np


def triangulate(R1, t1, R2, t2, camera_matrix, pts1, pts2, min_parallax_deg=1.0):
    """
    Triangulate matched points seen from two world-to-camera poses:
      X_cam = R @ X_world + t

    Returns (points_3d, valid_mask, in_front_mask, parallax_deg):
      points_3d      - Nx3 world-frame points
      valid_mask     - boolean mask, True where the point is worth keeping:
                       in_front_mask AND parallax_deg > min_parallax_deg
      in_front_mask  - boolean mask, True where the point has positive depth
                       in BOTH cameras (the cheirality check)
      parallax_deg   - per-point angle (degrees) between the two viewing rays;
                       rejects points near the direction of travel/rotated
                       viewing axis, where near-zero parallax makes
                       triangulation numerically unstable (the "sprinkler"
                       artifact - points thrown out to an arbitrary depth).

    Points collapsing to an implausibly CLOSE depth (the opposite-direction
    failure of the same instability, which this angle check alone doesn't
    catch) are no longer filtered here - that's now handled upstream by
    requiring independent re-observation before a point is trusted (see
    mapping.Map's provisional/confirmed point lifecycle), which is a more
    general check: it also catches whole-batch pose bias that a per-point
    geometric heuristic can't.
    """
    P1 = camera_matrix @ np.hstack([R1, t1])
    P2 = camera_matrix @ np.hstack([R2, t2])

    pts1_h = np.asarray(pts1, dtype=np.float64).reshape(-1, 2).T  # 2xN
    pts2_h = np.asarray(pts2, dtype=np.float64).reshape(-1, 2).T

    points_4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    points_3d = (points_4d[:3] / points_4d[3]).T  # Nx3

    depth1 = (R1 @ points_3d.T + t1).T[:, 2]
    depth2 = (R2 @ points_3d.T + t2).T[:, 2]
    in_front = (depth1 > 0) & (depth2 > 0)

    center1 = (-R1.T @ t1).ravel()
    center2 = (-R2.T @ t2).ravel()
    ray1 = points_3d - center1
    ray2 = points_3d - center2
    ray1 /= np.linalg.norm(ray1, axis=1, keepdims=True) + 1e-12
    ray2 /= np.linalg.norm(ray2, axis=1, keepdims=True) + 1e-12
    cos_angle = np.clip(np.sum(ray1 * ray2, axis=1), -1.0, 1.0)
    parallax_deg = np.degrees(np.arccos(cos_angle))

    valid = in_front & (parallax_deg > min_parallax_deg)

    return points_3d, valid, in_front, parallax_deg