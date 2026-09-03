"""
Triangulate 3D points from two keyframe poses and their matched image points.
"""

import cv2
import numpy as np


def triangulate(R1, t1, R2, t2, camera_matrix, pts1, pts2, min_parallax_deg=1.0,
                 octave1=None, octave2=None, pyramid_scale_factor=1.2,
                 max_reproj_chi2=5.991, max_scale_ratio_factor=1.5):
    """
    Triangulate matched points seen from two world-to-camera poses:
      X_cam = R @ X_world + t

    Returns (points_3d, valid_mask, in_front_mask, parallax_deg):
      points_3d      - Nx3 world-frame points
      valid_mask     - boolean mask, True where the point is worth keeping:
                       in_front_mask AND parallax_deg > min_parallax_deg AND
                       (when octave1/octave2 are given) the reprojection-
                       error/scale-consistency checks below
      in_front_mask  - boolean mask, True where the point has positive depth
                       in BOTH cameras (the cheirality check)
      parallax_deg   - per-point angle (degrees) between the two viewing rays;
                       rejects points near the direction of travel/rotated
                       viewing axis, where near-zero parallax makes
                       triangulation numerically unstable (the "sprinkler"
                       artifact - points thrown out to an arbitrary depth).

    octave1/octave2 - per-point ORB pyramid octave (parallel arrays to
        pts1/pts2) each keypoint was actually detected at in its own view.
        When given, folds the paper's two remaining §VI-C acceptance checks
        into valid_mask (until now, only cheirality/parallax were checked
        here - see the note that used to be here about a systematically-
        biased-but-internally-self-consistent triangulated batch being able
        to satisfy every check downstream too):

          - reprojection error: the candidate point's reprojection error in
            EACH view must fall within a chi-squared bound (max_reproj_chi2,
            default 5.991 - the standard 95%-confidence, 2-DOF threshold
            ORB-SLAM2 itself uses for monocular reprojection checks) scaled
            by that view's own detection-octave variance
            (pyramid_scale_factor ** (2*octave)) - a keypoint found at a
            coarser pyramid level is less precisely localized in original-
            image pixels, so it's allowed a proportionally larger error,
            consistent with this codebase's own scale-invariance bounds
            (see mapping.Map's d_min/d_max).
          - scale consistency: the ratio of the point's distance to each
            camera center must be consistent (within max_scale_ratio_factor,
            itself scaled by one extra factor of pyramid_scale_factor -
            matching ORB-SLAM2's own tolerance) with the ratio of the two
            views' pyramid scale factors at the octaves each keypoint was
            actually detected at - a point genuinely seen at those two
            octaves should sit at a matching relative distance; one that
            doesn't is evidence of a bad correspondence even though it
            passed cheirality/parallax.

        Omitted (None, the default, on either side) skips both checks
        entirely - valid_mask reverts to the original cheirality+parallax-
        only test. pipeline.pose's older standalone demo (superseded by
        mapping.Map's persistent-map pipeline) has no per-keypoint octave to
        pass and relies on this default.

    Points collapsing to an implausibly CLOSE depth (the opposite-direction
    failure of the parallax instability above) are NOT caught by the
    reprojection-error check, octave1/octave2 or not: for exactly 2 views,
    cv2.triangulatePoints's linear DLT solution reprojects with ~zero error
    to WHATEVER pts1/pts2 it was given, at whatever depth that implies -
    this holds for any epipolar-consistent correspondence, wrong depth
    included, not just a correct one (only a correspondence that violates
    epipolar geometry - i.e. wasn't a real match to begin with - leaves
    genuine reprojection residual once triangulated). The reprojection-error
    check above is therefore a check on the CORRESPONDENCE (was this really
    the same 3D point in both views), not on the resulting depth's
    plausibility; a numerically-unstable-but-epipolar-consistent collapse to
    an implausible depth is still only caught by the parallax-angle check.
    """
    P1 = camera_matrix @ np.hstack([R1, t1])
    P2 = camera_matrix @ np.hstack([R2, t2])

    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d[:3] / points_4d[3]).T  # Nx3

    depth1 = (R1 @ points_3d.T + t1).T[:, 2]
    depth2 = (R2 @ points_3d.T + t2).T[:, 2]
    in_front = (depth1 > 0) & (depth2 > 0)

    center1 = (-R1.T @ t1).ravel()
    center2 = (-R2.T @ t2).ravel()
    ray1 = points_3d - center1
    ray2 = points_3d - center2
    dist1 = np.linalg.norm(ray1, axis=1)
    dist2 = np.linalg.norm(ray2, axis=1)
    ray1 = ray1 / (dist1[:, None] + 1e-12)
    ray2 = ray2 / (dist2[:, None] + 1e-12)
    cos_angle = np.clip(np.sum(ray1 * ray2, axis=1), -1.0, 1.0)
    parallax_deg = np.degrees(np.arccos(cos_angle))

    valid = in_front & (parallax_deg > min_parallax_deg)

    if octave1 is not None and octave2 is not None:
        octave1 = np.asarray(octave1).ravel()
        octave2 = np.asarray(octave2).ravel()

        proj1, _ = cv2.projectPoints(points_3d, cv2.Rodrigues(R1)[0], t1, camera_matrix, None)
        proj2, _ = cv2.projectPoints(points_3d, cv2.Rodrigues(R2)[0], t2, camera_matrix, None)
        err1_sq = np.sum((proj1.reshape(-1, 2) - pts1) ** 2, axis=1)
        err2_sq = np.sum((proj2.reshape(-1, 2) - pts2) ** 2, axis=1)
        sigma1_sq = pyramid_scale_factor ** (2 * octave1)
        sigma2_sq = pyramid_scale_factor ** (2 * octave2)
        reproj_ok = (
            (err1_sq <= max_reproj_chi2 * sigma1_sq)
            & (err2_sq <= max_reproj_chi2 * sigma2_sq)
        )

        ratio_dist = dist2 / np.maximum(dist1, 1e-9)
        ratio_octave = (pyramid_scale_factor ** octave1) / (pyramid_scale_factor ** octave2)
        ratio_factor = max_scale_ratio_factor * pyramid_scale_factor
        scale_ok = (ratio_dist * ratio_factor >= ratio_octave) & (
            ratio_dist <= ratio_octave * ratio_factor
        )

        valid = valid & reproj_ok & scale_ok

    return points_3d, valid, in_front, parallax_deg