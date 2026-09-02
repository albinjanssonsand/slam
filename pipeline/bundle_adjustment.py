"""
Lightweight sliding-window local bundle adjustment.

Jointly refines a window of recent keyframe poses and the map points they
observe, minimizing total reprojection error. Without this, every triangulated
point is trusted as permanent ground truth the moment it's added, so small
pose errors get baked into the map and compound over time (see NOTES.md).
Every window includes at least one fixed keyframe as a gauge anchor -
otherwise the whole optimization is free to drift/rotate/translate as a
rigid whole with nothing to anchor "correct" against.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _build_sparsity(kf_idx, pt_idx, pose_to_free, n_free_poses, n_points):
    """
    Boolean sparsity pattern (2*n_obs x n_vars): which parameters affect which
    residuals. Each observation only depends on its own keyframe's 6 pose
    params (if free) and its own point's 3 coordinates - never anything else.
    Without this, scipy's finite-difference Jacobian perturbs every one of the
    (often 1000+) point variables and re-evaluates the FULL residual vector
    each time, which is what makes an unconstrained call freeze/crawl.

    pose_to_free[k] is the free-parameter index for keyframe k, or -1 if
    keyframe k is fixed (contributes no pose columns).
    """
    n_obs = len(kf_idx)
    n_vars = n_free_poses * 6 + n_points * 3
    sparsity = lil_matrix((2 * n_obs, n_vars), dtype=bool)

    for j in range(n_obs):
        rows = slice(2 * j, 2 * j + 2)
        free_k = pose_to_free[kf_idx[j]]
        if free_k >= 0:
            sparsity[rows, free_k * 3: free_k * 3 + 3] = True
            sparsity[rows, n_free_poses * 3 + free_k * 3: n_free_poses * 3 + free_k * 3 + 3] = True

        p = pt_idx[j]
        point_col = n_free_poses * 6 + p * 3
        sparsity[rows, point_col: point_col + 3] = True

    return sparsity.tocsr()


def local_bundle_adjustment(rotations, translations, points, observations, camera_matrix,
                             fixed_poses=None, outlier_threshold_px=3.0, max_nfev=1000,
                             ftol=1e-4, xtol=1e-4):
    """
    rotations, translations - lists of length K: world-to-camera poses
                               (X_cam = R @ X_world + t) for K keyframes.
    points                  - Nx3 array of map points observed by this window.
    observations            - list of (keyframe_idx, point_idx, x, y) - a pixel
                               observation of points[point_idx] in keyframe
                               rotations[keyframe_idx]/translations[keyframe_idx].
    camera_matrix           - shared 3x3 intrinsics.
    fixed_poses             - boolean array of length K marking which keyframe
                               poses are held fixed rather than refined, e.g.
                               a gauge anchor, or (see mapping._run_local_ba)
                               a keyframe outside the covisibility-scoped local
                               set that still observes one of this window's
                               points and so should still constrain it, without
                               its own pose drifting. None fixes only
                               rotations[0]/translations[0] - a single gauge
                               anchor, nothing else fixed.
    ftol, xtol              - least_squares relative convergence tolerances.
                               The defaults are tuned for local BA's per-keyframe
                               use case (exit fast once "good enough" - see the
                               comment at the solver call below); a one-shot
                               full-map pass has no such time pressure and
                               should be allowed to converge tighter, since a
                               map already refined incrementally by many
                               overlapping local BA windows can otherwise look
                               "converged" to a loose xtol/ftol within a
                               handful of iterations without actually reaching
                               a joint optimum (see _run_global_ba).
    outlier_threshold_px    - Huber loss transition point (pixels). Feature
                               matching occasionally lets a wrong correspondence
                               through the earlier RANSAC checks; with plain
                               least-squares a single such outlier's large
                               residual can dominate the whole solve and throw
                               a keyframe's pose way off (self-corrects later
                               once more data outweighs it, but the bad state
                               is still visible in the meantime). Huber loss
                               caps each residual's influence beyond this
                               threshold instead of letting it dominate.

    Returns (refined_rotations, refined_translations, refined_points) -
    refined_rotations/refined_translations cover all K poses (fixed ones
    returned unchanged), same convention as the rotations/translations input.
    """
    n_poses = len(rotations)
    n_points = len(points)

    if fixed_poses is None:
        fixed_poses = np.zeros(n_poses, dtype=bool)
        fixed_poses[0] = True
    fixed_poses = np.asarray(fixed_poses, dtype=bool)

    free_idx = np.where(~fixed_poses)[0]
    fixed_idx = np.where(fixed_poses)[0]
    n_free_poses = len(free_idx)
    pose_to_free = -np.ones(n_poses, dtype=int)
    pose_to_free[free_idx] = np.arange(n_free_poses)

    obs = np.asarray(observations, dtype=np.float64)
    kf_idx = obs[:, 0].astype(int)
    pt_idx = obs[:, 1].astype(int)
    pixels = obs[:, 2:4]

    fixed_rvecs = np.array([cv2.Rodrigues(rotations[i])[0].ravel() for i in fixed_idx]).reshape(-1, 3)
    fixed_tvecs = np.array([np.asarray(translations[i]).ravel() for i in fixed_idx]).reshape(-1, 3)

    free_rvecs0 = np.array([cv2.Rodrigues(rotations[i])[0].ravel() for i in free_idx]).reshape(-1, 3)
    free_tvecs0 = np.array([np.asarray(translations[i]).ravel() for i in free_idx]).reshape(-1, 3)
    x0 = np.concatenate([free_rvecs0.ravel(), free_tvecs0.ravel(), points.ravel()])

    def residuals(x):
        free_rvecs = x[:n_free_poses * 3].reshape(n_free_poses, 3)
        free_tvecs = x[n_free_poses * 3:n_free_poses * 6].reshape(n_free_poses, 3)
        pts = x[n_free_poses * 6:].reshape(n_points, 3)

        all_rvecs = np.empty((n_poses, 3))
        all_tvecs = np.empty((n_poses, 3))
        all_rvecs[fixed_idx] = fixed_rvecs
        all_tvecs[fixed_idx] = fixed_tvecs
        all_rvecs[free_idx] = free_rvecs
        all_tvecs[free_idx] = free_tvecs

        res = np.empty((len(obs), 2))
        for k in range(n_poses):
            sel = kf_idx == k
            if not np.any(sel):
                continue
            proj, _ = cv2.projectPoints(
                pts[pt_idx[sel]], all_rvecs[k], all_tvecs[k], camera_matrix, None
            )
            res[sel] = proj.reshape(-1, 2) - pixels[sel]
        return res.ravel()

    # This is a refinement of an already-good PnP/triangulation estimate, not a
    # from-scratch solve - loose tolerances and a low iteration cap let it exit
    # quickly once "good enough" rather than grinding toward high-precision
    # convergence, which is what made this visibly lag at every keyframe.
    sparsity = _build_sparsity(kf_idx, pt_idx, pose_to_free, n_free_poses, n_points)
    result = least_squares(
        residuals, x0, jac_sparsity=sparsity, method="trf",
        loss="huber", f_scale=outlier_threshold_px,
        max_nfev=max_nfev, ftol=ftol, xtol=xtol, verbose=0,
    )

    free_rvecs = result.x[:n_free_poses * 3].reshape(n_free_poses, 3)
    free_tvecs = result.x[n_free_poses * 3:n_free_poses * 6].reshape(n_free_poses, 3)
    refined_points = result.x[n_free_poses * 6:].reshape(n_points, 3)

    refined_rotations = list(rotations)
    refined_translations = list(translations)
    for local_i, orig_i in enumerate(free_idx):
        refined_rotations[orig_i] = cv2.Rodrigues(free_rvecs[local_i])[0]
        # .copy(): free_tvecs is a view into result.x (the solver's full
        # parameter vector, poses AND every refined point) - storing the
        # view as-is would keep that whole buffer alive for as long as
        # this one keyframe's pose is kept, for every BA call ever run
        # (keyframe_poses entries live for the rest of the program).
        refined_translations[orig_i] = free_tvecs[local_i].reshape(3, 1).copy()

    return refined_rotations, refined_translations, refined_points
