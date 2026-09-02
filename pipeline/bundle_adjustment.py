"""
Lightweight sliding-window local bundle adjustment.

Jointly refines a window of recent keyframe poses and the map points they
observe, minimizing total reprojection error. Without this, every triangulated
point is trusted as permanent ground truth the moment it's added, so small
pose errors get baked into the map and compound over time (see NOTES.md). The
oldest keyframe in the window is held fixed as a gauge anchor - otherwise the
whole optimization is free to drift/rotate/translate as a rigid whole with
nothing to anchor "correct" against.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _build_sparsity(kf_idx, pt_idx, n_free_poses, n_points, first_free):
    """
    Boolean sparsity pattern (2*n_obs x n_vars): which parameters affect which
    residuals. Each observation only depends on its own keyframe's 6 pose
    params (if free) and its own point's 3 coordinates - never anything else.
    Without this, scipy's finite-difference Jacobian perturbs every one of the
    (often 1000+) point variables and re-evaluates the FULL residual vector
    each time, which is what makes an unconstrained call freeze/crawl.
    """
    n_obs = len(kf_idx)
    n_vars = n_free_poses * 6 + n_points * 3
    sparsity = lil_matrix((2 * n_obs, n_vars), dtype=bool)

    for j in range(n_obs):
        rows = slice(2 * j, 2 * j + 2)
        k = kf_idx[j]
        if k >= first_free:
            local_k = k - first_free
            sparsity[rows, local_k * 3: local_k * 3 + 3] = True
            sparsity[rows, n_free_poses * 3 + local_k * 3: n_free_poses * 3 + local_k * 3 + 3] = True

        p = pt_idx[j]
        point_col = n_free_poses * 6 + p * 3
        sparsity[rows, point_col: point_col + 3] = True

    return sparsity.tocsr()


def local_bundle_adjustment(rotations, translations, points, observations, camera_matrix,
                             fix_first_pose=True, outlier_threshold_px=3.0, max_nfev=1000,
                             ftol=1e-4, xtol=1e-4):
    """
    rotations, translations - lists of length K: world-to-camera poses
                               (X_cam = R @ X_world + t) for K keyframes.
    points                  - Nx3 array of map points observed by this window.
    observations            - list of (keyframe_idx, point_idx, x, y) - a pixel
                               observation of points[point_idx] in keyframe
                               rotations[keyframe_idx]/translations[keyframe_idx].
    camera_matrix           - shared 3x3 intrinsics.
    fix_first_pose          - hold rotations[0]/translations[0] fixed as the
                               gauge anchor for this window.
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

    Returns (refined_rotations, refined_translations, refined_points).
    """
    n_poses = len(rotations)
    n_points = len(points)
    first_free = 1 if fix_first_pose else 0
    n_free_poses = n_poses - first_free

    obs = np.asarray(observations, dtype=np.float64)
    kf_idx = obs[:, 0].astype(int)
    pt_idx = obs[:, 1].astype(int)
    pixels = obs[:, 2:4]

    fixed_rvec = cv2.Rodrigues(rotations[0])[0].ravel() if fix_first_pose else None
    fixed_tvec = np.asarray(translations[0]).ravel() if fix_first_pose else None

    free_rvecs0 = np.array([cv2.Rodrigues(R)[0].ravel() for R in rotations[first_free:]])
    free_tvecs0 = np.array([np.asarray(t).ravel() for t in translations[first_free:]])
    x0 = np.concatenate([free_rvecs0.ravel(), free_tvecs0.ravel(), points.ravel()])

    def residuals(x):
        free_rvecs = x[:n_free_poses * 3].reshape(n_free_poses, 3)
        free_tvecs = x[n_free_poses * 3:n_free_poses * 6].reshape(n_free_poses, 3)
        pts = x[n_free_poses * 6:].reshape(n_points, 3)

        all_rvecs = np.vstack([fixed_rvec, free_rvecs]) if fix_first_pose else free_rvecs
        all_tvecs = np.vstack([fixed_tvec, free_tvecs]) if fix_first_pose else free_tvecs

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
    sparsity = _build_sparsity(kf_idx, pt_idx, n_free_poses, n_points, first_free)
    result = least_squares(
        residuals, x0, jac_sparsity=sparsity, method="trf",
        loss="huber", f_scale=outlier_threshold_px,
        max_nfev=max_nfev, ftol=ftol, xtol=xtol, verbose=0,
    )

    free_rvecs = result.x[:n_free_poses * 3].reshape(n_free_poses, 3)
    free_tvecs = result.x[n_free_poses * 3:n_free_poses * 6].reshape(n_free_poses, 3)
    refined_points = result.x[n_free_poses * 6:].reshape(n_points, 3)

    refined_rotations = list(rotations[:first_free])
    refined_translations = list(translations[:first_free])
    for rvec, tvec in zip(free_rvecs, free_tvecs):
        refined_rotations.append(cv2.Rodrigues(rvec)[0])
        refined_translations.append(tvec.reshape(3, 1))

    return refined_rotations, refined_translations, refined_points