"""
Sim(3) primitives for Loop Closing (#55, paper §VII): a closed-form
similarity-transform solver (Umeyama), a RANSAC wrapper around it for
geometric verification, and an Essential Graph pose-graph optimizer that
corrects accumulated drift by adjusting per-keyframe Sim(3) poses rather than
re-triangulating anything.

Sim(3) - not just SE(3) - is needed here because this pipeline's monocular
map has a single, arbitrary scale established once at bootstrap, but nothing
keeps later parts of the map at exactly that same effective scale: ordinary
per-frame PnP + incremental triangulation + local BA only ever enforce local
consistency, so small systematic errors compound over a long trajectory the
same way orientation/position drift do (paper §VII-B - "we perform a
pose-graph optimization... to correct the scale drift"). A loop closure's
geometric verification (mapping._verify_loop_closure) discovers exactly this
kind of discrepancy between two, by-construction-non-covisible regions of one
map, and the pose-graph optimization below (mapping._apply_loop_correction)
distributes the correction back across every keyframe in between.

Sim(3) convention used throughout this module: a transform is a (s, R, t)
tuple acting on a 3-vector X as S(X) = s * (R @ X) + t.
"""

import cv2
import numpy as np


def sim3_compose(a, b):
    """B ∘ A - apply Sim(3) `a` first, then `b`. Matches mapping.compose_pose's
    own R_rel @ R_pos convention (the second argument is the "later" one)."""
    s_a, R_a, t_a = a
    s_b, R_b, t_b = b
    t_a = np.asarray(t_a).reshape(3, 1)
    t_b = np.asarray(t_b).reshape(3, 1)
    s = s_a * s_b
    R = R_b @ R_a
    t = s_b * (R_b @ t_a) + t_b
    return s, R, t


def sim3_inverse(a):
    """The Sim(3) transform undoing `a`: sim3_compose(a, sim3_inverse(a))
    is the identity (s=1, R=I, t=0)."""
    s, R, t = a
    t = np.asarray(t).reshape(3, 1)
    inv_s = 1.0 / s
    inv_R = R.T
    inv_t = -inv_s * (inv_R @ t)
    return inv_s, inv_R, inv_t


def umeyama_alignment(src, dst):
    """
    Closed-form similarity transform (s, R, t) minimizing
    sum ||dst_i - (s * R @ src_i + t)||^2 (Umeyama, 1991 - Horn's absolute
    orientation problem extended to also solve for scale).

    src, dst - Nx3 arrays, N >= 3. Raises np.linalg.LinAlgError if src's
    points are (numerically) coincident - callers doing RANSAC over minimal
    samples should treat that as a degenerate sample, same as any other
    failed minimal-set fit elsewhere in this codebase (e.g.
    pose.homography_score's own singular-H handling).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    var_src = (src_c ** 2).sum() / n
    if var_src < 1e-12:
        raise np.linalg.LinAlgError("umeyama_alignment: degenerate (coincident) source points")
    s = float(np.trace(np.diag(D) @ S) / var_src)
    t = (mu_dst - s * (R @ mu_src)).reshape(3, 1)
    return s, R, t


def estimate_sim3_ransac(src_points, dst_points, dst_pixels, dst_R, dst_t, camera_matrix,
                          reproj_threshold_px=8.0, min_inliers=8, max_iterations=500,
                          rng=None):
    """
    RANSAC-verified Sim(3) aligning src_points onto dst_points (Nx3,
    already-matched 3D-3D correspondences from two different keyframes'
    own observed map points - see mapping._verify_loop_closure), scored the
    same way the rest of this codebase scores geometry: reprojection error
    in pixels, not a raw 3D distance (which would need an arbitrary,
    map-scale-dependent threshold). Each hypothesis's transformed src
    points are projected with dst_R/dst_t (the keyframe pose dst_points/
    dst_pixels were actually observed from) and compared against
    dst_pixels - the real, already-known pixel location of each matched
    dst point in that keyframe.

    Only 3 points are needed per hypothesis (umeyama_alignment's minimum),
    but min_inliers is deliberately looser than ordinary PnP's own bar (see
    mapping.py's --loop-min-inliers --help) - #50 already showed literal
    correspondence across a large appearance change is inherently sparse
    here, and a loop constraint only needs to be roughly right (the
    Essential Graph optimization afterward is what actually distributes the
    correction, unlike ordinary tracking's PnP pose, which IS the
    trajectory).

    Returns (s, R, t, inlier_mask) - refit once from every inlier the
    winning hypothesis found, not just its own minimal sample - or None if
    no hypothesis reaches min_inliers.
    """
    rng = np.random.default_rng() if rng is None else rng
    src_points = np.asarray(src_points, dtype=np.float64)
    dst_points = np.asarray(dst_points, dtype=np.float64)
    dst_pixels = np.asarray(dst_pixels, dtype=np.float64)
    n = len(src_points)
    if n < 3:
        return None

    rvec_dst, _ = cv2.Rodrigues(dst_R)
    best_inliers = None

    def _score(s, R, t):
        transformed = s * (R @ src_points.T).T + t.reshape(1, 3)
        proj, _ = cv2.projectPoints(transformed, rvec_dst, dst_t, camera_matrix, None)
        err = np.linalg.norm(proj.reshape(-1, 2) - dst_pixels, axis=1)
        return err < reproj_threshold_px

    for _ in range(max_iterations):
        sample = rng.choice(n, size=3, replace=False)
        try:
            s, R, t = umeyama_alignment(src_points[sample], dst_points[sample])
        except np.linalg.LinAlgError:
            continue

        inliers = _score(s, R, t)
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < min_inliers:
        return None

    try:
        s, R, t = umeyama_alignment(src_points[best_inliers], dst_points[best_inliers])
    except np.linalg.LinAlgError:
        return None

    final_inliers = _score(s, R, t)
    if final_inliers.sum() < min_inliers:
        return None
    return s, R, t, final_inliers


def optimize_essential_graph(n_keyframes, initial_poses, edges, fixed_keyframes,
                              scale_weight=10.0, max_nfev=2000, ftol=1e-8, xtol=1e-8):
    """
    Pose-graph optimization (paper §VII-B) over per-keyframe Sim(3) poses:
    minimizes, over every edge (i, j, s_ij, R_ij, t_ij) in `edges`, the
    discrepancy between that edge's own measured relative transform (i's
    camera frame -> j's) and the relative transform implied by the two
    keyframes' CURRENT pose estimates during the solve.

    See mapping._apply_loop_correction for how edges are built: ordinary
    sequential/covisibility edges are measured from the keyframes' current,
    pre-correction poses (so the graph stays locally rigid where nothing
    new was learned), while the loop edge(s) are measured from geometric
    verification instead - the ONE piece of new information that actually
    drives a correction, which every other edge's own rigidity then
    distributes across the trajectory.

    n_keyframes/initial_poses cover every keyframe index (0..n_keyframes-1)
    even if only a subset appear in `edges` - initial_poses[k] is a
    (s, R, t) Sim(3) tuple (s=1 for a not-yet-corrected keyframe, since this
    pipeline never otherwise tracks a per-keyframe scale factor - see
    mapping.py's --loop-closing --help).

    fixed_keyframes are held fixed - callers should always include keyframe
    0, the map's sole gauge anchor everywhere else in this codebase (see
    bundle_adjustment.local_bundle_adjustment's own fixed_poses convention).

    scale_weight scales the log-scale residual relative to the rotation/
    translation ones (a fixed internal constant, not exposed via CLI: the
    residual vector mixes rotation (radians), translation (map units) and
    log-scale terms whose natural magnitudes aren't directly comparable,
    and this is a reasonable default rather than one exhaustively tuned per
    sequence).

    Returns a length-n_keyframes list of optimized (s, R, t) tuples (fixed
    keyframes returned unchanged).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    fixed = set(fixed_keyframes)
    free = [k for k in range(n_keyframes) if k not in fixed]
    free_index = {k: i for i, k in enumerate(free)}
    n_free = len(free)

    def pack(poses):
        x = np.empty(n_free * 7)
        for k, i in free_index.items():
            s, R, t = poses[k]
            rvec, _ = cv2.Rodrigues(R)
            x[i * 7:i * 7 + 3] = rvec.ravel()
            x[i * 7 + 3:i * 7 + 6] = np.asarray(t).ravel()
            x[i * 7 + 6] = np.log(s)
        return x

    def unpack(x):
        poses = list(initial_poses)
        for k, i in free_index.items():
            rvec = x[i * 7:i * 7 + 3]
            t = x[i * 7 + 3:i * 7 + 6].reshape(3, 1)
            s = float(np.exp(x[i * 7 + 6]))
            R, _ = cv2.Rodrigues(rvec)
            poses[k] = (s, R, t)
        return poses

    if n_free == 0 or not edges:
        return list(initial_poses)

    x0 = pack(initial_poses)

    def residuals(x):
        poses = unpack(x)
        res = np.empty((len(edges), 7))
        for e, (i, j, s_ij, R_ij, t_ij) in enumerate(edges):
            predicted_j = sim3_compose(poses[i], (s_ij, R_ij, t_ij))
            error = sim3_compose(sim3_inverse(predicted_j), poses[j])
            s_e, R_e, t_e = error
            rvec_e, _ = cv2.Rodrigues(R_e)
            res[e, :3] = rvec_e.ravel()
            res[e, 3:6] = t_e.ravel()
            res[e, 6] = scale_weight * np.log(max(s_e, 1e-9))
        return res.ravel()

    sparsity = lil_matrix((7 * len(edges), 7 * n_free), dtype=bool)
    for e, (i, j, *_rest) in enumerate(edges):
        rows = slice(7 * e, 7 * e + 7)
        if i in free_index:
            fi = free_index[i]
            sparsity[rows, fi * 7:fi * 7 + 7] = True
        if j in free_index:
            fj = free_index[j]
            sparsity[rows, fj * 7:fj * 7 + 7] = True

    result = least_squares(
        residuals, x0, jac_sparsity=sparsity.tocsr(), method="trf",
        max_nfev=max_nfev, ftol=ftol, xtol=xtol, verbose=0,
    )
    return unpack(result.x)
