"""
Persistent sparse map: bootstrap once via two-view triangulation, then track
pose against the existing map via PnP on every subsequent frame (motion-
predicted guided matching, see Map.match_against_guided), inserting a
keyframe - and extending the map - only once parallax has accumulated far
enough since the last one.

This replaces independent keyframe-to-keyframe two-view pose chaining (which
compounds scale drift, since cv2.recoverPose always returns a unit-length
translation with no memory of the previous segment's scale) with a single
consistent scale established at bootstrap: every later keyframe's pose is
solved directly against the map's own already-scaled 3D points, and only
genuinely new points are triangulated and added to that same map.
"""

import os
from typing import NamedTuple

import cv2
import numpy as np

from pipeline.pose import camera_center

# Precomputed bit-count per byte value, so Hamming distance between ORB
# descriptors (32 packed bytes each) can be computed as a vectorized
# XOR + table lookup instead of a per-descriptor cv2.norm call - needed to
# keep match_against_guided's per-map-point candidate scoring cheap.
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hamming_distances(query_desc, candidate_descs):
    """Hamming distance from one descriptor to each row of candidate_descs."""
    xor = np.bitwise_xor(candidate_descs, query_desc)
    return _POPCOUNT_TABLE[xor].sum(axis=1, dtype=np.int32)


class KeyframePose(NamedTuple):
    """One accepted keyframe's world-to-camera pose (X_cam = R @ X_world + t)
    plus its source frame's timestamp - the raw material --trajectory-output
    and the live/final trajectory plots are built from."""
    R: np.ndarray
    t: np.ndarray
    timestamp: float


class Map:
    """A growing set of 3D points, each tied to the ORB descriptor of its most
    recent observation so future frames can be matched against it directly.

    New points start out unconfirmed ("provisional"): a whole batch of new
    points triangulated from a single two-view pair all share that one pair's
    pose estimate, so if that pose has any systematic error, every point in
    the batch inherits the same bias - no per-point sanity filter (parallax
    angle, depth-vs-baseline) can catch this, since the batch can look
    perfectly self-consistent internally while still being collectively
    wrong. Provisional points are excluded from PnP pose estimation until
    they've accumulated `required_confirmations` independent re-observations
    (see `confirm`), so a bad batch can't influence the pose that would
    otherwise confirm its own siblings, and a single lucky coincidental
    reprojection isn't enough on its own to promote a point.
    """

    def __init__(self, required_confirmations=2):
        self.points = np.empty((0, 3), dtype=np.float64)
        self.descriptors = np.empty((0, 32), dtype=np.uint8)
        self.confirmation_count = np.empty(0, dtype=np.int32)
        self.required_confirmations = required_confirmations

    def __len__(self):
        return len(self.points)

    @property
    def confirmed(self):
        return self.confirmation_count >= self.required_confirmations

    @property
    def n_confirmed(self):
        return int(self.confirmed.sum())

    def add_points(self, points_3d, descriptors, confirmed=False):
        if len(points_3d) == 0:
            return
        n = len(points_3d)
        self.points = np.vstack([self.points, points_3d])
        self.descriptors = np.vstack([self.descriptors, descriptors])
        initial_count = self.required_confirmations if confirmed else 0
        self.confirmation_count = np.concatenate(
            [self.confirmation_count, np.full(n, initial_count, dtype=np.int32)]
        )

    def confirm(self, indices):
        """Register one independent re-observation for these points - they
        become 'confirmed' once they've accumulated required_confirmations."""
        self.confirmation_count[indices] += 1

    def match_against(self, desc, ratio=0.75, mask=None):
        """
        Match frame descriptors against a subset of the map (all points by
        default, or a boolean mask over self.points - e.g. self.confirmed,
        or ~self.confirmed to check provisional points specifically).
        Returns (map_indices, frame_indices) - map_indices are original
        (stable) indices, not positions within the subset.
        """
        from pipeline.features import match_descriptors

        subset = np.where(mask)[0] if mask is not None else np.arange(len(self))
        if len(subset) == 0 or desc is None or len(desc) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        matches = match_descriptors(self.descriptors[subset], desc, ratio)
        map_indices = subset[[m.queryIdx for m in matches]]
        frame_indices = np.array([m.trainIdx for m in matches], dtype=int)
        return map_indices, frame_indices

    def match_against_guided(self, kp, desc, camera_matrix, R_pred, t_pred,
                              window=25.0, ratio=0.75, mask=None):
        """
        Guided variant of match_against: project each map point (subset) into
        the frame using a predicted pose, and only consider frame keypoints
        within `window` pixels of that projection as candidates for its
        descriptor match, instead of searching every keypoint in the frame.
        This both narrows ambiguity (fewer, geometrically-plausible
        candidates per point) and is cheaper than a full brute-force pass,
        since a cKDTree radius query replaces an all-pairs search.

        Points that predict behind the camera, or outside `window` of every
        keypoint, contribute no match. Returns (map_indices, frame_indices),
        same convention as match_against.
        """
        from scipy.spatial import cKDTree

        subset = np.where(mask)[0] if mask is not None else np.arange(len(self))
        if len(subset) == 0 or desc is None or len(desc) == 0 or len(kp) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        rvec_pred, _ = cv2.Rodrigues(R_pred)
        proj, _ = cv2.projectPoints(self.points[subset], rvec_pred, t_pred, camera_matrix, None)
        proj = proj.reshape(-1, 2)

        cam_pts = (R_pred @ self.points[subset].T).T + t_pred.ravel()
        in_front = cam_pts[:, 2] > 0

        frame_pts = np.float32([k.pt for k in kp])
        tree = cKDTree(frame_pts)
        neighbor_lists = tree.query_ball_point(proj, r=window)

        map_indices = []
        frame_indices = []
        for local_i, neighbors in enumerate(neighbor_lists):
            if not in_front[local_i] or not neighbors:
                continue
            cand = np.array(neighbors)
            cand_desc = desc[cand]
            dists = _hamming_distances(self.descriptors[subset[local_i]], cand_desc)
            order = np.argsort(dists)
            if len(order) >= 2 and dists[order[0]] >= ratio * dists[order[1]]:
                continue
            map_indices.append(subset[local_i])
            frame_indices.append(int(cand[order[0]]))

        return np.array(map_indices, dtype=int), np.array(frame_indices, dtype=int)


def estimate_pose_pnp(object_points, image_points, camera_matrix):
    """
    Solve a world-to-camera pose (R, t) via PnP against known 3D map points:
      X_cam = R @ X_world + t   (same convention used everywhere else)

    Returns (R, t, inlier_mask) or None if there aren't enough correspondences
    or PnP fails to find a consistent pose.
    """
    if len(object_points) < 6:
        return None

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points.astype(np.float64), image_points.astype(np.float64),
        camera_matrix, None,
        reprojectionError=4.0, confidence=0.999, iterationsCount=200,
    )
    if not ok or inliers is None or len(inliers) < 6:
        return None

    R, _ = cv2.Rodrigues(rvec)
    inlier_mask = np.zeros(len(object_points), dtype=bool)
    inlier_mask[inliers.ravel()] = True
    return R, tvec, inlier_mask


def _run_local_ba(keyframe_poses, keyframe_observations, sparse_map, camera_matrix, window,
                   max_points=300, max_nfev=1000, ftol=1e-4, xtol=1e-4):
    """
    Compute a refinement of the last `window` keyframes' poses and the map
    points they observe - WITHOUT applying it. BA can itself occasionally
    produce an implausible pose if a window is poorly conditioned (sparse or
    noisy shared points), so the caller is expected to sanity-check the
    result (see _apply_ba_result) before committing to it, the same way a
    raw PnP pose is checked before being accepted.

    Returns None if there wasn't enough data for a meaningful refinement,
    otherwise (start, refined_rotations, refined_translations, point_ids,
    refined_points) - start is keyframe_poses' index for refined_rotations[0].

    If the window observes more than max_points points (common right after a
    keyframe that added a lot of new structure at once), only the most
    recently added ones are kept - they're the ones most relevant to
    correcting recent drift, and this bounds the per-call cost so BA doesn't
    visibly stall the pipeline at every keyframe. Pass max_points=None to
    disable the cap entirely (see _run_global_ba, which passes the whole
    trajectory as its "window" and needs every point kept, not just the
    most recent).
    """
    from pipeline.bundle_adjustment import local_bundle_adjustment

    if len(keyframe_poses) < 2:
        return None

    start = max(0, len(keyframe_poses) - window)
    window_obs = keyframe_observations[start:]

    point_ids = sorted({obs[0] for kf_obs in window_obs for obs in kf_obs})
    if len(point_ids) < 10:
        return None  # not enough constraints for a meaningful refinement

    if max_points is not None and len(point_ids) > max_points:
        # Prefer the most recently added points, but never let this starve a
        # keyframe down to too few observations - an under-constrained pose
        # (too few residuals for its 6 DOF) is far worse than a slightly
        # larger optimization, and is what caused wild trajectory jumps here.
        min_obs_per_keyframe = 15
        keep = set(point_ids[-max_points:])
        for kf_obs in window_obs:
            if not kf_obs:
                continue
            target = min(min_obs_per_keyframe, len(kf_obs))
            kept_here = sum(1 for o in kf_obs if o[0] in keep)
            if kept_here < target:
                candidates = sorted({o[0] for o in kf_obs} - keep, reverse=True)
                keep.update(candidates[:target - kept_here])
        point_ids = sorted(keep)
        window_obs = [[o for o in kf_obs if o[0] in keep] for kf_obs in window_obs]

    id_to_local = {pid: i for i, pid in enumerate(point_ids)}
    local_points = sparse_map.points[point_ids].copy()

    local_observations = []
    for local_kf_idx, kf_obs in enumerate(window_obs):
        for pid, x, y in kf_obs:
            local_observations.append((local_kf_idx, id_to_local[pid], x, y))

    rotations = [pose[0] for pose in keyframe_poses[start:]]
    translations = [pose[1] for pose in keyframe_poses[start:]]

    refined_rot, refined_trans, refined_pts = local_bundle_adjustment(
        rotations, translations, local_points, local_observations, camera_matrix,
        fix_first_pose=True, max_nfev=max_nfev, ftol=ftol, xtol=xtol,
    )

    return start, refined_rot, refined_trans, point_ids, refined_pts


def _run_global_ba(keyframe_poses, keyframe_observations, sparse_map, camera_matrix,
                    max_nfev=8000, ftol=1e-6, xtol=1e-6):
    """
    Full BA (paper Appendix; used offline in §VIII-E/Table VI): the same
    optimizer local BA uses, scoped to every keyframe and every map point
    instead of a sliding window - a thin wrapper around _run_local_ba rather
    than a second optimizer, since local_bundle_adjustment already handles an
    arbitrary keyframe/point set. window=len(keyframe_poses) makes
    _run_local_ba's start index 0 (the whole trajectory); max_points=None
    disables the point cap that exists only to bound per-keyframe cost during
    tracking, which doesn't apply to a one-shot end-of-run refinement.

    A full map needs far more solver iterations to converge than a small
    local window, hence the higher max_nfev default than local_bundle_adjustment's.
    ftol/xtol are tightened well past local BA's default (1e-4) for the same
    reason: this pipeline runs local BA continuously throughout tracking, so
    by the time a one-shot global pass runs, the map can already look
    "converged" to a loose relative tolerance within just a few iterations
    without local BA's incremental, overlapping-window refinements having
    actually reached a true joint optimum over the whole trajectory -
    confirmed empirically (the loose 1e-4 default terminated in ~20s with a
    byte-identical trajectory on freiburg1_xyz).
    """
    return _run_local_ba(
        keyframe_poses, keyframe_observations, sparse_map, camera_matrix,
        window=len(keyframe_poses), max_points=None, max_nfev=max_nfev,
        ftol=ftol, xtol=xtol,
    )


def _reprojection_error_stats(keyframe_poses, keyframe_observations, sparse_map, camera_matrix):
    """
    Mean/median/max reprojection error (px) across every (keyframe, point)
    observation in the map - used to sanity-check a full BA result directly
    (did it actually reduce error, not just "the solver returned"), since
    full BA operates on the whole trajectory where an implausible-pose
    check on the latest keyframe alone wouldn't catch a bad correction to an
    older one.
    """
    errors = []
    for kf_idx, kf_obs in enumerate(keyframe_observations):
        if not kf_obs:
            continue
        R, t, _ = keyframe_poses[kf_idx]
        pids = np.array([o[0] for o in kf_obs])
        pixels = np.array([(o[1], o[2]) for o in kf_obs])
        rvec, _ = cv2.Rodrigues(R)
        proj, _ = cv2.projectPoints(sparse_map.points[pids], rvec, t, camera_matrix, None)
        errors.append(np.linalg.norm(proj.reshape(-1, 2) - pixels, axis=1))
    if not errors:
        return None
    errors = np.concatenate(errors)
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }


def _apply_ba_result(keyframe_poses, sparse_map, ba_result):
    start, refined_rot, refined_trans, point_ids, refined_pts = ba_result
    for i, (R_ref, t_ref) in enumerate(zip(refined_rot, refined_trans)):
        keyframe_poses[start + i] = keyframe_poses[start + i]._replace(R=R_ref, t=t_ref)
    sparse_map.points[point_ids] = refined_pts


def _validate_and_apply_ba(ba_result, keyframe_poses, sparse_map, recent_step_sizes,
                            fallback_R, fallback_t, max_plausible_rotation, max_step_ratio):
    """
    Apply a proposed BA refinement only if EVERY pose in the window still
    passes the same rotation/step plausibility checks used for a fresh PnP
    pose - not just the latest one. BA can produce a plausible-looking latest
    pose while quietly pushing an implausible correction into an OLDER
    keyframe still in the window instead (jointly optimized, so error can
    land anywhere); checking only the latest pose would let that slip through
    silently; it would only show up later once that keyframe's rewritten
    position is reflected in the trajectory/plot, not at the moment of
    rejection.

    max_plausible_rotation/max_step_ratio are passed explicitly rather than
    read off `args` directly, because they mean something different for a
    local-BA window than for a full/global one: `recent_step_sizes` is a
    *per-frame tracking motion* statistic (typically cm-scale), which is the
    right yardstick for "did local BA nudge a just-tracked keyframe somewhere
    implausible" but the wrong one for "did full BA's cumulative drift
    correction move an old keyframe further than one frame's worth of
    motion" - the latter can legitimately be much larger without being wrong
    (that's the entire point of running full BA). Local-BA call sites pass
    args.max_plausible_rotation/args.max_step_ratio unchanged; the
    global-BA call site passes its own, looser thresholds.

    Returns the (R, t) pose to use going forward: the BA-refined latest pose
    if the whole window is accepted, otherwise fallback_R/fallback_t (the
    pre-BA pose) with nothing in keyframe_poses/sparse_map touched at all.
    """
    from pipeline.pose import rotation_angle_deg

    if ba_result is None:
        return fallback_R, fallback_t

    start, refined_rot, refined_trans, _, _ = ba_result

    for i, (new_R, new_t) in enumerate(zip(refined_rot, refined_trans)):
        old_R, old_t, _ = keyframe_poses[start + i]
        rot_change = rotation_angle_deg(new_R @ old_R.T)
        step_change = np.linalg.norm(
            camera_center(new_R, new_t) - camera_center(old_R, old_t)
        )
        implausible = rot_change > max_plausible_rotation or (
            len(recent_step_sizes) >= 5
            and step_change > max_step_ratio * np.median(recent_step_sizes)
        )
        if implausible:
            print(f"    [BA result rejected: implausible pose change at window index {i} "
                  f"(rotation={rot_change:.1f}deg, step={step_change:.2f})]")
            return fallback_R, fallback_t

    new_R, new_t = refined_rot[-1], refined_trans[-1]

    _apply_ba_result(keyframe_poses, sparse_map, ba_result)
    return new_R, new_t


def _run_depth_densify(frame_image, R_pos, t_pos, sparse_map, map_indices, image_points,
                        pnp_inlier_mask, camera_matrix, depth_estimator, depth_rows, scan_stride):
    """
    Estimates ML depth for this keyframe, fits it (scale + shift) against
    the confirmed map points PnP just matched (map_indices/image_points,
    restricted to pnp_inlier_mask - the same trusted set the pose itself was
    solved against), then back-projects a sample of scanline pixels into
    world points using that fit.

    Purely a visual sanity check for now (see NOTES.md's v2 plan) - the
    caller must not feed the returned points into sparse_map/PnP/BA.

    Returns (new_world_points, raw_depth, background_mask) - new_world_points
    is empty if there wasn't enough data for a stable fit; raw_depth and
    background_mask are always returned so the caller can still show them.
    """
    from pipeline.depth_ml import backproject_pixels, detect_background_mask, fit_disparity_scale_shift

    raw_depth = depth_estimator.estimate(frame_image)

    # a/(disp-b) is a reciprocal map (see below), so a small amount of the
    # model's per-pixel disparity noise gets amplified nonlinearly once
    # inverted - increasingly so at larger distances. A flat, receding
    # surface (e.g. a half-open door) can come out visibly warped even
    # though the underlying noise is roughly uniform across it. Median
    # blur (edge-preserving, unlike Gaussian) suppresses that noise before
    # it gets amplified, rather than cleaning up already-exploded Z values
    # afterward - used for both the calibration fit and scanline sampling
    # below so they stay in the same noise regime; the raw (unsmoothed)
    # map is still what gets displayed/returned.
    smoothed_depth = cv2.medianBlur(raw_depth.astype(np.float32), 5)

    # Excludes new points from open floor/wall - a scanline crossing
    # smoothly-receding floor produces a full-width strip of points every
    # keyframe, which is pure repeated clutter (no object information)
    # rather than useful densification. Deliberately a blacklist, not a
    # whitelist: an earlier version tried to positively identify and only
    # allow discrete objects, which ended up rejecting too many real ones -
    # excluding just the (easier to identify reliably) background is more
    # forgiving, since anything not confidently background still gets
    # through. Only gates which pixels get turned into new points below -
    # the calibration fit still uses every confirmed point regardless of
    # surface. Runs on the camera image, not the depth map - see
    # detect_background_mask.
    background_mask = detect_background_mask(frame_image)

    calib_pixels = image_points[pnp_inlier_mask]
    calib_map_idx = map_indices[pnp_inlier_mask]
    cam_pts = (R_pos @ sparse_map.points[calib_map_idx].T).T + t_pos.ravel()
    inv_depth = 1.0 / cam_pts[:, 2]
    us_i = np.clip(calib_pixels[:, 0].round().astype(int), 0, raw_depth.shape[1] - 1)
    vs_i = np.clip(calib_pixels[:, 1].round().astype(int), 0, raw_depth.shape[0] - 1)
    disparity = smoothed_depth[vs_i, us_i]

    fit = fit_disparity_scale_shift(disparity, inv_depth)
    if fit is None:
        print(f"    [depth-densify: only {len(disparity)} confirmed points visible - "
              f"skipping (need >= 10 for a stable fit)]")
        return np.empty((0, 3)), raw_depth, background_mask

    a, b = fit

    # A degenerate fit (a close to 0 - the confirmed points barely span any
    # disparity range, e.g. a near-planar/low-depth-variety calibration set)
    # can still show a deceptively low RMSE in DISPARITY space while being
    # useless in Z space, since a/(disp-b) amplifies whatever small disparity
    # residual remains far more when a is small - a keyframe with a=2 gave a
    # disparity RMSE of ~0.1 (looks fine) but reconstructed the calibration
    # points' own known Z at 68 instead of their true ~1-50 range (garbage).
    # Checking the reconstruction directly, in Z space, catches this - the
    # same amplification that would corrupt new scanline points also shows
    # up on the calibration points themselves when the fit is this unstable.
    pred_z_calib = a / (disparity - b)
    z_rel_error = np.median(np.abs(pred_z_calib - cam_pts[:, 2]) / cam_pts[:, 2])
    if z_rel_error > 0.3:
        print(f"    [depth-densify: fit unreliable (median Z reconstruction error "
              f"{z_rel_error * 100:.0f}% on its own calibration points) - skipping]")
        return np.empty((0, 3)), raw_depth, background_mask

    # Same failure mode as triangulation's "sprinkler" artifact: z_cam =
    # a/(disp-b) is a reciprocal map, so it's only well-conditioned close to
    # the disparity range the fit was actually calibrated on - a small
    # extrapolation in disparity becomes a huge one in Z once disp
    # approaches b. Bounding z_cam by a multiplier on the calibration
    # points' own Z range (tried first) still let extrapolated points
    # through, since the reciprocal relationship means a "moderate" looking
    # Z multiplier can correspond to a disparity far outside the fit's
    # support. Restricting to the calibration set's own observed *disparity*
    # range instead rejects extrapolation directly, at its actual source.
    #
    # The raw min/max of that range is itself fragile, though: a single
    # confirmed point that's unusually far (or just noisy) sets the boundary
    # right at the edge of the range - close to b - and every scanline pixel
    # near that same edge still explodes even though it's nominally "in
    # range" (this is what kept producing near-infinite points intermittently
    # after the disparity-range clamp alone). Percentiles instead of min/max
    # keep a handful of extreme calibration points from setting the boundary.
    disp_lo, disp_hi = np.percentile(disparity, [5, 95])

    new_points = []
    for row in depth_rows:
        us = np.arange(0, raw_depth.shape[1], scan_stride)
        disp_row = smoothed_depth[row, us]
        valid = (disp_row > disp_lo) & (disp_row < disp_hi) & ~background_mask[row, us]
        if not np.any(valid):
            continue
        z_cam = a / (disp_row[valid] - b)
        cam_xyz = backproject_pixels(us[valid], np.full(int(valid.sum()), row), z_cam, camera_matrix)
        new_points.append((R_pos.T @ (cam_xyz.T - t_pos)).T)

    new_points = np.vstack(new_points) if new_points else np.empty((0, 3))
    pred_disp = a * inv_depth + b
    rmse = float(np.sqrt(np.mean((pred_disp - disparity) ** 2)))
    print(f"    [depth-densify: fit a={a:.3f} b={b:.3f} rmse={rmse:.3f} "
          f"from {len(disparity)} confirmed points, {len(new_points)} ML points sampled]")
    return new_points, raw_depth, background_mask


def _keyframe_positions(keyframe_poses):
    """Camera centers (world coordinates) for every keyframe, as an (N, 3)
    array - shared by the live-display loop and the final matplotlib plot."""
    return np.array(
        [camera_center(pose.R, pose.t) for pose in keyframe_poses]
    ).reshape(-1, 3)


def _demo():
    import argparse

    from capture.video_source import open_calibrated_source
    from pipeline.depth_ml import DepthEstimator, colorize_depth_with_background, scanline_rows
    from pipeline.features import detect_and_compute_gridded, match_descriptors
    from pipeline.pose import (
        estimate_relative_pose, rotation_angle_deg, compose_pose,
        median_parallax, predict_constant_velocity, render_trajectory,
    )
    from pipeline.trajectory import write_tum_trajectory
    from pipeline.triangulation import triangulate

    parser = argparse.ArgumentParser(
        description="Bootstrap a sparse map once, then track pose every frame via "
                     "motion-predicted guided PnP against it, inserting a keyframe "
                     "(triangulation/confirmation/BA) only once parallax accumulates"
    )
    parser.add_argument("--video", required=True,
                         help="Video file path, integer device index, or image-sequence folder "
                              "(e.g. a TUM RGB-D sequence, containing rgb.txt)")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    parser.add_argument("--n-features", type=int, default=5000)
    parser.add_argument("--grid", default="4x4",
                         help="ROWSxCOLS grid for per-cell ORB feature quotas, so a richly "
                              "textured region (e.g. a near object) can't consume the whole "
                              "feature budget and starve other regions (e.g. the background)")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold")
    parser.add_argument("--min-parallax", type=float, default=10.0,
                         help="Minimum median pixel displacement vs the reference keyframe "
                              "before this frame's tracked pose is also inserted as a new "
                              "keyframe (triangulation/confirmation/BA) - does not gate "
                              "whether a pose is estimated at all; every frame attempts "
                              "motion-predicted guided PnP against the map (px)")
    parser.add_argument("--guided-window", type=float, default=60.0,
                         help="Pixel radius around each confirmed map point's motion-"
                              "predicted projection to search for a descriptor match "
                              "during per-frame PnP tracking, replacing an unguided "
                              "full-frame search (see Map.match_against_guided)")
    parser.add_argument("--min-inliers", type=int, default=60,
                         help="Minimum bootstrap (essential matrix) pose inliers to accept a keyframe")
    parser.add_argument("--pnp-min-inliers", type=int, default=20,
                         help="Minimum PnP inliers required to accept a tracked frame's pose "
                              "(checked every frame, not just when it becomes a keyframe)")
    parser.add_argument("--max-plausible-rotation", type=float, default=15.0,
                         help="Reject a PnP pose if the relative rotation vs. the previous "
                              "tracked frame exceeds this (deg) - real handheld motion between "
                              "two close frames shouldn't produce tens of degrees of rotation; "
                              "a jump this large usually means PnP locked onto a degenerate/"
                              "ambiguous alternate solution (common with poorly depth-"
                              "distributed points) rather than that real rotation occurred")
    parser.add_argument("--min-triangulation-angle", type=float, default=1.0,
                         help="Minimum parallax angle (deg) between viewing rays to keep a "
                              "triangulated point - catches points thrown out to an "
                              "implausibly FAR depth (the 'sprinkler' artifact)")
    parser.add_argument("--confirm-reproj-error", type=float, default=4.0,
                         help="Max reprojection error (px) for a provisional point to count "
                              "as independently re-observed")
    parser.add_argument("--confirm-count", type=int, default=1,
                         help="Number of independent re-observations a provisional point "
                              "needs before being promoted to confirmed/trusted")
    parser.add_argument("--max-step-ratio", type=float, default=6.0,
                         help="Reject a PnP pose if the camera-center displacement vs. the "
                              "previous tracked frame exceeds this many multiples of the "
                              "recent median step size. Companion check to "
                              "--max-plausible-rotation: "
                              "a degenerate/ambiguous PnP solution doesn't always show up as "
                              "a rotation flip - it can instead keep a plausible rotation but "
                              "put the camera in the wrong place, which the rotation check "
                              "alone won't catch (and which then corrupts every subsequent "
                              "frame's rotation-vs-previous comparison once accepted)")
    parser.add_argument("--ba-window", type=int, default=5,
                         help="Number of most recent keyframes jointly refined by local "
                              "bundle adjustment after each new keyframe")
    parser.add_argument("--ba-every", type=int, default=1,
                         help="Only run local bundle adjustment every Nth accepted keyframe "
                              "(default: every keyframe)")
    parser.add_argument("--ba-max-points", type=int, default=300,
                         help="Cap on how many of the window's points a single BA call "
                              "refines (keeps the most recently added ones if exceeded)")
    parser.add_argument("--no-ba", action="store_true",
                         help="Disable local bundle adjustment (for comparison)")
    parser.add_argument("--global-ba-at-end", action="store_true",
                         help="After tracking completes, run one full bundle adjustment pass "
                              "over every keyframe and every map point (paper Appendix; offline "
                              "accuracy refinement per paper §VIII-E) before writing "
                              "--trajectory-output/--plot-output. Expensive relative to local "
                              "BA - a one-shot end-of-run pass, not per-keyframe")
    parser.add_argument("--global-ba-max-nfev", type=int, default=8000,
                         help="Solver iteration budget for --global-ba-at-end - a full map "
                              "needs far more than local BA's default (1000) to converge")
    parser.add_argument("--global-ba-ftol", type=float, default=1e-6,
                         help="Solver relative cost-change convergence tolerance for "
                              "--global-ba-at-end, tighter than local BA's default (1e-4) - "
                              "local BA already runs continuously during tracking, so a "
                              "loose tolerance lets a one-shot global pass falsely report "
                              "convergence after only a few iterations without reaching a "
                              "true joint optimum over the whole trajectory")
    parser.add_argument("--global-ba-xtol", type=float, default=1e-6,
                         help="Solver relative parameter-change convergence tolerance for "
                              "--global-ba-at-end - see --global-ba-ftol")
    parser.add_argument("--global-ba-max-plausible-rotation", type=float, default=90.0,
                         help="Per-keyframe rotation-change plausibility bound for "
                              "--global-ba-at-end's result, looser than --max-plausible-rotation "
                              "(15deg default, tuned for local BA/PnP) - a full-trajectory drift "
                              "correction can legitimately move an old keyframe's rotation more "
                              "than a per-frame sanity check allows; this still catches a wild "
                              "PnP-ambiguity-style flip")
    parser.add_argument("--global-ba-max-step-ratio", type=float, default=200.0,
                         help="Per-keyframe step-change plausibility bound for "
                              "--global-ba-at-end's result (as a multiple of recent per-frame "
                              "tracking step size), looser than --max-step-ratio (6.0 default, "
                              "tuned for local BA/PnP) for the same reason as "
                              "--global-ba-max-plausible-rotation - a full-trajectory correction "
                              "isn't bounded by one frame's worth of motion")
    parser.add_argument("--plot-output", default="results/map_trajectory.png")
    parser.add_argument("--trajectory-output",
                         help="Write each accepted keyframe's pose to this path in TUM format "
                              "(\"timestamp tx ty tz qx qy qz qw\", one line per keyframe) for "
                              "scoring against ground truth (e.g. with evo_ape/evo_rpe)")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable the live matches+trajectory window")
    parser.add_argument("--depth-densify", action="store_true",
                         help="At each accepted keyframe, estimate ML depth and fit it "
                              "(scale + shift) against the map's own confirmed points, "
                              "then back-project scanline samples for visual "
                              "sanity-checking. Not yet fed into pose estimation, PnP, "
                              "or bundle adjustment - plotting only")
    parser.add_argument("--model", help="Path to the depth model ONNX checkpoint "
                                         "(required if --depth-densify is set)")
    parser.add_argument("--depth-scan-rows", type=int, default=5,
                         help="Number of horizontal scanlines sampled per keyframe for "
                              "densification - horizontal only, since a vertical sweep "
                              "collapses onto a single ray in the top-down map regardless "
                              "of depth (see pipeline/depth_ml.py)")
    parser.add_argument("--depth-scan-stride", type=int, default=4,
                         help="Column stride when sampling each densification scanline")
    args = parser.parse_args()
    if args.depth_densify and not args.model:
        parser.error("--depth-densify requires --model")

    grid_rows, grid_cols = (int(v) for v in args.grid.lower().split("x"))
    sparse_map = Map(required_confirmations=args.confirm_count)

    R_pos = np.eye(3)
    t_pos = np.zeros((3, 1))

    # keyframe_poses[i] / keyframe_observations[i] describe the i-th accepted
    # keyframe: its world-to-camera pose, its source frame's timestamp, and
    # the (map_point_idx, x, y) pixel observations made in it - the raw
    # material local bundle adjustment (and --trajectory-output) use. The
    # timestamp defaults to 0.0 (not read yet) and is overwritten with the
    # real first-frame timestamp below; a source that opens but never
    # yields a frame (e.g. a truncated video) leaves it at this default
    # rather than None, so --trajectory-output can't crash formatting it.
    keyframe_poses = [KeyframePose(R_pos, t_pos, 0.0)]
    keyframe_observations = [[]]

    ref_kp = None
    ref_desc = None
    ref_image = None
    ref_R = None  # pose of the current reference keyframe (ref_kp/ref_desc's source) -
    ref_t = None  # distinct from R_pos/t_pos, which now updates every tracked frame
    recent_step_sizes = []
    # prev_pose/cur_pose_frame track the two most recent accepted per-frame
    # poses (bootstrap counts as the first) that constant-velocity prediction
    # extrapolates from, plus the frame index each was accepted at - used to
    # confirm the two are exactly one frame apart before trusting their
    # implied velocity; a gap (from one or more frames failing to track in
    # between) means that velocity actually spans more than one frame and
    # would overshoot if extrapolated another single frame ahead, so
    # prediction falls back to "no motion" instead in that case.
    prev_pose = None
    prev_pose_frame = None
    cur_pose_frame = None
    # Consecutive per-frame tracking failures since the last accepted pose -
    # widens match_against_guided's search radius (see effective_window
    # below) to compensate for the constant-velocity prediction growing
    # staler the longer it goes unconfirmed.
    n_consecutive_untracked = 0
    n_keyframes = 0
    n_tracked_only = 0
    n_skipped = 0

    # --depth-densify state: an ML depth model, run only at accepted keyframes
    # (not every frame - keyframes are already sparse). ml_points is purely
    # for visual sanity-checking (see below) - never fed into sparse_map.
    depth_estimator = DepthEstimator(args.model) if args.depth_densify else None
    ml_points = np.empty((0, 3), dtype=np.float64)
    last_depth_vis = None
    depth_rows = None

    with open_calibrated_source(args.video, args.calibration) as frames:
        K = frames.camera_matrix_undistorted
        fps = frames.fps
        delay_ms = max(1, int(1000 / fps))

        for frame in frames:
            kp, desc = detect_and_compute_gridded(
                frame.image, args.n_features, grid=(grid_rows, grid_cols)
            )

            if ref_desc is None:
                ref_kp, ref_desc, ref_image = kp, desc, frame.image
                ref_R, ref_t = R_pos, t_pos
                cur_pose_frame = frame.index
                keyframe_poses[0] = KeyframePose(R_pos, t_pos, frame.timestamp)
                continue

            matches_ref = match_descriptors(ref_desc, desc, args.ratio)

            status = "insufficient matches"
            parallax = 0.0
            is_keyframe = False
            pts1 = pts2 = np.empty((0, 2), dtype=np.float32)
            has_ref_baseline = len(matches_ref) >= 8
            if has_ref_baseline:
                pts1 = np.float32([ref_kp[m.queryIdx].pt for m in matches_ref])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches_ref])
                parallax = median_parallax(pts1, pts2)

            if len(sparse_map) == 0:
                # --- Bootstrap: two-view pose + triangulation, once - still
                # gated on accumulated parallax vs. the reference frame, since
                # (unlike guided per-frame tracking below) a two-view
                # essential-matrix solve has no map yet to be guided by and
                # needs a real baseline to be well-conditioned at all ---
                if has_ref_baseline and parallax < args.min_parallax:
                    status = "accumulating parallax"
                elif has_ref_baseline:
                    result = estimate_relative_pose(ref_kp, kp, matches_ref, K)
                    if result is None:
                        status = "bootstrap pose estimation failed"
                    else:
                        R_rel, t_rel, mask_pose, _, _ = result
                        inliers = int(mask_pose.sum())
                        if inliers < args.min_inliers:
                            status = f"bootstrap: too few inliers ({inliers})"
                        else:
                            R_new, t_new = compose_pose(R_pos, t_pos, R_rel, t_rel)
                            inlier_mask = mask_pose.ravel().astype(bool)

                            new_points, valid, in_front, parallax_deg = triangulate(
                                R_pos, t_pos, R_new, t_new, K,
                                pts1[inlier_mask], pts2[inlier_mask],
                                min_parallax_deg=args.min_triangulation_angle,
                            )
                            kept_matches = [m for m, keep in zip(matches_ref, inlier_mask) if keep]
                            kept_matches = [m for m, keep in zip(kept_matches, valid) if keep]
                            new_desc = desc[[m.trainIdx for m in kept_matches]]

                            base_idx = len(sparse_map)
                            # Bootstrap points are seeded as confirmed directly - there's
                            # no "already trusted" map yet to independently re-observe
                            # against, and bootstrap already required a strict essential-
                            # matrix + high-inlier-count pose rather than a lenient PnP.
                            sparse_map.add_points(new_points[valid], new_desc, confirmed=True)
                            new_ids = range(base_idx, base_idx + int(valid.sum()))
                            obs_ref = pts1[inlier_mask][valid]
                            obs_cur = pts2[inlier_mask][valid]
                            keyframe_observations[0].extend(
                                (pid, x, y) for pid, (x, y) in zip(new_ids, obs_ref)
                            )
                            keyframe_observations.append(
                                [(pid, x, y) for pid, (x, y) in zip(new_ids, obs_cur)]
                            )

                            print(f"frame {frame.index}: BOOTSTRAP  parallax={parallax:.1f}px  "
                                  f"{inliers} pose inliers, {int(valid.sum())} points seeded")

                            recent_step_sizes.append(
                                np.linalg.norm(camera_center(R_new, t_new) - camera_center(R_pos, t_pos))
                            )

                            prev_pose = (R_pos, t_pos)
                            prev_pose_frame = cur_pose_frame
                            R_pos, t_pos = R_new, t_new
                            cur_pose_frame = frame.index
                            keyframe_poses.append(KeyframePose(R_pos, t_pos, frame.timestamp))
                            if not args.no_ba and len(keyframe_poses) % args.ba_every == 0:
                                ba_result = _run_local_ba(keyframe_poses, keyframe_observations,
                                                           sparse_map, K, args.ba_window,
                                                           max_points=args.ba_max_points)
                                R_pos, t_pos = _validate_and_apply_ba(
                                    ba_result, keyframe_poses, sparse_map,
                                    recent_step_sizes, R_pos, t_pos,
                                    args.max_plausible_rotation, args.max_step_ratio,
                                )
                            n_keyframes += 1
                            is_keyframe = True
                            status = f"BOOTSTRAP ({int(valid.sum())} points seeded)"

            else:
                # --- Track: motion-predicted guided PnP against the map,
                # attempted every frame (confirmed points only - provisional
                # points must never influence the pose that could end up
                # confirming them). --min-parallax no longer gates whether a
                # pose is estimated at all - only whether this frame's
                # tracked pose *also* gets promoted to a keyframe, below. ---
                track_accepted = False
                if (
                    prev_pose is not None
                    and cur_pose_frame - prev_pose_frame == 1
                    and frame.index - cur_pose_frame == 1
                ):
                    R_pred, t_pred = predict_constant_velocity(
                        prev_pose[0], prev_pose[1], R_pos, t_pos
                    )
                else:
                    R_pred, t_pred = R_pos, t_pos

                # Widen the search radius the longer tracking has gone without
                # a successfully accepted pose: each consecutive failure means
                # the constant-velocity prediction itself grows less trustworthy
                # (extrapolated over a longer real gap than one frame), so the
                # window compensates for that growing uncertainty - capped so a
                # genuinely lost stretch still costs at most a few times the
                # base window rather than degrading into a full-frame search.
                effective_window = args.guided_window * min(
                    1.0 + 0.5 * n_consecutive_untracked, 5.0
                )
                map_indices, frame_indices = sparse_map.match_against_guided(
                    kp, desc, K, R_pred, t_pred,
                    window=effective_window, ratio=args.ratio, mask=sparse_map.confirmed,
                )
                if len(map_indices) < 6:
                    status = f"too few guided map matches ({len(map_indices)})"
                else:
                    object_points = sparse_map.points[map_indices]
                    image_points = np.float32([kp[i].pt for i in frame_indices])
                    result = estimate_pose_pnp(object_points, image_points, K)
                    if result is None:
                        status = "PnP failed"
                    else:
                        R_new, t_new, pnp_inlier_mask = result
                        pnp_inliers = int(pnp_inlier_mask.sum())
                        rot_deg = rotation_angle_deg(R_new @ R_pos.T)
                        step_size = np.linalg.norm(
                            camera_center(R_new, t_new) - camera_center(R_pos, t_pos)
                        )
                        implausible_step = (
                            len(recent_step_sizes) >= 5
                            and step_size > args.max_step_ratio * np.median(recent_step_sizes)
                        )
                        if pnp_inliers < args.pnp_min_inliers:
                            status = f"PnP: too few inliers ({pnp_inliers})"
                        elif rot_deg > args.max_plausible_rotation:
                            # A confident-looking inlier count doesn't mean the pose is
                            # right - PnP can lock onto a degenerate/ambiguous alternate
                            # solution (near-planar or otherwise poorly depth-distributed
                            # points are especially prone to this) that fits the same 2D
                            # observations almost as well as the true pose. Real motion
                            # between two close frames shouldn't produce a huge rotation
                            # jump, so treat one as a red flag and reject it rather than
                            # trusting whatever PnP returned.
                            status = f"PnP: implausible rotation ({rot_deg:.1f}deg)"
                        elif implausible_step:
                            # Companion check: the same kind of degenerate PnP solution
                            # doesn't always show up as a rotation flip - it can instead
                            # keep a plausible rotation but put the camera in the wrong
                            # place. Left unchecked, accepting this would also corrupt
                            # every subsequent frame's rotation-vs-previous comparison
                            # (measured against this now-wrong pose), which is how a
                            # single undetected bad pose turns into a run of repeated
                            # "implausible rotation" rejections afterward.
                            status = (
                                f"PnP: implausible step "
                                f"({step_size:.2f} vs median {np.median(recent_step_sizes):.2f})"
                            )
                        else:
                            # Pose accepted for this frame - update the running
                            # estimate and motion history regardless of whether
                            # this frame goes on to become a keyframe below, so
                            # the constant-velocity prediction stays current
                            # every frame rather than only at keyframes.
                            recent_step_sizes.append(step_size)
                            del recent_step_sizes[:-20]
                            track_accepted = True
                            prev_pose = (R_pos, t_pos)
                            prev_pose_frame = cur_pose_frame
                            R_pos, t_pos = R_new, t_new
                            cur_pose_frame = frame.index

                            if has_ref_baseline and parallax >= args.min_parallax:
                                # --- Promote to keyframe: confirm provisional
                                # points, triangulate new structure, run BA ---
                                this_kf_observations = [
                                    (int(idx), float(x), float(y))
                                    for idx, (x, y) in zip(
                                        map_indices[pnp_inlier_mask],
                                        image_points[pnp_inlier_mask],
                                    )
                                ]

                                # Now that the pose is trustworthy (confirmed points
                                # only), check provisional points for independent
                                # re-observation. Each one that reprojects consistently
                                # gets one confirmation credit - promoted to confirmed
                                # only after accumulating enough of them (see Map).
                                prov_map_idx, prov_frame_idx = sparse_map.match_against(
                                    desc, args.ratio, mask=~sparse_map.confirmed
                                )
                                n_reobserved = 0
                                if len(prov_map_idx) > 0:
                                    rvec_new, _ = cv2.Rodrigues(R_new)
                                    proj, _ = cv2.projectPoints(
                                        sparse_map.points[prov_map_idx], rvec_new, t_new, K, None
                                    )
                                    proj = proj.reshape(-1, 2)
                                    prov_pixels = np.float32([kp[i].pt for i in prov_frame_idx])
                                    errors = np.linalg.norm(proj - prov_pixels, axis=1)
                                    good = errors < args.confirm_reproj_error
                                    reobserved_idx = prov_map_idx[good]
                                    sparse_map.confirm(reobserved_idx)
                                    this_kf_observations.extend(
                                        (int(idx), float(x), float(y))
                                        for idx, (x, y) in zip(reobserved_idx, prov_pixels[good])
                                    )
                                    n_reobserved = int(good.sum())

                                # Triangulate new points from pairs not already tied to
                                # an existing map point (confirmed or provisional),
                                # using this frame's just-solved (map-scale-consistent)
                                # pose.
                                already_in_map = set(frame_indices.tolist()) | set(prov_frame_idx.tolist())
                                new_mask = np.array(
                                    [m.trainIdx not in already_in_map for m in matches_ref]
                                )
                                candidates = [m for m, keep in zip(matches_ref, new_mask) if keep]

                                new_count = 0
                                if len(candidates) >= 8:
                                    # pts1/pts2 are ref-keyframe/current-frame pixel pairs
                                    # (from matches_ref) - triangulate against the
                                    # REFERENCE keyframe's own pose (ref_R/ref_t), not
                                    # R_pos/t_pos, which by this point already equals
                                    # R_new/t_new (updated above for every tracked frame,
                                    # not just keyframes) and would give a zero-baseline,
                                    # degenerate triangulation against itself.
                                    new_points, valid, in_front, parallax_deg = triangulate(
                                        ref_R, ref_t, R_new, t_new, K,
                                        pts1[new_mask], pts2[new_mask],
                                        min_parallax_deg=args.min_triangulation_angle,
                                    )
                                    kept = [m for m, keep in zip(candidates, valid) if keep]
                                    new_desc = desc[[m.trainIdx for m in kept]]

                                    base_idx = len(sparse_map)
                                    sparse_map.add_points(new_points[valid], new_desc)
                                    new_ids = range(base_idx, base_idx + int(valid.sum()))
                                    obs_ref = pts1[new_mask][valid]
                                    obs_cur = pts2[new_mask][valid]
                                    keyframe_observations[-1].extend(
                                        (pid, x, y) for pid, (x, y) in zip(new_ids, obs_ref)
                                    )
                                    this_kf_observations.extend(
                                        (pid, x, y) for pid, (x, y) in zip(new_ids, obs_cur)
                                    )
                                    new_count = int(valid.sum())

                                print(f"frame {frame.index}: KEYFRAME  parallax={parallax:.1f}px  "
                                      f"rotation={rot_deg:.1f}deg  "
                                      f"{pnp_inliers}/{len(map_indices)} PnP inliers, "
                                      f"{new_count} new (provisional) points, "
                                      f"{n_reobserved} re-observed "
                                      f"({sparse_map.n_confirmed} confirmed / {len(sparse_map)} total)")

                                keyframe_poses.append(KeyframePose(R_pos, t_pos, frame.timestamp))
                                keyframe_observations.append(this_kf_observations)
                                if not args.no_ba and len(keyframe_poses) % args.ba_every == 0:
                                    ba_result = _run_local_ba(keyframe_poses, keyframe_observations,
                                                               sparse_map, K, args.ba_window,
                                                               max_points=args.ba_max_points)
                                    R_pos, t_pos = _validate_and_apply_ba(
                                        ba_result, keyframe_poses, sparse_map,
                                        recent_step_sizes, R_pos, t_pos,
                                        args.max_plausible_rotation, args.max_step_ratio,
                                    )

                                if args.depth_densify:
                                    if depth_rows is None:
                                        depth_rows = scanline_rows(
                                            frame.image.shape[0], args.depth_scan_rows
                                        )
                                    new_ml_points, raw_depth, background_mask = _run_depth_densify(
                                        frame.image, R_pos, t_pos, sparse_map,
                                        map_indices, image_points, pnp_inlier_mask, K,
                                        depth_estimator, depth_rows, args.depth_scan_stride,
                                    )
                                    if len(new_ml_points) > 0:
                                        ml_points = np.vstack([ml_points, new_ml_points])
                                    last_depth_vis = colorize_depth_with_background(raw_depth, background_mask)

                                n_keyframes += 1
                                is_keyframe = True
                                status = f"KEYFRAME ({pnp_inliers} inliers, {len(sparse_map)} map points)"
                            else:
                                n_tracked_only += 1
                                status = (
                                    f"TRACK ({pnp_inliers}/{len(map_indices)} PnP inliers, "
                                    f"frame-only)"
                                )
                                print(f"frame {frame.index}: TRACK  parallax={parallax:.1f}px  "
                                      f"rotation={rot_deg:.1f}deg  "
                                      f"{pnp_inliers}/{len(map_indices)} PnP inliers "
                                      f"(frame-only, not a keyframe)")

                n_consecutive_untracked = 0 if track_accepted else n_consecutive_untracked + 1

            if not args.no_display:
                match_vis = cv2.drawMatches(
                    ref_image, ref_kp, frame.image, kp, matches_ref[:200], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                cv2.putText(
                    match_vis, f"frame {frame.index}  matches={len(matches_ref)}  "
                    f"parallax={parallax:.1f}px  {status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if is_keyframe else (0, 200, 0), 2,
                )

                panels = [match_vis]
                if args.depth_densify:
                    # match_vis is drawMatches' side-by-side ref+current pair (double
                    # width) - the depth panel is a single frame, so only its height
                    # needs to line up for hstack, not its width.
                    depth_panel = (
                        last_depth_vis if last_depth_vis is not None
                        else np.zeros((frame.image.shape[0], frame.image.shape[1], 3), dtype=np.uint8)
                    )
                    if depth_panel.shape[0] != match_vis.shape[0]:
                        scale = match_vis.shape[0] / depth_panel.shape[0]
                        depth_panel = cv2.resize(depth_panel, None, fx=scale, fy=scale)
                    panels.append(depth_panel)

                positions = _keyframe_positions(keyframe_poses)
                if args.depth_densify:
                    # Cleaner plot when densifying: just the trajectory and
                    # the ML depth points it's meant to be compared against,
                    # not also the ORB map that's already shown implicitly
                    # (via which pixels contribute to ml_points) - showing
                    # both clutters the exact thing being visually checked.
                    traj_vis = render_trajectory(positions, ml_points=ml_points, size=match_vis.shape[0])
                else:
                    traj_vis = render_trajectory(
                        positions,
                        sparse_map.points[sparse_map.confirmed],
                        sparse_map.points[~sparse_map.confirmed],
                        size=match_vis.shape[0],
                    )
                panels.append(traj_vis)
                combined = np.hstack(panels)

                # The raw combined image (two video frames + a square map panel
                # sized to match their height) is often wider/taller than a
                # typical screen for portrait phone footage - scale it down to
                # fit a display-sized window rather than letting part of it
                # render off-screen.
                max_w, max_h = 1600, 900
                display_scale = min(max_w / combined.shape[1], max_h / combined.shape[0], 1.0)
                if display_scale < 1.0:
                    combined = cv2.resize(
                        combined, None, fx=display_scale, fy=display_scale,
                        interpolation=cv2.INTER_AREA,
                    )

                window_title = (
                    "SLAM v1+v2 - map tracking + ML depth densify (bootstrap + PnP)"
                    if args.depth_densify else
                    "SLAM v1 - map tracking (bootstrap + PnP)"
                )
                cv2.imshow(window_title, combined)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

            if is_keyframe:
                ref_kp, ref_desc, ref_image = kp, desc, frame.image
                ref_R, ref_t = R_pos, t_pos
            else:
                n_skipped += 1

    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n{n_keyframes} keyframes accepted "
          f"({sparse_map.n_confirmed} confirmed / {len(sparse_map)} total map points), "
          f"{n_skipped} frames not promoted to a keyframe "
          f"({n_tracked_only} still tracked frame-only, "
          f"{n_skipped - n_tracked_only} lost tracking entirely)")
    if args.depth_densify:
        print(f"{len(ml_points)} ML-depth points sampled (sanity-check plot only - "
              f"not part of the tracked map)")

    if args.global_ba_at_end:
        import time

        if len(keyframe_poses) < 2:
            print("[global BA: skipped - fewer than 2 keyframes]")
        else:
            before_stats = _reprojection_error_stats(
                keyframe_poses, keyframe_observations, sparse_map, K
            )
            t0 = time.perf_counter()
            ba_result = _run_global_ba(
                keyframe_poses, keyframe_observations, sparse_map, K,
                max_nfev=args.global_ba_max_nfev,
                ftol=args.global_ba_ftol, xtol=args.global_ba_xtol,
            )
            elapsed = time.perf_counter() - t0
            if ba_result is None:
                print(f"[global BA: skipped in {elapsed:.1f}s - fewer than 10 map points "
                      f"observed across the trajectory]")
            else:
                _validate_and_apply_ba(
                    ba_result, keyframe_poses, sparse_map, recent_step_sizes,
                    keyframe_poses[-1].R, keyframe_poses[-1].t,
                    args.global_ba_max_plausible_rotation, args.global_ba_max_step_ratio,
                )
                after_stats = _reprojection_error_stats(
                    keyframe_poses, keyframe_observations, sparse_map, K
                )
                print(f"[global BA: {elapsed:.1f}s - reprojection error (px) "
                      f"mean {before_stats['mean']:.2f}->{after_stats['mean']:.2f}, "
                      f"median {before_stats['median']:.2f}->{after_stats['median']:.2f}, "
                      f"max {before_stats['max']:.2f}->{after_stats['max']:.2f} "
                      f"(unchanged before==after if the result was rejected as implausible - "
                      f"see message above)]")

    positions = _keyframe_positions(keyframe_poses)

    os.makedirs(os.path.dirname(args.plot_output), exist_ok=True)

    if args.trajectory_output:
        write_tum_trajectory(args.trajectory_output, keyframe_poses)
        print(f"Saved TUM-format trajectory ({len(keyframe_poses)} keyframes) to "
              f"{args.trajectory_output}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    if args.depth_densify:
        # Same reasoning as the live view: just trajectory + ML points for a
        # clean comparison, not also the ORB map cluttering the same plot.
        if len(ml_points) > 0:
            ax.scatter(ml_points[:, 0], ml_points[:, 2],
                       c="lightblue", s=2, label="ML depth (unverified)", zorder=0)
    elif len(sparse_map) > 0:
        confirmed_pts = sparse_map.points[sparse_map.confirmed]
        provisional_pts = sparse_map.points[~sparse_map.confirmed]
        if len(provisional_pts) > 0:
            ax.scatter(provisional_pts[:, 0], provisional_pts[:, 2],
                       c="orange", s=4, label="provisional points", zorder=1)
        if len(confirmed_pts) > 0:
            ax.scatter(confirmed_pts[:, 0], confirmed_pts[:, 2],
                       c="black", s=4, label="confirmed points", zorder=1)
    ax.plot(positions[:, 0], positions[:, 2], "-o", markersize=2, linewidth=1, zorder=2)
    ax.scatter(positions[0, 0], positions[0, 2], c="green", s=80, label="start", zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 2], c="red", s=80, label="end", zorder=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Z (forward)")
    ax.set_title("Camera trajectory + persistent map (top-down, fixed scale after bootstrap)")
    ax.axis("equal")
    ax.legend()
    ax.grid(True)
    fig.savefig(args.plot_output, dpi=150)
    print(f"Saved trajectory plot to {args.plot_output}")


if __name__ == "__main__":
    _demo()