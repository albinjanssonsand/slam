"""
Two-view relative pose estimation via the essential matrix.
"""

import cv2
import numpy as np


def estimate_relative_pose(kp1, kp2, matches, camera_matrix):
    """
    Estimate the relative camera pose between two frames from matched keypoints.

    Returns (R, t, mask, pts1, pts2):
      R, t       - rotation matrix and *unit-length* translation from camera1
                   to camera2. Monocular scale is unknown: t is a direction,
                   not a metric displacement, and is not comparable in scale
                   from one frame pair to the next.
      mask       - inlier mask (Nx1) marking which of pts1/pts2 pairs agree
                   with the recovered pose.
      pts1, pts2 - matched pixel coordinates (Nx2 float32), same order as mask.

    Returns None if there aren't enough matches to attempt an estimate, or if
    essential matrix / pose recovery fails.
    """
    if len(matches) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(
        pts1, pts2, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or E.shape != (3, 3):
        return None

    inlier_count, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, camera_matrix, mask=mask)
    if inlier_count < 8:
        return None

    return R, t, mask_pose, pts1, pts2


def estimate_homography(pts1, pts2, ransac_threshold=3.0, max_iters=2000, confidence=0.999):
    """
    Estimate a planar homography (x2 ~ H x1, pixel coordinates) via RANSAC -
    the paper's parallel model to estimate_fundamental_matrix (§IV steps
    1-2, mapping._bootstrap_dual_model), scored against it via
    homography_score before one of the two is picked.

    Returns (H, mask) - mask is an inlier mask (Nx1), matching
    estimate_relative_pose's convention - or None if there aren't enough
    matches or the RANSAC fit fails.
    """
    if len(pts1) < 4:
        return None
    H, mask = cv2.findHomography(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_threshold,
        maxIters=max_iters, confidence=confidence,
    )
    if H is None:
        return None
    return H, mask


def estimate_fundamental_matrix(pts1, pts2, ransac_threshold=3.0, max_iters=2000, confidence=0.999):
    """
    Estimate the (uncalibrated) fundamental matrix via RANSAC - the paper's
    parallel model to estimate_homography (§IV steps 1-2). Deliberately not
    calibrated to an essential matrix here: the R_H model-selection score
    (fundamental_score) operates directly on pixel coordinates, and E is
    only needed at all once this model has actually been selected (see
    decompose_essential, given camera_matrix.T @ F @ camera_matrix).

    Returns (F, mask) or None if there aren't enough matches or the RANSAC
    fit fails.
    """
    if len(pts1) < 8:
        return None
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=ransac_threshold,
        confidence=confidence, maxIters=max_iters,
    )
    if F is None or F.shape != (3, 3):
        return None
    return F, mask


def decompose_homography(H, camera_matrix):
    """
    Recover motion hypotheses from a homography (paper §IV step 4's
    homography branch, up to 8 solutions before OpenCV's own built-in
    positive-depth filtering - see cv2.decomposeHomographyMat) rather than
    committing to a single one - mapping._bootstrap_dual_model disambiguates
    them itself by directly triangulating each.

    Returns a list of (R, t) - t is a direction only (unit-length up to the
    homography's own unknown-depth normalization), like decompose_essential.
    """
    _, rotations, translations, _ = cv2.decomposeHomographyMat(H, camera_matrix)
    return [(R, t.reshape(3, 1)) for R, t in zip(rotations, translations)]


def pure_rotation_homography(R, camera_matrix):
    """
    The homography a camera rotation alone (zero translation) induces
    between two views: H = K @ R @ K^-1 - unlike a general planar
    homography, independent of scene depth or plane geometry entirely, since
    a pure rotation maps every scene point's ray the same way regardless of
    its distance from the camera.

    Used to score decompose_homography's rotation hypotheses against each
    other (mapping._rotation_only_fallback, #36's degenerate-translation
    tracking fallback) by checking how well each hypothesis's OWN implied
    zero-translation homography explains the actual observed 2D-2D
    correspondences (via homography_score) - the candidate whose rotation
    alone reproduces the motion best is the one to trust, without needing
    any 3D map or triangulation to disambiguate (both unavailable/
    meaningless when there's genuinely no translation to triangulate from).
    """
    return camera_matrix @ R @ np.linalg.inv(camera_matrix)


def decompose_essential(E):
    """
    Recover all 4 motion hypotheses [R1,t], [R1,-t], [R2,t], [R2,-t] from an
    essential matrix via SVD (paper §IV step 4's fundamental-matrix branch).

    Unlike estimate_relative_pose's cv2.recoverPose (which already commits
    to a single hypothesis via its own internal cheirality vote), this
    exposes every candidate so mapping._bootstrap_dual_model can disambiguate
    them itself by directly triangulating each, per the paper - cheirality
    alone is unreliable under low parallax.
    """
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = t.reshape(3, 1)
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def _symmetric_transfer_score(err1_sq, err2_sq, threshold, gamma):
    """
    Paper §IV Eq. 2: sum, over every correspondence, of (gamma - d1^2) +
    (gamma - d2^2), each term included only where that OWN direction's
    squared transfer error falls under threshold (0 otherwise) - matching
    ORB-SLAM2's actual CheckHomography/CheckFundamental (each direction is
    its own independent inlier test and reward, not a combined two-way
    error checked against one threshold): a correspondence that fits
    perfectly in one direction but poorly in the other still banks the
    first direction's reward, and a correspondence whose two individual
    errors are each just under threshold isn't wrongly zeroed out by their
    SUM exceeding it. Rewards inliers in proportion to how well they fit
    rather than a flat inlier count, so the two models' scores stay
    comparable regardless of how many of their own RANSAC iterations'
    correspondences happen to qualify.
    """
    s1 = np.where(err1_sq < threshold, gamma - err1_sq, 0.0)
    s2 = np.where(err2_sq < threshold, gamma - err2_sq, 0.0)
    return float(s1.sum() + s2.sum())


def homography_score(H, pts1, pts2, threshold=5.99, gamma=5.99):
    """
    Paper §IV Eq. 2 score for a homography model: symmetric transfer error
    is the squared reprojection distance, scored independently in each
    direction (x2 vs H x1, x1 vs H^-1 x2 - see _symmetric_transfer_score).
    threshold/gamma default to T_H=5.99 (paper's own 2-DOF, 95%-confidence
    chi-squared bound).

    Returns 0.0 (the model simply doesn't score, rather than raising) if H
    is numerically singular - a real possibility straight out of RANSAC (a
    near-planar/low-parallax correspondence set, exactly the kind of scene
    this dual-model selection exists to handle, can produce a degenerate
    fit) that would otherwise crash mapping._bootstrap_dual_model's caller
    instead of letting it fall through to "no dominant motion hypothesis"/
    keep accumulating frames like every other bootstrap-rejection path.
    """
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0

    p1h = np.hstack([pts1, np.ones((len(pts1), 1))])
    p2h = np.hstack([pts2, np.ones((len(pts2), 1))])
    proj2 = (H @ p1h.T).T
    proj2 = proj2[:, :2] / proj2[:, 2:3]
    proj1 = (H_inv @ p2h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    err2_sq = np.sum((proj2 - pts2) ** 2, axis=1)
    err1_sq = np.sum((proj1 - pts1) ** 2, axis=1)

    return _symmetric_transfer_score(err1_sq, err2_sq, threshold, gamma)


def fundamental_score(F, pts1, pts2, threshold=3.84, gamma=5.99):
    """
    Paper §IV Eq. 2 score for a fundamental-matrix model: symmetric transfer
    error is the squared point-to-epipolar-line distance, scored
    independently in each direction (see _symmetric_transfer_score).
    threshold defaults to T_F=3.84 (paper's own 1-DOF, 95%-confidence
    chi-squared bound - a point-to-line distance has one degree of freedom,
    unlike a point-to-point reprojection's two); gamma stays T_H=5.99 (same
    as homography_score's default) so both models' rewards are on the same
    scale, per the paper - the whole point of R_H being able to compare them.
    """
    p1h = np.hstack([pts1, np.ones((len(pts1), 1))])
    p2h = np.hstack([pts2, np.ones((len(pts2), 1))])
    lines2 = (F @ p1h.T).T
    lines1 = (F.T @ p2h.T).T
    num2 = np.sum(lines2[:, :2] * p2h[:, :2], axis=1) + lines2[:, 2]
    num1 = np.sum(lines1[:, :2] * p1h[:, :2], axis=1) + lines1[:, 2]
    denom2 = np.maximum(np.sum(lines2[:, :2] ** 2, axis=1), 1e-12)
    denom1 = np.maximum(np.sum(lines1[:, :2] ** 2, axis=1), 1e-12)
    err2_sq = (num2 ** 2) / denom2
    err1_sq = (num1 ** 2) / denom1

    return _symmetric_transfer_score(err1_sq, err2_sq, threshold, gamma)


def rotation_angle_deg(R):
    """Magnitude of the rotation represented by R, in degrees."""
    rvec, _ = cv2.Rodrigues(R)
    return np.degrees(np.linalg.norm(rvec))


def compose_pose(R_pos, t_pos, R_rel, t_rel):
    """
    Chain a relative pose (camera_{k-1} -> camera_k) onto a global world-to-camera
    pose (world -> camera_{k-1}) to get the global pose for camera_k.

    world-to-camera convention: X_cam = R_pos @ X_world + t_pos
    """
    R_pos_new = R_rel @ R_pos
    t_pos_new = R_rel @ t_pos + t_rel
    return R_pos_new, t_pos_new


def camera_center(R_pos, t_pos):
    """Camera center in world coordinates, given a world-to-camera pose."""
    return -R_pos.T @ t_pos


def predict_constant_velocity(R_prev, t_prev, R_cur, t_cur):
    """
    Predict the next pose assuming the camera continues its most recent
    relative motion unchanged (constant-velocity model): the world-to-camera
    transform observed going from R_prev/t_prev to R_cur/t_cur is assumed to
    repeat identically going from R_cur/t_cur to the predicted pose.
    """
    R_rel = R_cur @ R_prev.T
    t_rel = t_cur - R_rel @ t_prev
    return compose_pose(R_cur, t_cur, R_rel, t_rel)


def median_parallax(pts1, pts2):
    """Median pixel displacement between two matched point sets - a cheap proxy
    for how much camera baseline has accumulated between two frames."""
    return float(np.median(np.linalg.norm(pts2 - pts1, axis=1)))


def render_trajectory(positions, map_points=None, ml_points=None,
                       size=600, margin=40):
    """
    Render the top-down (X, Z) camera trajectory + sparse map into a BGR image.

    map_points, if given, are drawn black - the current sparse map (already
    excludes any point removed by §VI-B culling, see Map.active).
    ml_points, if given, are drawn light blue - ML-depth-derived points (see
    pipeline/depth_ml.py), plotted for visual sanity-checking only; they are
    not part of the map used for pose estimation.
    """
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)

    pos_xz = np.asarray(positions).reshape(-1, 3)[:, [0, 2]]
    map_xz = (
        np.asarray(map_points).reshape(-1, 3)[:, [0, 2]]
        if map_points is not None and len(map_points) > 0
        else np.empty((0, 2))
    )
    ml_xz = (
        np.asarray(ml_points).reshape(-1, 3)[:, [0, 2]]
        if ml_points is not None and len(ml_points) > 0
        else np.empty((0, 2))
    )

    if len(pos_xz) < 2:
        return canvas

    all_xz = np.vstack([a for a in (pos_xz, map_xz, ml_xz) if len(a)])
    # Percentile bounds rather than literal min/max - a single outlier point
    # (e.g. an ML-depth point that's still somewhat off despite the upstream
    # sanity checks) would otherwise dictate the whole canvas's scale on its
    # own, shrinking the majority of real structure down to an unreadable
    # speck. A point outside this range still gets drawn - to_canvas doesn't
    # clip - it just may land outside the visible canvas.
    #
    # The camera trajectory itself is exempted from being clipped this way -
    # unlike the point clouds, it's already pose-plausibility-checked
    # upstream (--max-plausible-rotation/--max-step-ratio), so it isn't the
    # thing outliers come from, and losing sight of "where the camera even
    # is" would defeat the point of the plot. Expand the bounds to always
    # cover it fully, on top of (never instead of) the robust point bounds.
    min_xy = np.minimum(np.percentile(all_xz, 2, axis=0), pos_xz.min(axis=0))
    max_xy = np.maximum(np.percentile(all_xz, 98, axis=0), pos_xz.max(axis=0))
    span = np.maximum(max_xy - min_xy, 1e-3)
    scale = (size - 2 * margin) / span.max()

    def to_canvas(p):
        x = int((p[0] - min_xy[0]) * scale + margin)
        y = int((p[1] - min_xy[1]) * scale + margin)
        return x, size - y  # flip so +Z (forward) points up

    for p in ml_xz:
        cv2.circle(canvas, to_canvas(p), 1, (139, 0, 0), -1)  # dark blue - ML depth (unverified)
    for p in map_xz:
        cv2.circle(canvas, to_canvas(p), 2, (0, 0, 0), -1)  # black - sparse map

    for i in range(1, len(pos_xz)):
        cv2.line(canvas, to_canvas(pos_xz[i - 1]), to_canvas(pos_xz[i]), (60, 60, 60), 2)
    for p in pos_xz:
        cv2.circle(canvas, to_canvas(p), 3, (200, 100, 0), -1)
    cv2.circle(canvas, to_canvas(pos_xz[0]), 7, (0, 160, 0), -1)   # start
    cv2.circle(canvas, to_canvas(pos_xz[-1]), 7, (0, 0, 255), -1)  # latest keyframe

    cv2.putText(canvas, "trajectory + sparse map (top-down)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return canvas


def _demo():
    import argparse

    from capture.video_source import open_calibrated_source
    from pipeline.features import create_orb, detect_and_compute, match_descriptors
    from pipeline.triangulation import triangulate

    parser = argparse.ArgumentParser(
        description="Chain keyframe-to-keyframe poses across a video and plot the trajectory"
    )
    parser.add_argument("--video", required=True,
                         help="Video file path, integer device index, or image-sequence folder "
                              "(e.g. a TUM RGB-D sequence, containing rgb.txt)")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    parser.add_argument("--n-features", type=int, default=2000)
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold")
    parser.add_argument("--min-parallax", type=float, default=30.0,
                         help="Minimum median pixel displacement vs the reference keyframe "
                              "before attempting pose estimation (px)")
    parser.add_argument("--min-inliers", type=int, default=60,
                         help="Minimum pose inliers required to accept a new keyframe")
    parser.add_argument("--min-triangulation-angle", type=float, default=1,
                         help="Minimum parallax angle (deg) between viewing rays to keep a "
                              "triangulated point; points near the direction of travel are "
                              "numerically unstable below this and get discarded")
    parser.add_argument("--plot-output", default="pipeline/data/trajectory.png")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable the live matches+trajectory window")
    args = parser.parse_args()

    orb = create_orb(args.n_features)

    R_pos = np.eye(3)
    t_pos = np.zeros((3, 1))
    positions = [camera_center(R_pos, t_pos)]
    map_points = []

    ref_kp = None
    ref_desc = None
    ref_image = None
    n_keyframes = 0
    n_skipped = 0

    with open_calibrated_source(args.video, args.calibration) as frames:
        K = frames.camera_matrix_undistorted
        fps = frames.fps
        delay_ms = max(1, int(1000 / fps))

        for frame in frames:
            kp, desc = detect_and_compute(orb, frame.image)

            if ref_desc is None:
                ref_kp, ref_desc, ref_image = kp, desc, frame.image
                continue

            matches = match_descriptors(ref_desc, desc, args.ratio)

            status = "insufficient matches"
            parallax = 0.0
            is_keyframe = False

            if len(matches) >= 8:
                pts1 = np.float32([ref_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp[m.trainIdx].pt for m in matches])
                parallax = median_parallax(pts1, pts2)

                if parallax < args.min_parallax:
                    status = "accumulating parallax"
                else:
                    result = estimate_relative_pose(ref_kp, kp, matches, K)
                    if result is None:
                        status = "pose estimation failed"
                    else:
                        R_rel, t_rel, mask_pose, _, _ = result
                        inliers = int(mask_pose.sum())
                        if inliers < args.min_inliers:
                            status = f"too few inliers ({inliers})"
                        else:
                            angle = rotation_angle_deg(R_rel)
                            print(f"frame {frame.index}: KEYFRAME  parallax={parallax:.1f}px  "
                                  f"{len(matches)} matches, {inliers} inliers, rotation={angle:.2f} deg")

                            inlier_mask = mask_pose.ravel().astype(bool)
                            R_pos_prev, t_pos_prev = R_pos, t_pos
                            R_pos, t_pos = compose_pose(R_pos, t_pos, R_rel, t_rel)

                            new_points, valid, in_front, parallax_deg = triangulate(
                                R_pos_prev, t_pos_prev, R_pos, t_pos, K,
                                pts1[inlier_mask], pts2[inlier_mask],
                                min_parallax_deg=args.min_triangulation_angle,
                            )
                            behind_count = int((~in_front).sum())
                            low_parallax_count = int((in_front & ~valid).sum())
                            print(f"    triangulation: {int(valid.sum())} kept, "
                                  f"{behind_count} behind a camera (cheirality), "
                                  f"{low_parallax_count} below {args.min_triangulation_angle}deg parallax, "
                                  f"max parallax seen={parallax_deg.max():.2f}deg")
                            if valid.any():
                                kept = new_points[valid]
                                map_points.append(kept)
                                depths = kept[:, 2]
                                print(f"    depth range [{depths.min():.2f}, {depths.max():.2f}], "
                                      f"median {np.median(depths):.2f}")

                            positions.append(camera_center(R_pos, t_pos))
                            n_keyframes += 1
                            is_keyframe = True
                            status = f"KEYFRAME ({inliers} inliers, {int(valid.sum())} triangulated)"

            if not args.no_display:
                # Draw against the reference that was actually used for matching
                # above - it may be about to change below if this is a keyframe.
                match_vis = cv2.drawMatches(
                    ref_image, ref_kp, frame.image, kp, matches[:200], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                cv2.putText(
                    match_vis, f"frame {frame.index}  matches={len(matches)}  "
                    f"parallax={parallax:.1f}px  {status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if is_keyframe else (0, 200, 0), 2,
                )

                all_map_points = np.vstack(map_points) if map_points else None
                traj_vis = render_trajectory(positions, all_map_points, size=match_vis.shape[0])
                combined = np.hstack([match_vis, traj_vis])

                cv2.imshow("SLAM v1 - matches to reference keyframe + trajectory", combined)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

            if is_keyframe:
                ref_kp, ref_desc, ref_image = kp, desc, frame.image
            else:
                n_skipped += 1

    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n{n_keyframes} keyframes accepted, {n_skipped} frames skipped "
          f"(insufficient parallax/matches/inliers)")

    positions = np.array(positions).reshape(-1, 3)

    import os
    os.makedirs(os.path.dirname(args.plot_output), exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    if map_points:
        all_map_points = np.vstack(map_points)
        ax.scatter(all_map_points[:, 0], all_map_points[:, 2], c="gray", s=4, label="map points", zorder=1)
    ax.plot(positions[:, 0], positions[:, 2], "-o", markersize=2, linewidth=1, zorder=2)
    ax.scatter(positions[0, 0], positions[0, 2], c="green", s=80, label="start", zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 2], c="red", s=80, label="end", zorder=5)
    ax.set_xlabel("X")
    ax.set_ylabel("Z (forward)")
    ax.set_title("Camera trajectory + sparse map (top-down, arbitrary/inconsistent scale)")
    ax.axis("equal")
    ax.legend()
    ax.grid(True)
    fig.savefig(args.plot_output, dpi=150)
    print(f"Saved trajectory plot to {args.plot_output}")


if __name__ == "__main__":
    _demo()