"""
ORB feature detection and matching between frames.
"""

import cv2
import numpy as np


def create_orb(n_features=2000):
    return cv2.ORB_create(nfeatures=n_features)


def detect_and_compute(orb, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_and_compute_gridded(image, n_features=2000, grid=(4, 4), fallback_thresholds=(10, 5)):
    """
    ORB detection with a per-cell feature quota, instead of one global budget.

    A single cv2.ORB_create(nfeatures=N) call returns whatever N strongest
    corners exist in the WHOLE image - if one region (e.g. a richly textured
    foreground object) has much stronger corners than another (e.g. a distant
    background), that region can consume nearly the entire budget, leaving
    the other region with few or no features. That starves the map of points
    in the under-represented region, which becomes a real problem once the
    over-represented region leaves the frame later in a sequence.

    Matches ORB-SLAM's construction (paper Sec. V-A): one FAST/ORB pyramid
    built over the WHOLE, uncropped image - not per-cell-cropped pyramids
    (an earlier version of this function cropped per cell, which loses
    scale/context for larger-scale features near cell boundaries) - with a
    grid used only to enforce a per-cell corner QUOTA on the resulting
    keypoints, and progressively lower FAST thresholds retried (as further
    full-image passes, never cropped) for any cell still short of quota at
    the default threshold - so a low-texture region gets a real second
    chance instead of just receiving fewer features.

    Detection requests far more candidates (candidate_budget) than the
    final n_features budget: cv2.ORB_create keeps only its N strongest
    corners GLOBALLY by response, so passing n_features directly here would
    let a strongly-textured region crowd out a weakly-textured one before
    our own per-cell selection ever runs - silently reintroducing the exact
    starvation bug this function exists to prevent. Over-generating
    candidates and doing the spatial/per-cell selection ourselves below
    avoids that - but requesting a high nfeatures also makes a single pass
    report the same physical corner more than once far more often than a
    normal-sized request would (confirmed empirically: ~20% of raw
    keypoints from one full-image pass were within 2px of another one,
    typically the same strong corner surviving at more than one pyramid
    octave). detect_pass() collapses each small spatial bin down to one
    representative (highest response) before returning, for exactly this
    reason - without it, duplicate detections of a handful of strong
    corners can fill an entire cell's quota, inflating match counts (more
    near-identical descriptors to match against) without adding real
    geometric diversity, which is what caused a near-total triangulation
    collapse during development (836 essential-matrix inliers -> 5
    triangulated points on freiburg1_xyz's bootstrap pair).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    rows, cols = grid
    h, w = gray.shape[:2]
    per_cell = max(1, n_features // (rows * cols))
    candidate_budget = max(n_features * 20, 20000)
    dup_radius = 2.0

    def cell_of(x, y):
        r = min(rows - 1, int(y * rows / h))
        c = min(cols - 1, int(x * cols / w))
        return r * cols + c

    def detect_pass(fast_threshold=None):
        """One full-image ORB pass; returns a list of (keypoint, descriptor)
        pairs, with duplicates already suppressed via greedy NMS (see
        docstring above) - a fixed-size spatial bin was tried first and
        rejected: two points near a bin boundary can be under dup_radius
        apart yet land in different bins, so it only caught a fraction of
        the actual duplicates (confirmed: 1727 near-duplicate pairs still
        present among selected keypoints with binning, vs. this radius-
        based approach). scipy.spatial.cKDTree is already a project
        dependency (pipeline/mapping.py's Map.match_against_guided)."""
        kwargs = {"nfeatures": candidate_budget}
        if fast_threshold is not None:
            kwargs["fastThreshold"] = fast_threshold
        kp, desc = cv2.ORB_create(**kwargs).detectAndCompute(gray, None)
        if not kp:
            return []

        from scipy.spatial import cKDTree

        pts = np.array([k.pt for k in kp])
        tree = cKDTree(pts)
        order = np.argsort([-k.response for k in kp])
        alive = np.ones(len(kp), dtype=bool)
        kept = []
        for i in order:
            if not alive[i]:
                continue
            kept.append((kp[i], desc[i]))
            for j in tree.query_ball_point(pts[i], r=dup_radius):
                alive[j] = False
        return kept

    buckets = [[] for _ in range(rows * cols)]  # each: list of (keypoint, descriptor)

    for k, d in detect_pass():
        buckets[cell_of(*k.pt)].append((k, d))

    for fast_threshold in fallback_thresholds:
        shortfall = {i for i in range(rows * cols) if len(buckets[i]) < per_cell}
        if not shortfall:
            break

        by_cell = {}
        for k, d in detect_pass(fast_threshold):
            idx = cell_of(*k.pt)
            if idx in shortfall:
                by_cell.setdefault(idx, []).append((k, d))

        for idx, candidates in by_cell.items():
            existing = buckets[idx]
            if not existing:
                buckets[idx].extend(candidates)
                continue
            # A lower threshold re-detects the same strong corners already
            # picked up by the default-threshold pass, alongside new,
            # weaker ones - drop anything within a couple pixels of a
            # keypoint this cell already has, or the "new" corner is just
            # the same physical feature counted twice. Vectorized (not a
            # per-candidate Python loop): profiling showed the naive
            # pairwise any()-over-a-generator version dominating total
            # runtime (~4x slower than the old cropped-cell implementation
            # on a fallback-heavy frame), almost entirely in this check.
            existing_pts = np.array([ek.pt for ek, _ in existing])
            cand_pts = np.array([k.pt for k, _ in candidates])
            dists_sq = ((cand_pts[:, None, :] - existing_pts[None, :, :]) ** 2).sum(axis=2)
            is_dup = dists_sq.min(axis=1) < dup_radius ** 2
            buckets[idx].extend(kd for kd, dup in zip(candidates, is_dup) if not dup)

    all_keypoints = []
    all_descriptors = []
    for bucket in buckets:
        if not bucket:
            continue
        bucket.sort(key=lambda kd: kd[0].response, reverse=True)
        for k, d in bucket[:per_cell]:
            all_keypoints.append(k)
            all_descriptors.append(d)

    if not all_keypoints:
        return [], None

    return all_keypoints, np.vstack(all_descriptors)


def match_descriptors(desc1, desc2, ratio=0.75, metric="hamming"):
    """
    Brute-force matching with Lowe's ratio test, followed by a
    mutual-nearest-neighbor cross-check: a forward match desc1[i]->desc2[j]
    is kept only if desc2[j]'s own best match in desc1 is also i. The ratio
    test alone only asks "is the best candidate much better than the
    second-best," which repetitive/low-distinctiveness texture (desk edges,
    wallpaper, room corners) can pass easily even when the match is wrong;
    cross-check catches many of those since a wrong match's target usually
    has some other, unrelated descriptor as its own true nearest neighbor.
    This reasoning is distance-metric-agnostic, hence the pluggable
    `metric` ("hamming" for ORB's binary descriptors, "l2" for float
    descriptors e.g. SuperPoint's - see issue #38).
    Returns a list of cv2.DMatch, sorted by distance (best first).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    norm = {"hamming": cv2.NORM_HAMMING, "l2": cv2.NORM_L2}[metric]
    matcher = cv2.BFMatcher(norm)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if not good:
        return []

    reverse_best = np.full(len(desc2), -1, dtype=int)
    for m in matcher.match(desc2, desc1):
        reverse_best[m.queryIdx] = m.trainIdx
    good = [m for m in good if reverse_best[m.trainIdx] == m.queryIdx]

    good.sort(key=lambda m: m.distance)
    return good


def _demo():
    import argparse

    from capture.video_source import open_calibrated_source

    parser = argparse.ArgumentParser(description="Visualize ORB matches between consecutive frames")
    parser.add_argument("--video", required=True,
                         help="Video file path, integer device index, or image-sequence folder "
                              "(e.g. a TUM RGB-D sequence, containing rgb.txt)")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    parser.add_argument("--n-features", type=int, default=2000)
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold")
    args = parser.parse_args()

    orb = create_orb(args.n_features)

    prev_image = None
    prev_kp = None
    prev_desc = None

    with open_calibrated_source(args.video, args.calibration) as frames:
        fps = frames.fps
        delay_ms = max(1, int(1000 / fps))

        for frame in frames:
            kp, desc = detect_and_compute(orb, frame.image)

            if prev_desc is not None:
                matches = match_descriptors(prev_desc, desc, args.ratio)
                vis = cv2.drawMatches(
                    prev_image, prev_kp, frame.image, kp, matches[:200], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                cv2.putText(
                    vis, f"frame {frame.index}  matches={len(matches)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                cv2.imshow("orb matches", vis)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

            prev_image, prev_kp, prev_desc = frame.image, kp, desc

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()