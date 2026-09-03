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


def _representative_descriptor(descs):
    """The observation descriptor with the minimum summed Hamming distance
    to every other observation of the same point (paper §III-C) - the most
    centrally-located descriptor among all of a point's observations,
    rather than simply whichever one was seen last."""
    if len(descs) == 1:
        return descs[0]
    summed = np.array([_hamming_distances(d, descs).sum() for d in descs])
    return descs[np.argmin(summed)]


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

    Every point is usable for PnP/guided matching/BA from the moment it's
    created - there's no separate provisional/confirmed admission gate.
    Instead, a point is removed outright if it fails the paper's Recent Map
    Points Culling test (§VI-B): checked only during its first three
    keyframes after creation, a point is removed if it's found by tracking
    in at most 25% of the frames it was predicted visible in, or - once more
    than one keyframe has passed since its creation - if it hasn't been
    observed from at least three keyframes (see `cull_new_points`). Once a
    point survives that window, it's only removed later if its observing-
    keyframe count drops below three for some other reason (e.g. future
    keyframe culling or BA marking observations as outliers - see
    `cull_low_observation_points`). Removed points are excluded from every
    match/local-map/BA query (via `active`/`remove_points`) but their rows
    are never physically deleted, since point indices are used as stable
    identifiers everywhere else (keyframe_observations, BA point ids).
    """

    def __init__(self, pyramid_scale_factor=1.2, pyramid_n_levels=8,
                 covisibility_min_shared=15):
        self.points = np.empty((0, 3), dtype=np.float64)
        self.descriptors = np.empty((0, 32), dtype=np.uint8)

        # §VI-B Recent Map Points Culling bookkeeping: the keyframe a point
        # was created at, how many frames predicted it visible (projected
        # in front, within the frame, passing the viewing-angle/scale-
        # invariance gates) versus how many actually found/matched it
        # (record_visible/record_found - called every frame from guided
        # Track Local Map matching), and whether it's been removed
        # (cull_new_points/cull_low_observation_points/remove_points).
        self.created_kf = np.empty(0, dtype=np.int64)
        self.n_visible = np.empty(0, dtype=np.int64)
        self.n_found = np.empty(0, dtype=np.int64)
        self.removed = np.empty(0, dtype=bool)

        # Per-point metadata (§III-C): viewing direction (mean unit vector,
        # observing-camera-center -> point) and the scale-invariance
        # distance range [d_min, d_max] the point can be reliably matched
        # within, given the ORB pyramid level(s) it was actually detected
        # at. Both are maintained incrementally by add_observation as new
        # observations arrive - never recomputed over the whole map.
        # pyramid_scale_factor/pyramid_n_levels must match the ORB pyramid
        # actually used for detection (pipeline.features' cv2.ORB_create
        # calls all use the cv2 defaults of 1.2/8 - see detect_and_compute_gridded).
        self.viewing_direction = np.empty((0, 3), dtype=np.float64)
        self.d_min = np.empty(0, dtype=np.float64)
        self.d_max = np.empty(0, dtype=np.float64)
        self.pyramid_scale_factor = pyramid_scale_factor
        self.pyramid_n_levels = pyramid_n_levels

        # Per point: list of (kf_idx, camera_center, descriptor, octave)
        # tuples, one per observation, and the set of keyframe indices that
        # have observed it - the raw material add_observation's incremental
        # metadata/graph updates are computed from.
        self._observations = []
        self._point_keyframes = []

        # Covisibility graph (§III-D, plain covisibility only - no Essential
        # Graph/spanning tree/loop-closure edges): kf_idx -> {other_kf_idx:
        # shared_point_count}, symmetric, incrementally updated by
        # add_observation (each point newly shared between two keyframes
        # bumps their edge weight by exactly one - never a full recompute).
        # Edges below covisibility_min_shared are still stored, so the
        # threshold can change later without re-deriving anything;
        # covisible_keyframes() applies the cutoff.
        self._covisibility = {}
        self._keyframe_points = {}
        self.covisibility_min_shared = covisibility_min_shared

        # kf_idx -> set of that keyframe's own ORB feature indices already
        # tied to a map point (across every observation ever registered for
        # it) - lets §VI-C new-point search (see mapping._create_new_points_
        # from_covisible_keyframes) restrict candidate correspondences in a
        # covisible keyframe to its still-UNmatched features, so it doesn't
        # spawn a duplicate point right next to one that keyframe already
        # observes.
        self._keyframe_matched_frame_idx = {}

    def __len__(self):
        return len(self.points)

    @property
    def active(self):
        """Points still in the map (not yet removed by §VI-B culling)."""
        return ~self.removed

    @property
    def n_active(self):
        return int(self.active.sum())

    def add_points(self, points_3d, descriptors, created_kf):
        """created_kf: the keyframe index these points are being created at -
        §VI-B's culling window is measured relative to it."""
        if len(points_3d) == 0:
            return
        n = len(points_3d)
        self.points = np.vstack([self.points, points_3d])
        self.descriptors = np.vstack([self.descriptors, descriptors])
        self.created_kf = np.concatenate(
            [self.created_kf, np.full(n, created_kf, dtype=np.int64)]
        )
        self.n_visible = np.concatenate([self.n_visible, np.zeros(n, dtype=np.int64)])
        self.n_found = np.concatenate([self.n_found, np.zeros(n, dtype=np.int64)])
        self.removed = np.concatenate([self.removed, np.zeros(n, dtype=bool)])
        self.viewing_direction = np.vstack([self.viewing_direction, np.zeros((n, 3))])
        self.d_min = np.concatenate([self.d_min, np.zeros(n)])
        self.d_max = np.concatenate([self.d_max, np.zeros(n)])
        self._observations.extend([] for _ in range(n))
        self._point_keyframes.extend(set() for _ in range(n))

    def record_visible(self, indices):
        """Register that these points were predicted visible (projected in
        front of the camera, within the frame, passing the viewing-angle/
        scale-invariance gates) in the current frame - §VI-B's found/
        predicted-visible ratio denominator."""
        if len(indices) == 0:
            return
        np.add.at(self.n_visible, indices, 1)

    def record_found(self, indices):
        """Register that these points were actually matched by tracking in
        the current frame - §VI-B's found/predicted-visible ratio numerator."""
        if len(indices) == 0:
            return
        np.add.at(self.n_found, indices, 1)

    def remove_points(self, indices):
        """
        Permanently remove these points from the map (§VI-B culling):
        dropped from every keyframe's observed-point set and the
        covisibility graph's edge weights, which excludes them from every
        future match_against/match_against_guided/local-map/BA query (all
        derive their candidate set from _keyframe_points/_point_keyframes).
        self.points/self.descriptors rows are left in place - point indices
        are used as stable identifiers everywhere else (keyframe_
        observations, BA point ids), so nothing is ever reindexed.
        """
        indices = [int(i) for i in indices if not self.removed[i]]
        if not indices:
            return
        self.removed[indices] = True
        for idx in indices:
            kfs = list(self._point_keyframes[idx])
            for kf in kfs:
                self._keyframe_points.get(kf, set()).discard(idx)
            for a in range(len(kfs)):
                for b in range(a + 1, len(kfs)):
                    k1, k2 = kfs[a], kfs[b]
                    if self._covisibility.get(k1, {}).get(k2):
                        self._covisibility[k1][k2] -= 1
                        if self._covisibility[k1][k2] <= 0:
                            del self._covisibility[k1][k2]
                    if self._covisibility.get(k2, {}).get(k1):
                        self._covisibility[k2][k1] -= 1
                        if self._covisibility[k2][k1] <= 0:
                            del self._covisibility[k2][k1]
            self._point_keyframes[idx] = set()
            # Free each observation's (keyframe, ORB feature index) slot
            # back up too - otherwise a covisible keyframe's feature that
            # only ever founded this now-removed point would stay flagged
            # "already matched" forever, permanently blocking any future
            # §VI-C search from spawning a replacement point there.
            for obs_kf, _, _, _, obs_frame_idx in self._observations[idx]:
                if obs_frame_idx is not None:
                    self._keyframe_matched_frame_idx.get(obs_kf, set()).discard(int(obs_frame_idx))

    def cull_new_points(self, current_kf_idx):
        """
        §VI-B Recent Map Points Culling, checked only during a point's first
        three keyframes after creation: remove it if its found/predicted-
        visible ratio is <= 25%, or - once more than one keyframe has passed
        since its creation - if it hasn't been observed from at least three
        keyframes. Call once per keyframe insertion, with that keyframe's
        own index. Returns the removed point indices.
        """
        elapsed = current_kf_idx - self.created_kf
        candidates = np.where(self.active & (elapsed >= 1) & (elapsed <= 3))[0]
        removed = []
        for idx in candidates:
            idx = int(idx)
            ratio = self.n_found[idx] / self.n_visible[idx] if self.n_visible[idx] > 0 else 0.0
            fails_ratio = ratio <= 0.25
            fails_observations = elapsed[idx] > 1 and self.n_observing_keyframes(idx) < 3
            if fails_ratio or fails_observations:
                removed.append(idx)
        self.remove_points(removed)
        return removed

    def cull_low_observation_points(self, current_kf_idx):
        """
        Ongoing §VI-B rule for points that already passed the initial
        three-keyframe test: remove any that have since dropped below three
        observing keyframes (e.g. via future keyframe culling or bundle
        adjustment marking observations as outliers - neither exists in
        this pipeline yet, so this is a no-op today; the rule itself is
        implemented so those can plug into it later without touching this
        method). Call once per keyframe insertion. Returns the removed
        point indices.
        """
        elapsed = current_kf_idx - self.created_kf
        candidates = np.where(self.active & (elapsed > 3))[0]
        removed = [int(idx) for idx in candidates if self.n_observing_keyframes(idx) < 3]
        self.remove_points(removed)
        return removed

    def add_observation(self, point_idx, kf_idx, camera_center, descriptor, octave, frame_idx=None):
        """
        Register one new (keyframe, point) observation - called for every
        keyframe that observes a map point, whether that's the point's
        initial triangulation (two observations: the reference and current
        keyframe of the pair it was triangulated from), a further founding
        observation from another covisible keyframe, or an already-existing
        point matched again by ordinary per-keyframe PnP tracking.

        Incrementally updates this point's §III-C metadata (viewing
        direction, scale-invariance bounds, representative descriptor) from
        its own observation list only, and the §III-D covisibility graph's
        edges to keyframes that already observe it - never a full
        recompute over the map's history.

        frame_idx, if given, is kf_idx's own ORB feature index behind this
        observation - recorded so keyframe_matched_frame_idx(kf_idx) can
        later tell a §VI-C new-point search which of kf_idx's features are
        already spoken for.
        """
        camera_center = np.asarray(camera_center, dtype=np.float64).ravel()
        # .copy(), not .asarray(): descriptor is typically a row-view into a
        # frame's full (n_features, 32) descriptor array (e.g. desc[fidx] in
        # _demo()) - storing the view as-is would keep that entire array
        # alive in memory for as long as this one 32-byte observation is
        # kept, for every observation ever registered.
        descriptor = np.array(descriptor, dtype=np.uint8, copy=True)
        obs = self._observations[point_idx]
        obs.append((kf_idx, camera_center, descriptor, int(octave), frame_idx))

        # Viewing direction: mean unit vector, observing camera center ->
        # point, across every observation (§III-C) - deliberately NOT
        # renormalized to unit length afterward (matches the paper: a
        # point observed from widely diverging angles ends up with a
        # shorter mean vector, which is exactly what makes the viewing-
        # angle gate v.n < cos(60 deg) meaningful downstream).
        point = self.points[point_idx]
        rays = np.array([point - o[1] for o in obs])
        unit_rays = rays / np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-9)
        self.viewing_direction[point_idx] = unit_rays.mean(axis=0)

        # Scale-invariance bounds: derived from THIS (latest) observation's
        # distance and pyramid octave only, matching the paper's own
        # incremental update - not an aggregate over the point's whole
        # observation history.
        dist = max(float(np.linalg.norm(point - camera_center)), 1e-9)
        level_scale = self.pyramid_scale_factor ** octave
        self.d_max[point_idx] = dist * level_scale
        self.d_min[point_idx] = self.d_max[point_idx] / (
            self.pyramid_scale_factor ** (self.pyramid_n_levels - 1)
        )

        # Representative descriptor: recomputed over every observation.
        descs = np.array([o[2] for o in obs])
        self.descriptors[point_idx] = _representative_descriptor(descs)

        # Covisibility graph: this point is now newly shared between
        # kf_idx and every other keyframe that already observed it, so
        # each such pair's edge weight goes up by exactly one.
        observing_kfs = self._point_keyframes[point_idx]
        for other_kf in observing_kfs:
            self._covisibility.setdefault(kf_idx, {})
            self._covisibility.setdefault(other_kf, {})
            self._covisibility[kf_idx][other_kf] = self._covisibility[kf_idx].get(other_kf, 0) + 1
            self._covisibility[other_kf][kf_idx] = self._covisibility[other_kf].get(kf_idx, 0) + 1
        observing_kfs.add(kf_idx)
        self._keyframe_points.setdefault(kf_idx, set()).add(point_idx)

        if frame_idx is not None:
            self._keyframe_matched_frame_idx.setdefault(kf_idx, set()).add(int(frame_idx))

    def keyframe_matched_frame_idx(self, kf_idx):
        """This keyframe's own ORB feature indices already tied to a map
        point (via any observation ever registered with a frame_idx)."""
        return set(self._keyframe_matched_frame_idx.get(kf_idx, ()))

    def covisible_keyframes(self, kf_idx, min_shared=None):
        """
        Keyframes connected to kf_idx in the covisibility graph (§III-D):
        those sharing at least `min_shared` (default: covisibility_min_shared)
        observed map points with it. Returns {other_kf_idx: shared_point_count}.
        """
        threshold = self.covisibility_min_shared if min_shared is None else min_shared
        edges = self._covisibility.get(kf_idx, {})
        return {k: w for k, w in edges.items() if w >= threshold}

    def keyframe_points(self, kf_idx):
        """Map point indices observed by this keyframe."""
        return set(self._keyframe_points.get(kf_idx, ()))

    def observing_keyframes(self, point_idx):
        """Keyframe indices that have observed this point."""
        return set(self._point_keyframes[point_idx])

    def n_observing_keyframes(self, point_idx):
        """Number of distinct keyframes that have observed this point - O(1),
        the direct-query form §VI-B's culling rules need every keyframe."""
        return len(self._point_keyframes[point_idx])

    def local_map_keyframes(self, seed_points, max_keyframes=30):
        """
        K1 union K2 (§V-D "Track Local Map"): K1 is every keyframe that
        observes any of `seed_points` (typically the map points matched in
        the most recently tracked frame), K2 is K1's covisibility-graph
        neighbors. Returns a set of keyframe indices.

        K1 is deliberately unthresholded (any shared point counts, per the
        paper), which means a single popular point can pull in every
        keyframe that has ever observed it - harmless for a large,
        rarely-revisited environment, but on a small or heavily-revisited
        scene K1 alone can end up covering most of the trajectory's
        keyframes, and K2 then expands that further. Capped to the
        max_keyframes most recent (highest-index) keyframes - both before
        expanding to K2 (so a huge K1 doesn't also pay to compute K2 for
        every one of its members) and on the final result - since recency
        is a reasonable proxy for "relevant to the current viewpoint" for
        a continuously-moving camera, and an uncapped local map defeats the
        entire point of scoping guided matching to a LOCAL map at all.
        """
        k1 = set()
        for p in seed_points:
            k1.update(self._point_keyframes[p])
        if len(k1) > max_keyframes:
            k1 = set(sorted(k1, reverse=True)[:max_keyframes])

        k2 = set()
        for kf in k1:
            k2.update(self.covisible_keyframes(kf))

        result = k1 | k2
        if len(result) > max_keyframes:
            result = set(sorted(result, reverse=True)[:max_keyframes])
        return result

    def local_map_points(self, keyframes):
        """Map point indices observed by any of `keyframes`."""
        points = set()
        for kf in keyframes:
            points.update(self._keyframe_points.get(kf, set()))
        return points

    def match_against(self, desc, ratio=0.75, mask=None):
        """
        Match frame descriptors against a subset of the map (every active -
        not yet §VI-B-culled - point by default, or a boolean mask over
        self.points, always further restricted to active points regardless
        of what mask is passed). Returns (map_indices, frame_indices) -
        map_indices are original (stable) indices, not positions within the
        subset.
        """
        from pipeline.features import match_descriptors

        subset_mask = self.active if mask is None else (mask & self.active)
        subset = np.where(subset_mask)[0]
        if len(subset) == 0 or desc is None or len(desc) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        matches = match_descriptors(self.descriptors[subset], desc, ratio)
        map_indices = subset[[m.queryIdx for m in matches]]
        frame_indices = np.array([m.trainIdx for m in matches], dtype=int)
        return map_indices, frame_indices

    def match_against_guided(self, kp, desc, camera_matrix, R_pred, t_pred,
                              window=25.0, ratio=0.75, mask=None, image_shape=None):
        """
        Guided variant of match_against, following the paper's §V-D "Track
        Local Map" projection sequence rather than a flat pixel window: for
        each candidate map point (subset - normally the local map, K1 union
        K2, see local_map_keyframes/local_map_points, always further
        restricted to active - not yet §VI-B-culled - points), project it
        into the frame with the predicted pose and discard it if it's behind
        the camera, if its viewing angle against the point's stored §III-C
        viewing direction exceeds 60 deg, if its predicted distance falls
        outside the point's stored [d_min, d_max] scale-invariance range, or
        (only when image_shape - (height, width) - is given) if it projects
        outside the frame. Every point still standing predicts the ORB
        pyramid octave it should now appear at (the paper's PredictScale,
        derived from d_max and how much the distance has changed since the
        point's own reference observation) - this sets both the candidate
        search radius (scaled by that octave's pyramid downsampling factor,
        so a point predicted at a coarser level searches a wider pixel
        radius) and an octave tolerance band ([predicted-1, predicted+1])
        candidate frame keypoints must fall within, on top of the existing
        spatial+ratio-test narrowing.

        Without image_shape, a point projecting outside the frame simply
        finds no real keypoint nearby and contributes no match - there's no
        separate bounds check in that case.

        Returns (map_indices, frame_indices, visible_indices) - map_indices/
        frame_indices as match_against; visible_indices are every candidate
        point that survived the in-front/angle/scale(/bounds) gates this
        call, whether or not it went on to find an actual match - §VI-B's
        "predicted visible" set (see Map.record_visible/record_found).
        """
        from scipy.spatial import cKDTree

        subset_mask = self.active if mask is None else (mask & self.active)
        subset = np.where(subset_mask)[0]
        if len(subset) == 0 or desc is None or len(desc) == 0 or len(kp) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int)

        rvec_pred, _ = cv2.Rodrigues(R_pred)
        proj, _ = cv2.projectPoints(self.points[subset], rvec_pred, t_pred, camera_matrix, None)
        proj = proj.reshape(-1, 2)

        cam_pts = (R_pred @ self.points[subset].T).T + t_pred.ravel()
        in_front = cam_pts[:, 2] > 0

        pred_center = camera_center(R_pred, t_pred).ravel()
        point_vecs = self.points[subset] - pred_center
        dist = np.linalg.norm(point_vecs, axis=1)

        unit_vecs = point_vecs / np.maximum(dist, 1e-9)[:, None]
        view_cos = np.einsum("ij,ij->i", unit_vecs, self.viewing_direction[subset])
        angle_ok = view_cos >= np.cos(np.radians(60.0))

        d_min, d_max = self.d_min[subset], self.d_max[subset]
        scale_ok = (dist >= d_min) & (dist <= d_max)

        candidate_mask = in_front & angle_ok & scale_ok
        if image_shape is not None:
            h, w = image_shape[0], image_shape[1]
            bounds_ok = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
            candidate_mask &= bounds_ok

        survivors = np.where(candidate_mask)[0]
        if len(survivors) == 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int), np.empty(0, dtype=int)

        visible_indices = subset[survivors]

        # Predicted pyramid octave (paper's PredictScale): the level at
        # which each surviving point should appear given its current
        # distance, derived from its own scale-invariance range - not a
        # single fixed level assumed for the whole map.
        scale_ratio = d_max[survivors] / dist[survivors]
        predicted_octave = np.clip(
            np.ceil(np.log(scale_ratio) / np.log(self.pyramid_scale_factor)),
            0, self.pyramid_n_levels - 1,
        ).astype(int)

        frame_pts = np.float32([k.pt for k in kp])
        frame_octaves = np.array([k.octave for k in kp])
        tree = cKDTree(frame_pts)
        radii = window * self.pyramid_scale_factor ** predicted_octave
        neighbor_lists = tree.query_ball_point(proj[survivors], r=radii)

        map_indices = []
        frame_indices = []
        for local_i, pred_level, neighbors in zip(survivors, predicted_octave, neighbor_lists):
            if not neighbors:
                continue
            cand = np.array([j for j in neighbors if abs(frame_octaves[j] - pred_level) <= 1])
            if len(cand) == 0:
                continue
            cand_desc = desc[cand]
            dists = _hamming_distances(self.descriptors[subset[local_i]], cand_desc)
            order = np.argsort(dists)
            if len(order) >= 2 and dists[order[0]] >= ratio * dists[order[1]]:
                continue
            map_indices.append(subset[local_i])
            frame_indices.append(int(cand[order[0]]))

        return np.array(map_indices, dtype=int), np.array(frame_indices, dtype=int), visible_indices


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


def _run_ba(keyframe_poses, keyframe_observations, sparse_map, camera_matrix,
            free_kfs, fixed_kfs, point_ids, max_points, max_nfev, ftol, xtol):
    """
    Shared BA plumbing for both _run_local_ba (§VI-D covisibility-scoped)
    and _run_global_ba (whole-trajectory-scoped): given an already-chosen
    set of free/fixed keyframes and points, build local_bundle_adjustment's
    inputs and call it - WITHOUT applying the result (see
    _apply_ba_result/_validate_and_apply_ba).

    free_kfs/fixed_kfs must partition every keyframe that observes any
    point in point_ids (the caller's responsibility - see
    _run_local_ba/_run_global_ba); fixed_kfs still constrains those points
    via their observations, just without their own pose being refined.

    Returns None if there wasn't enough data for a meaningful refinement,
    otherwise (free_kf_indices, refined_rotations, refined_translations,
    point_ids, refined_points) - free_kf_indices[i] is the GLOBAL keyframe
    index refined_rotations[i]/refined_translations[i] belongs to (sorted
    ascending, so free_kf_indices[-1] is always the most recent one -
    fixed keyframes' poses are never returned, since they don't change).

    If point_ids has more than max_points points (common right after a
    keyframe that added a lot of new structure at once), only the most
    recently added ones are kept - they're the ones most relevant to
    correcting recent drift, and this bounds the per-call cost so BA doesn't
    visibly stall the pipeline at every keyframe - but never at the cost of
    starving a free keyframe down to too few observations (an under-
    constrained pose, too few residuals for its 6 DOF, is far worse than a
    slightly larger optimization, and is what caused wild trajectory jumps
    here previously). Pass max_points=None to disable the cap entirely (see
    _run_global_ba, a one-shot end-of-run pass with no such time pressure).
    """
    from pipeline.bundle_adjustment import local_bundle_adjustment

    if len(point_ids) < 10:
        return None  # not enough constraints for a meaningful refinement

    if max_points is not None and len(point_ids) > max_points:
        full_point_set = set(point_ids)
        min_obs_per_keyframe = 15
        keep = set(point_ids[-max_points:])
        for kf in free_kfs:
            kf_points = sparse_map.keyframe_points(kf) & full_point_set
            target = min(min_obs_per_keyframe, len(kf_points))
            kept_here = len(kf_points & keep)
            if kept_here < target:
                candidates = sorted(kf_points - keep, reverse=True)
                keep.update(candidates[:target - kept_here])
        point_ids = sorted(keep)

    point_id_set = set(point_ids)
    all_kfs = sorted(set(free_kfs) | set(fixed_kfs))
    kf_to_local = {kf: i for i, kf in enumerate(all_kfs)}
    fixed_lookup = set(fixed_kfs)
    fixed_mask = np.array([kf in fixed_lookup for kf in all_kfs])
    id_to_local = {pid: i for i, pid in enumerate(point_ids)}
    local_points = sparse_map.points[point_ids].copy()

    local_observations = [
        (kf_to_local[kf], id_to_local[pid], x, y)
        for kf in all_kfs
        for pid, x, y in keyframe_observations[kf]
        if pid in point_id_set
    ]
    if len(local_observations) < 10:
        return None

    rotations = [keyframe_poses[k].R for k in all_kfs]
    translations = [keyframe_poses[k].t for k in all_kfs]

    refined_rot, refined_trans, refined_pts = local_bundle_adjustment(
        rotations, translations, local_points, local_observations, camera_matrix,
        fixed_poses=fixed_mask, max_nfev=max_nfev, ftol=ftol, xtol=xtol,
    )

    free_lookup = set(free_kfs)
    free_kf_indices = [kf for kf in all_kfs if kf in free_lookup]
    refined_by_kf = dict(zip(all_kfs, zip(refined_rot, refined_trans)))
    free_refined_rot = [refined_by_kf[kf][0] for kf in free_kf_indices]
    free_refined_trans = [refined_by_kf[kf][1] for kf in free_kf_indices]

    return free_kf_indices, free_refined_rot, free_refined_trans, point_ids, refined_pts


def _run_local_ba(keyframe_poses, keyframe_observations, sparse_map, camera_matrix,
                   max_points=300, max_nfev=1000, ftol=1e-4, xtol=1e-4):
    """
    Refine the current (most recently accepted) keyframe's covisibility-
    scoped local BA window (§VI-D): free keyframes Kl = the current
    keyframe union its covisibility-graph neighbors; local points = every
    point any of them observes; fixed keyframes Kf = every OTHER keyframe
    that also observes one of those points, held fixed so it still
    constrains the point without itself drifting - not simply dropped,
    which is what a fixed insertion-order sliding window did whenever a
    point's older observations fell outside it.

    If every keyframe that observes a local point is already inside Kl
    itself (Kf ends up empty - e.g. a brand-new keyframe whose points are
    all just-triangulated and not yet re-observed by anyone outside Kl),
    there would be no fixed keyframe left to anchor the gauge, so the
    oldest keyframe in Kl is held fixed instead (mirrors the single-gauge-
    anchor behavior local BA always had).

    Kl is capped to the current keyframe plus its 20 most recent
    covisibility neighbors - on a small or heavily-revisited scene, a
    single keyframe can have far more than 20 neighbors crossing the
    15-shared-point threshold, and refining that many keyframes jointly
    every single keyframe insertion is exactly the unbounded-per-keyframe-
    cost this rescoping exists to avoid (this is a real, measured
    regression on freiburg1_xyz, not a hypothetical).

    See _run_ba for the return value and max_points/return-None conventions.
    """
    if len(keyframe_poses) < 2:
        return None

    current_kf_idx = len(keyframe_poses) - 1
    neighbors = sorted(sparse_map.covisible_keyframes(current_kf_idx), reverse=True)[:20]
    local_kfs = {current_kf_idx} | set(neighbors)
    point_ids = sorted(sparse_map.local_map_points(local_kfs))
    if not point_ids:
        return None

    observing_kfs = set()
    for p in point_ids:
        observing_kfs.update(sparse_map.observing_keyframes(p))
    fixed_kfs = observing_kfs - local_kfs

    if not fixed_kfs:
        oldest = min(local_kfs)
        local_kfs = local_kfs - {oldest}
        fixed_kfs = {oldest}
        if not local_kfs:
            return None  # only one keyframe observes any local point - nothing to refine

    return _run_ba(keyframe_poses, keyframe_observations, sparse_map, camera_matrix,
                    sorted(local_kfs), sorted(fixed_kfs), point_ids,
                    max_points, max_nfev, ftol, xtol)


def _run_global_ba(keyframe_poses, keyframe_observations, sparse_map, camera_matrix,
                    max_nfev=8000, ftol=1e-6, xtol=1e-6):
    """
    Full BA (paper Appendix; used offline in §VIII-E/Table VI): every
    keyframe (except keyframe 0, held fixed as the sole gauge anchor) and
    every map point, jointly - unlike _run_local_ba, not scoped to the
    current keyframe's covisibility neighborhood, since a one-shot
    end-of-run pass should refine the whole trajectory at once.

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
    if len(keyframe_poses) < 2:
        return None
    free_kfs = list(range(1, len(keyframe_poses)))
    fixed_kfs = [0]
    # keyframe_observations still carries stale entries for any point §VI-B
    # has since culled (Map.remove_points only cleans the Map's own
    # _keyframe_points/_covisibility, not this demo-local list) - filter
    # against sparse_map.active so a removed point can't leak back into a
    # full-trajectory BA pass as a live constraint.
    point_ids = sorted(
        {obs[0] for kf_obs in keyframe_observations for obs in kf_obs}
        & set(np.where(sparse_map.active)[0].tolist())
    )
    return _run_ba(keyframe_poses, keyframe_observations, sparse_map, camera_matrix,
                    free_kfs, fixed_kfs, point_ids, max_points=None,
                    max_nfev=max_nfev, ftol=ftol, xtol=xtol)


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
        # Same stale-entry issue as _run_global_ba's point_ids: drop any
        # observation of a point §VI-B has since culled.
        kf_obs = [o for o in kf_obs if sparse_map.active[o[0]]]
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
    free_kf_indices, refined_rot, refined_trans, point_ids, refined_pts = ba_result
    for kf_idx, R_ref, t_ref in zip(free_kf_indices, refined_rot, refined_trans):
        keyframe_poses[kf_idx] = keyframe_poses[kf_idx]._replace(R=R_ref, t=t_ref)
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

    free_kf_indices, refined_rot, refined_trans, _, _ = ba_result

    for kf_idx, new_R, new_t in zip(free_kf_indices, refined_rot, refined_trans):
        old_R, old_t, _ = keyframe_poses[kf_idx]
        rot_change = rotation_angle_deg(new_R @ old_R.T)
        step_change = np.linalg.norm(
            camera_center(new_R, new_t) - camera_center(old_R, old_t)
        )
        implausible = rot_change > max_plausible_rotation or (
            len(recent_step_sizes) >= 5
            and step_change > max_step_ratio * np.median(recent_step_sizes)
        )
        if implausible:
            print(f"    [BA result rejected: implausible pose change at keyframe {kf_idx} "
                  f"(rotation={rot_change:.1f}deg, step={step_change:.2f})]")
            return fallback_R, fallback_t

    new_R, new_t = refined_rot[-1], refined_trans[-1]

    _apply_ba_result(keyframe_poses, sparse_map, ba_result)
    return new_R, new_t


def _run_depth_densify(frame_image, R_pos, t_pos, sparse_map, map_indices, image_points,
                        pnp_inlier_mask, camera_matrix, depth_estimator, depth_rows, scan_stride):
    """
    Estimates ML depth for this keyframe, fits it (scale + shift) against
    the map points PnP just matched (map_indices/image_points, restricted to
    pnp_inlier_mask - the same trusted set the pose itself was solved
    against), then back-projects a sample of scanline pixels into world
    points using that fit.

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
    # the calibration fit still uses every matched point regardless of
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
        print(f"    [depth-densify: only {len(disparity)} points visible - "
              f"skipping (need >= 10 for a stable fit)]")
        return np.empty((0, 3)), raw_depth, background_mask

    a, b = fit

    # A degenerate fit (a close to 0 - the calibration points barely span any
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
    # calibration point that's unusually far (or just noisy) sets the boundary
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
          f"from {len(disparity)} points, {len(new_points)} ML points sampled]")
    return new_points, raw_depth, background_mask


def _keyframe_positions(keyframe_poses):
    """Camera centers (world coordinates) for every keyframe, as an (N, 3)
    array - shared by the live-display loop and the final matplotlib plot."""
    return np.array(
        [camera_center(pose.R, pose.t) for pose in keyframe_poses]
    ).reshape(-1, 3)


def _fundamental_matrix(R1, t1, R2, t2, camera_matrix):
    """Fundamental matrix between two ALREADY-KNOWN world-to-camera poses
    (not estimated from correspondences, unlike pose.estimate_relative_pose)
    - satisfies x2^T F x1 ~= 0 for a true correspondence (x1 in camera 1,
    x2 in camera 2), used by _epipolar_line_distance to gate §VI-C
    candidate correspondences before they're triangulated."""
    R_rel = R2 @ R1.T
    t_rel = (t2 - R_rel @ t1).ravel()
    t_x = np.array([
        [0, -t_rel[2], t_rel[1]],
        [t_rel[2], 0, -t_rel[0]],
        [-t_rel[1], t_rel[0], 0],
    ])
    E = t_x @ R_rel
    K_inv = np.linalg.inv(camera_matrix)
    return K_inv.T @ E @ K_inv


def _epipolar_line_distance(F, pts1, pts2):
    """Distance (px) from each pts2 point to the epipolar line F projects
    its corresponding pts1 point onto - near zero for a true correspondence,
    large for a false one (e.g. a repeated texture pattern matched to the
    wrong instance of itself in another keyframe)."""
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])
    lines2 = pts1_h @ F.T
    norm = np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2) + 1e-12
    return np.abs(np.sum(lines2 * pts2_h, axis=1)) / norm


def _covisible_candidates(sparse_map, kf_idx, fallback_kf_idx, max_keyframes):
    """
    Keyframes to search for §VI-C new-point correspondences against:
    kf_idx's covisibility-graph neighbors, most-shared-points-first, capped
    to max_keyframes. Falls back to including fallback_kf_idx (the
    reference keyframe kf_idx was tracked against) if the graph doesn't
    have edges for kf_idx yet crossing covisibility_min_shared - e.g. right
    after bootstrap, before any keyframe pair has accumulated that many
    shared points - mirroring local_map_keyframes' same fallback, so a
    brand-new keyframe is never left with zero candidates to triangulate
    against (today's single-previous-keyframe behavior, at minimum).
    """
    edges = sparse_map.covisible_keyframes(kf_idx)
    ranked = sorted(edges, key=lambda k: edges[k], reverse=True)
    if fallback_kf_idx != kf_idx and fallback_kf_idx not in ranked:
        ranked = [fallback_kf_idx] + ranked
    return ranked[:max_keyframes]


def _create_new_points_from_covisible_keyframes(
        sparse_map, new_kf_idx, R_new, t_new, kp_new, desc_new,
        already_matched_frame_idx, covisible_candidates,
        keyframe_poses, keyframe_kp, keyframe_desc, camera_matrix,
        ratio, min_triangulation_angle, epipolar_max_error):
    """
    §VI-C new-point creation: for each of new_kf_idx's covisible keyframes
    (covisible_candidates, most-shared-points-first, already capped by the
    caller - see _covisible_candidates), match new_kf_idx's still-unmatched
    ORB features against that keyframe's own still-unmatched features
    (brute-force Hamming + ratio test, features.match_descriptors), discard
    candidate correspondences that don't satisfy the epipolar constraint
    between the two keyframes' already-solved poses, and triangulate the
    survivors (triangulation.triangulate, unchanged - its cheirality/
    parallax checks remain the acceptance criteria).

    A new-keyframe feature already used for a point triangulated against one
    covisible keyframe is excluded from candidate matching against the next
    one, so the same feature can't spawn two different map points in a
    single call.

    Returns (point_ids, source_kf, source_pixels, new_kf_frame_idx,
    new_kf_pixels) - five parallel lists, one entry per newly created point.
    source_kf/source_pixels describe the COVISIBLE keyframe's side of each
    point's founding observation (already registered here directly, since
    that keyframe's index/pose/descriptors are already fully known);
    new_kf_frame_idx/new_kf_pixels describe new_kf_idx's own side, left for
    the caller to register alongside every other observation it makes
    (matched, re-observed, and newly triangulated alike).
    """
    from pipeline.features import match_descriptors
    from pipeline.triangulation import triangulate

    point_ids, source_kf, source_pixels = [], [], []
    new_kf_frame_idx, new_kf_pixels = [], []
    claimed_new = set(already_matched_frame_idx)

    for kf_i in covisible_candidates:
        R_i, t_i, _ = keyframe_poses[kf_i]
        kp_i, desc_i = keyframe_kp[kf_i], keyframe_desc[kf_i]
        matched_i = sparse_map.keyframe_matched_frame_idx(kf_i)

        free_new = np.array(
            [i for i in range(len(kp_new)) if i not in claimed_new], dtype=int
        )
        free_i = np.array(
            [i for i in range(len(kp_i)) if i not in matched_i], dtype=int
        )
        if len(free_new) == 0 or len(free_i) == 0:
            continue

        matches = match_descriptors(desc_new[free_new], desc_i[free_i], ratio)
        if len(matches) == 0:
            continue

        q_idx = free_new[[m.queryIdx for m in matches]]
        t_idx = free_i[[m.trainIdx for m in matches]]
        pts_new = np.float32([kp_new[i].pt for i in q_idx])
        pts_i = np.float32([kp_i[i].pt for i in t_idx])

        F = _fundamental_matrix(R_i, t_i, R_new, t_new, camera_matrix)
        epi_ok = _epipolar_line_distance(F, pts_i, pts_new) < epipolar_max_error
        if not np.any(epi_ok):
            continue
        q_idx, t_idx = q_idx[epi_ok], t_idx[epi_ok]
        pts_new, pts_i = pts_new[epi_ok], pts_i[epi_ok]

        points_3d, valid, _, _ = triangulate(
            R_i, t_i, R_new, t_new, camera_matrix, pts_i, pts_new,
            min_parallax_deg=min_triangulation_angle,
        )
        if not np.any(valid):
            continue

        kept_q, kept_t = q_idx[valid], t_idx[valid]
        kept_pts_new, kept_pts_i = pts_new[valid], pts_i[valid]
        kept_points = points_3d[valid]

        base_idx = len(sparse_map)
        sparse_map.add_points(kept_points, desc_new[kept_q], created_kf=new_kf_idx)
        ids = list(range(base_idx, base_idx + len(kept_points)))

        i_center = camera_center(R_i, t_i)
        for pid, fidx_i in zip(ids, kept_t):
            sparse_map.add_observation(
                pid, kf_i, i_center, desc_i[int(fidx_i)], kp_i[int(fidx_i)].octave,
                frame_idx=int(fidx_i),
            )

        point_ids.extend(ids)
        source_kf.extend([kf_i] * len(ids))
        source_pixels.extend((float(x), float(y)) for x, y in kept_pts_i)
        new_kf_frame_idx.extend(int(f) for f in kept_q)
        new_kf_pixels.extend((float(x), float(y)) for x, y in kept_pts_new)
        claimed_new.update(int(f) for f in kept_q)

    return point_ids, source_kf, source_pixels, new_kf_frame_idx, new_kf_pixels


def _extend_new_points_to_other_covisible_keyframes(
        sparse_map, point_ids, source_kf, covisible_candidates,
        keyframe_poses, keyframe_kp, keyframe_desc, camera_matrix, window, ratio):
    """
    §VI-C last paragraph: project each just-created point into every
    covisible keyframe OTHER than the one it was actually triangulated from
    and search for a further correspondence, reusing
    Map.match_against_guided's §V-D Track Local Map projection/matching
    (viewing-angle/scale-invariance gating, octave-aware search radius)
    rather than a separate search. Each match found is registered as an
    additional observation of the point - not fed into §VI-B's found/
    predicted-visible bookkeeping, which is specifically about per-frame
    tracking, not this keyframe-level founding search.

    Returns the total number of extra observations found.
    """
    if not point_ids:
        return 0

    by_source = {}
    for pid, kf in zip(point_ids, source_kf):
        by_source.setdefault(kf, []).append(pid)

    n_extra = 0
    for kf_i, ids in by_source.items():
        mask = np.zeros(len(sparse_map), dtype=bool)
        mask[ids] = True
        for kf_j in covisible_candidates:
            if kf_j == kf_i:
                continue
            R_j, t_j, _ = keyframe_poses[kf_j]
            kp_j, desc_j = keyframe_kp[kf_j], keyframe_desc[kf_j]
            map_idx, frame_idx, _ = sparse_map.match_against_guided(
                kp_j, desc_j, camera_matrix, R_j, t_j, window=window, ratio=ratio, mask=mask,
            )
            if len(map_idx) == 0:
                continue
            j_center = camera_center(R_j, t_j)
            for pid, fidx in zip(map_idx, frame_idx):
                sparse_map.add_observation(
                    pid, kf_j, j_center, desc_j[int(fidx)], kp_j[int(fidx)].octave,
                    frame_idx=int(fidx),
                )
            n_extra += len(map_idx)

    return n_extra


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
                     "(triangulation/culling/BA) once the paper's own §V-E "
                     "multi-condition policy says one is needed"
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
                         help="Minimum median pixel displacement vs the reference frame "
                              "before bootstrap accepts a two-view essential-matrix pose - "
                              "avoids the degenerate/ill-conditioned solve pure-rotation "
                              "motion produces. Does not gate TRACK-branch keyframe "
                              "promotion (that's --kf-* below, the paper's own §V-E policy - "
                              "an earlier version of this pipeline also required this here, "
                              "but measured worse on every metric; see EVALUATION_RESULTS.md) "
                              "or whether a pose is estimated at all once tracking a map (px)")
    parser.add_argument("--kf-min-tracked-points", type=int, default=50,
                         help="Paper §V-E condition: minimum PnP-tracked map "
                              "points a frame must have before it can be promoted to a "
                              "keyframe")
    parser.add_argument("--kf-ref-ratio", type=float, default=0.9,
                         help="Paper §V-E condition: promote a frame to a keyframe only "
                              "if it tracks fewer than this fraction of the map points "
                              "the current reference keyframe itself tracked via PnP when "
                              "it was inserted (vacuously satisfied until the reference "
                              "keyframe is itself a PnP-tracked one, e.g. right after "
                              "bootstrap)")
    parser.add_argument("--kf-min-frames-since-relocalization", type=int, default=20,
                         help="Paper §V-E condition: only promote a keyframe more than "
                              "this many frames after the last global relocalization - "
                              "always satisfied today, since relocalization (#13) doesn't "
                              "exist yet in this pipeline")
    parser.add_argument("--kf-max-frames-since-keyframe", type=int, default=20,
                         help="Force a keyframe insertion (bypassing every other §V-E "
                              "condition) after this many frames with none accepted - "
                              "stands in for the paper's "
                              "'local mapping idle' condition, which has no meaning here "
                              "since there's no separate mapping thread")
    parser.add_argument("--guided-window", type=float, default=60.0,
                         help="Pixel radius around each map point's motion-"
                              "predicted projection to search for a descriptor match "
                              "during per-frame PnP tracking, replacing an unguided "
                              "full-frame search (see Map.match_against_guided)")
    parser.add_argument("--track-local-map-max-points", type=int, default=6000,
                         help="Cap on the §V-D Track Local Map point set (K1 union K2's "
                              "observed points) guided matching searches against - keeps "
                              "the most recently added points if exceeded. On a small or "
                              "heavily-revisited scene, even a handful of keyframes can "
                              "together observe most of the map, so bounding "
                              "keyframe count alone (see Map.local_map_keyframes) isn't "
                              "enough to keep this per-frame search cost bounded")
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
    parser.add_argument("--epipolar-max-error", type=float, default=2.0,
                         help="Max distance (px) a §VI-C candidate correspondence's point may "
                              "fall from its epipolar line (computed from the two keyframes' "
                              "already-solved poses) before it's discarded, prior to "
                              "triangulation - see mapping._epipolar_line_distance")
    parser.add_argument("--new-point-max-covisible-keyframes", type=int, default=10,
                         help="Cap on how many of a new keyframe's covisibility-graph "
                              "neighbors (most shared points first) §VI-C new-point creation "
                              "searches for correspondences against - bounds the added "
                              "per-keyframe cost of matching+epipolar-checking against "
                              "multiple keyframes instead of just the previous one")
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
    parser.add_argument("--ba-every", type=int, default=1,
                         help="Only run local bundle adjustment every Nth accepted keyframe "
                              "(default: every keyframe)")
    parser.add_argument("--ba-max-points", type=int, default=300,
                         help="Cap on how many of the local BA scope's points (§VI-D: the "
                              "current keyframe + its covisibility-graph neighbors) a single "
                              "BA call refines (keeps the most recently added ones if exceeded)")
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
                              "(scale + shift) against the map's own points, "
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
    sparse_map = Map()

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
    # keyframe_kp[i]/keyframe_desc[i]: the i-th keyframe's full ORB
    # detection output (not just the subset tied to map points), kept for
    # EVERY keyframe rather than just the current reference one - §VI-C new-
    # point creation needs to search for correspondences against any of a
    # new keyframe's covisible neighbors, not only the immediately
    # preceding keyframe.
    keyframe_kp = [None]
    keyframe_desc = [None]

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
    # §VI-B culling totals across the whole run, for the final report.
    total_culled_trial = 0
    total_culled_ongoing = 0

    # §V-D Track Local Map state: ref_kf_idx is the keyframe index backing
    # ref_kp/ref_desc/ref_R/ref_t (kept in lockstep with them below);
    # last_tracked_map_indices is the set of map points matched in the most
    # recently *accepted* tracked frame - together these seed
    # local_map_keyframes' K1 (falling back to just the reference keyframe
    # before any frame has been tracked against this map yet).
    ref_kf_idx = 0
    last_tracked_map_indices = None

    # Paper §V-E new-keyframe policy state (TRACK branch only - bootstrap
    # keeps its own unconditional parallax-gated promotion). frames_since_
    # last_keyframe drives the max-frames fallback that stands in for the
    # paper's "local mapping idle" condition (no separate mapping thread
    # here - see --kf-max-frames-since-keyframe). frames_since_relocalization
    # drives condition 1 - there's no relocalization in this pipeline yet
    # (#13), so it starts with a head start large enough that the condition
    # is always satisfied; #13 can wire in a real reset-on-relocalization
    # event later without reworking this policy. ref_kf_tracked_count is
    # how many map points the current reference keyframe itself tracked via
    # PnP when it was inserted (condition 4's baseline) - None for a
    # bootstrap-created reference keyframe, which never ran PnP against a
    # prior map, so condition 4 is vacuously satisfied until a real
    # PnP-tracked keyframe becomes the reference.
    frames_since_last_keyframe = 0
    frames_since_relocalization = 10**9
    ref_kf_tracked_count = None

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
                keyframe_kp[0], keyframe_desc[0] = kp, desc
                continue

            frames_since_last_keyframe += 1
            frames_since_relocalization += 1

            matches_ref = match_descriptors(ref_desc, desc, args.ratio)

            status = "insufficient matches"
            parallax = 0.0
            is_keyframe = False
            pnp_inliers = None
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

                            new_kf_idx = len(keyframe_poses)  # kf 1 - about to be appended below
                            base_idx = len(sparse_map)
                            sparse_map.add_points(new_points[valid], new_desc, created_kf=new_kf_idx)
                            new_ids = list(range(base_idx, base_idx + int(valid.sum())))
                            obs_ref = pts1[inlier_mask][valid]
                            obs_cur = pts2[inlier_mask][valid]
                            keyframe_observations[0].extend(
                                (pid, x, y) for pid, (x, y) in zip(new_ids, obs_ref)
                            )
                            keyframe_observations.append(
                                [(pid, x, y) for pid, (x, y) in zip(new_ids, obs_cur)]
                            )

                            # §III-C/III-D bookkeeping: each new point's two
                            # founding observations (kf 0 = ref, kf 1 = this
                            # frame) - registered here (rather than folded
                            # into keyframe_observations above) since it also
                            # needs each observation's descriptor/pyramid
                            # octave, not just its pixel position.
                            ref_frame_idx = [m.queryIdx for m in kept_matches]
                            cur_frame_idx = [m.trainIdx for m in kept_matches]
                            ref_center = camera_center(R_pos, t_pos)
                            cur_center = camera_center(R_new, t_new)
                            for pid, fidx in zip(new_ids, ref_frame_idx):
                                sparse_map.add_observation(
                                    pid, 0, ref_center, ref_desc[fidx], ref_kp[fidx].octave,
                                    frame_idx=fidx,
                                )
                            for pid, fidx in zip(new_ids, cur_frame_idx):
                                sparse_map.add_observation(
                                    pid, new_kf_idx, cur_center, desc[fidx], kp[fidx].octave,
                                    frame_idx=fidx,
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
                            keyframe_kp.append(kp)
                            keyframe_desc.append(desc)
                            culled_trial = sparse_map.cull_new_points(new_kf_idx)
                            culled_ongoing = sparse_map.cull_low_observation_points(new_kf_idx)
                            total_culled_trial += len(culled_trial)
                            total_culled_ongoing += len(culled_ongoing)
                            if not args.no_ba and len(keyframe_poses) % args.ba_every == 0:
                                ba_result = _run_local_ba(keyframe_poses, keyframe_observations,
                                                           sparse_map, K,
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
                # attempted every frame against every active (not yet §VI-B-
                # culled) point. --min-parallax plays no role here at all -
                # it only gates bootstrap's essential-matrix estimation above;
                # whether this frame's tracked pose *also* gets promoted to a
                # keyframe is decided purely by the §V-E policy below. ---
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

                # §V-D Track Local Map: K1 (keyframes sharing points with the
                # most recently tracked frame, or just the reference keyframe
                # if none has been tracked yet against this map) union K2
                # (K1's covisibility-graph neighbors) - bounds guided
                # matching to a subset of the map that stays a roughly
                # constant size as the map grows, instead of every point
                # ever added. match_against_guided further restricts this to
                # active (not yet §VI-B-culled) points regardless.
                local_seed = (
                    last_tracked_map_indices if last_tracked_map_indices is not None
                    else np.empty(0, dtype=int)
                )
                local_kfs = sparse_map.local_map_keyframes(local_seed) | {ref_kf_idx}
                local_points = sparse_map.local_map_points(local_kfs)
                if len(local_points) > args.track_local_map_max_points:
                    local_points = set(
                        sorted(local_points, reverse=True)[:args.track_local_map_max_points]
                    )
                local_mask = np.zeros(len(sparse_map), dtype=bool)
                if local_points:
                    local_mask[list(local_points)] = True

                map_indices, frame_indices, visible_indices = sparse_map.match_against_guided(
                    kp, desc, K, R_pred, t_pred,
                    window=effective_window, ratio=args.ratio, mask=local_mask,
                    image_shape=frame.image.shape,
                )
                # §VI-B bookkeeping: every frame this point was predicted
                # visible (a Track Local Map candidate that survived the
                # in-front/angle/scale/bounds gates) counts toward the
                # denominator, whether or not it was actually matched below;
                # every frame it was actually matched counts toward the
                # numerator - independent of whether PnP goes on to accept
                # this frame's pose at all.
                sparse_map.record_visible(visible_indices)
                sparse_map.record_found(map_indices)
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
                            last_tracked_map_indices = map_indices[pnp_inlier_mask]
                            prev_pose = (R_pos, t_pos)
                            prev_pose_frame = cur_pose_frame
                            R_pos, t_pos = R_new, t_new
                            cur_pose_frame = frame.index

                            # Paper §V-E new-keyframe policy: the max-frames
                            # fallback forces promotion regardless of the other
                            # three conditions (standing in for "local mapping
                            # idle" - no separate mapping thread here); otherwise
                            # all of "long enough since relocalization" (#13, a
                            # no-op today), "tracks enough map points", and
                            # "tracks meaningfully less than the reference
                            # keyframe did" must hold. Deliberately NO separate
                            # --min-parallax requirement on top here - the paper
                            # doesn't have one either: CreateNewMapPoints' own
                            # per-point parallax-angle rejection (this
                            # codebase's equivalent: triangulate's
                            # min_parallax_deg, already running per covisible-
                            # keyframe pair since #24) is what actually protects
                            # triangulation, not a whole-frame proxy at the
                            # keyframe-decision level. An earlier version of
                            # this policy kept --min-parallax as an extra gate
                            # here "to be safe"; measured directly on
                            # freiburg1_xyz (#25's own required evaluation, see
                            # EVALUATION_RESULTS.md's ablation), that gate was
                            # actively counterproductive - dropping it improved
                            # ATE (-17%), RPE (-14%), AND more than halved full
                            # tracking-loss frames (77->36/798), because it
                            # blocked refreshing a stale reference keyframe
                            # exactly when condition 4 said tracking had
                            # degraded for reasons unrelated to translation
                            # (blur, lighting, slow rotation) - working against
                            # §V-E's own aggressive-insertion-for-robustness
                            # intent.
                            kf_max_frames_reached = (
                                frames_since_last_keyframe >= args.kf_max_frames_since_keyframe
                            )
                            kf_since_reloc_ok = (
                                frames_since_relocalization
                                > args.kf_min_frames_since_relocalization
                            )
                            kf_min_tracked_ok = pnp_inliers >= args.kf_min_tracked_points
                            kf_ref_ratio_ok = (
                                ref_kf_tracked_count is None
                                or pnp_inliers < args.kf_ref_ratio * ref_kf_tracked_count
                            )
                            kf_policy_ok = kf_max_frames_reached or (
                                kf_since_reloc_ok and kf_min_tracked_ok and kf_ref_ratio_ok
                            )

                            if has_ref_baseline and kf_policy_ok:
                                # --- Promote to keyframe: confirm provisional
                                # points, create new structure (§VI-C: searched
                                # across every covisible keyframe, not just the
                                # previous one - see _create_new_points_from_
                                # covisible_keyframes), run BA ---
                                new_kf_idx = len(keyframe_poses)
                                new_center = camera_center(R_pos, t_pos)

                                this_kf_observations = [
                                    (int(idx), float(x), float(y))
                                    for idx, (x, y) in zip(
                                        map_indices[pnp_inlier_mask],
                                        image_points[pnp_inlier_mask],
                                    )
                                ]
                                # Parallel to this_kf_observations - the current frame's
                                # keypoint index behind each entry, so §III-C/III-D
                                # bookkeeping below can look up each observation's
                                # descriptor/pyramid octave once the full list (matched +
                                # re-observed + newly triangulated) is assembled.
                                this_kf_frame_idx = frame_indices[pnp_inlier_mask].tolist()

                                # Register this keyframe's observations of already-
                                # existing map points FIRST, before anything else - the
                                # §VI-C new-point search below needs the covisibility
                                # graph to already carry real edges for new_kf_idx (built
                                # from exactly these shared-point observations) so it
                                # knows which keyframes are actually its neighbors.
                                for (pid, x, y), fidx in zip(this_kf_observations, this_kf_frame_idx):
                                    sparse_map.add_observation(
                                        pid, new_kf_idx, new_center, desc[fidx], kp[fidx].octave,
                                        frame_idx=fidx,
                                    )

                                # §VI-C: search for new-point correspondences across
                                # EVERY keyframe connected to this one in the
                                # covisibility graph (not just the immediately
                                # preceding one). Excludes, on this keyframe's side,
                                # features already tied to an existing map point
                                # (matched above); each covisible keyframe's own
                                # already-matched features are excluded on its side
                                # inside the helper itself.
                                already_matched_frame_idx = set(frame_indices.tolist())
                                covisible_candidates = _covisible_candidates(
                                    sparse_map, new_kf_idx, ref_kf_idx,
                                    args.new_point_max_covisible_keyframes,
                                )
                                (new_point_ids, new_source_kf, new_source_pixels,
                                 new_kf_frame_idx_new, new_kf_pixels_new) = (
                                    _create_new_points_from_covisible_keyframes(
                                        sparse_map, new_kf_idx, R_new, t_new, kp, desc,
                                        already_matched_frame_idx, covisible_candidates,
                                        keyframe_poses, keyframe_kp, keyframe_desc, K,
                                        args.ratio, args.min_triangulation_angle,
                                        args.epipolar_max_error,
                                    )
                                )
                                new_count = len(new_point_ids)

                                # Each new point's covisible-keyframe-side observation
                                # was already registered inside the helper above; register
                                # its OTHER side (this new keyframe's own observation)
                                # and this keyframe's raw pixel record for BA.
                                for pid, kf_i, (xi, yi) in zip(new_point_ids, new_source_kf, new_source_pixels):
                                    keyframe_observations[kf_i].append((pid, xi, yi))
                                for pid, fidx, (x, y) in zip(new_point_ids, new_kf_frame_idx_new, new_kf_pixels_new):
                                    sparse_map.add_observation(
                                        pid, new_kf_idx, new_center, desc[fidx], kp[fidx].octave,
                                        frame_idx=fidx,
                                    )
                                this_kf_observations.extend(
                                    (pid, x, y) for pid, (x, y) in zip(new_point_ids, new_kf_pixels_new)
                                )
                                this_kf_frame_idx.extend(new_kf_frame_idx_new)

                                # §VI-C last paragraph: project each just-created point
                                # into the covisible keyframes it WASN'T triangulated
                                # from and search for a further correspondence there too
                                # - reuses match_against_guided (§V-D Track Local Map)
                                # rather than a separate search. Each additional match
                                # counts as one more observation of the point.
                                n_extra_obs = _extend_new_points_to_other_covisible_keyframes(
                                    sparse_map, new_point_ids, new_source_kf, covisible_candidates,
                                    keyframe_poses, keyframe_kp, keyframe_desc, K,
                                    args.guided_window, args.ratio,
                                )

                                kf_trigger = (
                                    "max-frames-fallback" if kf_max_frames_reached
                                    else "paper-conditions"
                                )
                                print(f"frame {frame.index}: KEYFRAME  parallax={parallax:.1f}px  "
                                      f"rotation={rot_deg:.1f}deg  "
                                      f"{pnp_inliers}/{len(map_indices)} PnP inliers, "
                                      f"trigger={kf_trigger}  "
                                      f"{new_count} new points from "
                                      f"{len(set(new_source_kf))}/{len(covisible_candidates)} covisible "
                                      f"keyframes searched, {n_extra_obs} extra observations "
                                      f"({len(sparse_map)} total map points)")

                                keyframe_poses.append(KeyframePose(R_pos, t_pos, frame.timestamp))
                                keyframe_kp.append(kp)
                                keyframe_desc.append(desc)
                                keyframe_observations.append(this_kf_observations)
                                # §VI-B Recent Map Points Culling - run once per
                                # keyframe insertion, on this keyframe's own index.
                                culled_trial = sparse_map.cull_new_points(new_kf_idx)
                                culled_ongoing = sparse_map.cull_low_observation_points(new_kf_idx)
                                total_culled_trial += len(culled_trial)
                                total_culled_ongoing += len(culled_ongoing)
                                print(f"    [culling: {len(culled_trial)} removed by the "
                                      f"first-3-keyframe test, {len(culled_ongoing)} removed by "
                                      f"the ongoing <3-observing-keyframe rule "
                                      f"({sparse_map.n_active} active / {len(sparse_map)} total "
                                      f"map points)]")
                                if not args.no_ba and len(keyframe_poses) % args.ba_every == 0:
                                    ba_result = _run_local_ba(keyframe_poses, keyframe_observations,
                                                               sparse_map, K,
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
                        sparse_map.points[sparse_map.active],
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
                ref_kf_idx = len(keyframe_poses) - 1
                frames_since_last_keyframe = 0
                # None for a bootstrap-created keyframe (pnp_inliers stays
                # None all frame - no PnP was run against a prior map), which
                # keeps condition 4 vacuously satisfied until a real
                # PnP-tracked keyframe becomes the reference.
                ref_kf_tracked_count = pnp_inliers
            else:
                n_skipped += 1

    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n{n_keyframes} keyframes accepted "
          f"({sparse_map.n_active} active / {len(sparse_map)} total map points - "
          f"{total_culled_trial} removed by the first-3-keyframe test, "
          f"{total_culled_ongoing} removed by the ongoing <3-observing-keyframe rule), "
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
        active_pts = sparse_map.points[sparse_map.active]
        if len(active_pts) > 0:
            ax.scatter(active_pts[:, 0], active_pts[:, 2],
                       c="black", s=4, label="map points", zorder=1)
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