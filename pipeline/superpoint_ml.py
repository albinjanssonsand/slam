"""
Learned local feature detection via a pretrained SuperPoint ONNX checkpoint
(fabio-sim/LightGlue-ONNX's standalone export - github.com/fabio-sim/
LightGlue-ONNX/releases/tag/v0.1.3, superpoint.onnx), run per-frame on CPU
through onnxruntime - see pipeline/depth_ml.py's DepthEstimator for the
ONNX inference pattern this mirrors.

SuperPoint is single-scale (one forward pass over the full-resolution
image, no image pyramid) - unlike ORB, which this codebase's
pyramid_scale_factor/pyramid_n_levels machinery (Map.d_min/d_max,
match_against_guided's PredictScale, #33's octave-aware chi-squared
thresholds in triangulation.py/bundle_adjustment.py) all assume. Issue
#38's resolution, chosen over running the model across an explicit
pyramid (rejected - see the issue for why that repeats #21/#58's
duplicate-detection and per-frame-cost regressions on a second detector):
bucket the model's own per-keypoint confidence score into a synthetic
octave, so every existing octave-aware consumer keeps working unchanged.
See _score_to_octave.
"""

import cv2
import numpy as np
import onnxruntime as ort


class SuperPointEstimator:
    """Wraps a standalone SuperPoint ONNX checkpoint (keypoints+scores+
    descriptors from a single grayscale image) for per-frame learned
    feature detection on CPU."""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name

    def detect_and_compute(self, image, n_features=2000, pyramid_n_levels=8):
        """
        Returns (keypoints, descriptors) in the same shape ORB's
        detect_and_compute/detect_and_compute_gridded return: keypoints a
        list of real cv2.KeyPoint (required - cv2.drawMatches and every
        other .pt/.octave reader in this codebase expects the real C++
        type, not a duck-typed substitute), descriptors an (N, 256)
        np.float32 array of L2-normalized embeddings (already normalized
        by the checkpoint itself).

        Only the n_features highest-confidence keypoints are kept,
        mirroring ORB's n_features budget - the checkpoint's own NMS/
        border-removal already ran inside the graph, but places no cap on
        count, and an unbounded count would make downstream matching
        (brute-force or LightGlue's O(N*M) attention) far more expensive
        than the ORB path it replaces.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        chw = (gray.astype(np.float32) / 255.0)[None, None]

        keypoints, scores, descriptors = self.session.run(None, {self._input_name: chw})
        keypoints, scores, descriptors = keypoints[0], scores[0], descriptors[0]

        if len(keypoints) == 0:
            return [], None

        if len(keypoints) > n_features:
            top = np.argsort(-scores)[:n_features]
            keypoints, scores, descriptors = keypoints[top], scores[top], descriptors[top]

        octaves = _score_to_octave(scores, pyramid_n_levels)
        cv_keypoints = [
            cv2.KeyPoint(float(x), float(y), 1.0, -1.0, float(s), int(o), -1)
            for (x, y), s, o in zip(keypoints, scores, octaves)
        ]
        return cv_keypoints, np.ascontiguousarray(descriptors, dtype=np.float32)


def _score_to_octave(scores, pyramid_n_levels):
    """
    Buckets SuperPoint's per-keypoint confidence score into a synthetic
    ORB-style octave, since SuperPoint runs single-scale (no real pyramid
    level to report) - see this module's docstring and issue #38. Ranks
    keypoints by score WITHIN THIS FRAME and buckets by percentile into
    pyramid_n_levels equal-count bins, rather than an absolute score
    cutoff: a fixed absolute scale (e.g. a log curve calibrated so score
    1.0 lands at octave 0) turned out to collapse ~90% of real keypoints
    into a single bucket, since SuperPoint's raw confidence on real TUM
    RGB-D frames rarely exceeds ~0.6-0.7 and is often much lower - found
    during #38's own review, see EVALUATION_RESULTS.md. Percentile ranking
    guarantees a real spread across all pyramid_n_levels buckets
    regardless of the raw score distribution's absolute scale, at the
    cost of octave now depending on this frame's whole keypoint
    population rather than being a pure per-keypoint property (the same
    physical corner's score could rank slightly differently frame to
    frame as the population changes) - accepted as part of the same
    approximation issue #38 itself calls for measuring via ATE/RPE, not
    eliminating outright (SuperPoint exposes no true scale-space value to
    bucket instead). Highest-confidence keypoints land at octave 0 (ORB's
    finest, most-precisely-localized level); least-confident at
    pyramid_n_levels - 1.
    """
    n = len(scores)
    order = np.argsort(scores)  # ascending: order[0] is the lowest score
    rank = np.empty(n, dtype=np.float64)
    rank[order] = np.arange(n)
    percentile = rank / max(n - 1, 1)
    octave = np.round((1.0 - percentile) * (pyramid_n_levels - 1))
    return octave.astype(int)
