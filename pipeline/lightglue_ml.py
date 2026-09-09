"""
Learned feature matching via a pretrained LightGlue ONNX checkpoint
(fabio-sim/LightGlue-ONNX's standalone SuperPoint+LightGlue matcher export -
github.com/fabio-sim/LightGlue-ONNX/releases/tag/v0.1.3,
superpoint_lightglue.onnx), run per frame-pair on CPU through onnxruntime -
see pipeline/depth_ml.py's DepthEstimator for the ONNX inference pattern
this mirrors.

Only usable against pipeline.superpoint_ml.SuperPointEstimator's float
keypoints/descriptors, not ORB's binary ones (see issue #38) - this
checkpoint was trained against SuperPoint's specific descriptor
distribution and has no defined behavior otherwise.
"""

import cv2
import numpy as np
import onnxruntime as ort


class LightGlueMatcher:
    """Wraps a standalone SuperPoint+LightGlue ONNX matcher checkpoint
    (keypoints+descriptors from both frames in, match indices out)."""

    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def match(self, kp1, desc1, image_shape1, kp2, desc2, image_shape2):
        """
        Matches two SuperPoint keypoint/descriptor sets. image_shape1/2 are
        each an (height, width) pair for the image the corresponding
        keypoints were detected in - LightGlue's positional encoding needs
        keypoints normalized to roughly [-1, 1] by image size (see
        _normalize_keypoints), not raw pixel coordinates.

        Returns a list of cv2.DMatch (queryIdx into kp1/desc1, trainIdx
        into kp2/desc2, distance = 1 - match confidence, so downstream
        code that sorts/thresholds by .distance keeps the same "lower is
        better" convention Hamming/L2 distance already have) - every
        existing consumer (PnP, triangulation, cv2.drawMatches) needs no
        changes.
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        pts1 = np.array([kp.pt for kp in kp1], dtype=np.float32)
        pts2 = np.array([kp.pt for kp in kp2], dtype=np.float32)

        kpts0 = _normalize_keypoints(pts1, *image_shape1)[None]
        kpts1 = _normalize_keypoints(pts2, *image_shape2)[None]

        matches0, _matches1, mscores0, _mscores1 = self.session.run(
            None,
            {
                "kpts0": kpts0,
                "kpts1": kpts1,
                "desc0": desc1[None].astype(np.float32),
                "desc1": desc2[None].astype(np.float32),
            },
        )
        matches0, mscores0 = matches0[0], mscores0[0]

        valid = matches0 > -1
        query_idx = np.where(valid)[0]
        train_idx = matches0[valid]
        confidence = mscores0[valid]

        dmatches = [
            cv2.DMatch(int(q), int(t), float(1.0 - c))
            for q, t, c in zip(query_idx, train_idx, confidence)
        ]
        # match_descriptors' own contract returns matches sorted best-first
        # (lowest distance) - kept here too, since callers (e.g. mapping.py's
        # live matches_ref[:200] visualization) rely on that ordering.
        dmatches.sort(key=lambda m: m.distance)
        return dmatches


def _normalize_keypoints(pts, height, width):
    """Same normalization LightGlue-ONNX's own reference runner uses
    (onnx_runner/lightglue.py's LightGlueRunner.normalize_keypoints):
    center at the image midpoint and scale by the longer side / 2, so
    coordinates land roughly in [-1, 1] regardless of image aspect
    ratio."""
    size = np.array([width, height], dtype=np.float32)
    shift = size / 2
    scale = size.max() / 2
    return (pts - shift) / scale
