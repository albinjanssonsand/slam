"""
Appearance-based keyframe embedding via a pretrained CLIP (ViT-B/32) vision
encoder ONNX checkpoint, run per-keyframe on CPU through onnxruntime.

Unlike depth_ml.py's per-frame dense output, this produces a single 512-d
whole-image embedding per keyframe - used by --relocalize (#13) as an
appearance-based pre-filter (#50): rank candidate keyframes by cosine
similarity of their CLIP embeddings before attempting brute-force ORB point
matching against them, so a revisited scene can be recognized despite
viewpoint/lighting changes that break literal point correspondence (what
DBoW2 does for the reference paper - see EVALUATION_RESULTS.md's "#49"
section for why plain point matching alone doesn't recover freiburg1_desk's
whip-pan gap).
"""

import cv2
import numpy as np
import onnxruntime as ort

# CLIP's own preprocessing stats + target size (Xenova/clip-vit-base-patch32's
# preprocessor_config.json: image_mean, image_std, crop_size 224, resample 3
# = PIL BICUBIC).
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_CROP_SIZE = 224


def _resize_shortest_edge_and_center_crop(rgb, target=_CROP_SIZE):
    """Same policy as CLIPFeatureExtractor: resize so the shortest edge is
    `target`, keeping aspect ratio, then center-crop to target x target."""
    h, w = rgb.shape[:2]
    scale = target / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    top = (new_h - target) // 2
    left = (new_w - target) // 2
    return resized[top:top + target, left:left + target]


class CLIPEmbedder:
    """Wraps CLIP's vision-tower ONNX checkpoint (Xenova/clip-vit-base-patch32's
    vision-only export) for per-keyframe whole-image appearance embedding on
    CPU."""

    def __init__(self, model_path="pipeline/models/clip_vit_b32_vision.onnx"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name

    def embed(self, image_bgr):
        """Returns a 512-d, L2-normalized float32 embedding, so cosine
        similarity between two embeddings is just their dot product."""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cropped = _resize_shortest_edge_and_center_crop(rgb)
        normalized = (cropped.astype(np.float32) / 255.0 - _MEAN) / _STD
        chw = normalized.transpose(2, 0, 1)[None].astype(np.float32)

        embedding = self.session.run(None, {self._input_name: chw})[0][0]
        return embedding / (np.linalg.norm(embedding) + 1e-8)
