"""
ML monocular depth estimation via a pretrained Depth Anything V2 (small)
ONNX checkpoint, run per-frame on CPU through onnxruntime.

This is v2 groundwork only: a per-frame relative (non-metric) depth map -
not yet fused into the sparse map (see NOTES.md's v2 plan for how it'll
eventually anchor bootstrap scale and fill low-texture triangulation gaps).
"""

import cv2
import numpy as np
import onnxruntime as ort

# ImageNet stats + patch-multiple constraint the checkpoint was exported
# with (models/depth_anything_v2_small/preprocessor_config.json).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_MULTIPLE_OF = 14
_TARGET_SIZE = 518


def _resize_output_size(height, width, target=_TARGET_SIZE, multiple_of=_MULTIPLE_OF):
    """
    Same keep-aspect-ratio resize policy as the checkpoint's DPTImageProcessor
    config (keep_aspect_ratio=True): scale by whichever axis lands closer to
    the target size, then round both dimensions to a multiple of the ViT
    patch size (the model requires this, not just a preference).
    """
    scale_h = target / height
    scale_w = target / width
    if abs(1 - scale_w) < abs(1 - scale_h):
        scale_h = scale_w
    else:
        scale_w = scale_h
    new_h = max(multiple_of, round(scale_h * height / multiple_of) * multiple_of)
    new_w = max(multiple_of, round(scale_w * width / multiple_of) * multiple_of)
    return new_h, new_w


class DepthEstimator:
    """Wraps the Depth Anything V2 Small ONNX checkpoint for per-frame
    relative monocular depth estimation on CPU."""

    def __init__(self, model_path="pipeline/models/depth_anything_v2_vits.onnx"):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name

    def estimate(self, image_bgr):
        """Returns a float32 (H, W) relative depth map at the input image's
        original resolution (larger value = closer, per Depth Anything's
        convention) - NOT metric, and not comparable across frames without
        alignment."""
        h, w = image_bgr.shape[:2]
        new_h, new_w = _resize_output_size(h, w)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
        chw = normalized.transpose(2, 0, 1)[None].astype(np.float32)

        depth = self.session.run(None, {self._input_name: chw})[0][0]
        return cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)


def colorize_depth(depth, colormap=cv2.COLORMAP_INFERNO):
    """Normalizes a relative depth map to 0-255 per-frame (min/max, since the
    model's output scale isn't metric or consistent across frames) and
    applies a colormap for visualization."""
    d_min, d_max = float(depth.min()), float(depth.max())
    normalized = (depth - d_min) / (d_max - d_min + 1e-6)
    depth_8u = (normalized * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_8u, colormap)


def _demo():
    import argparse
    import time

    from capture.video_source import CalibratedVideoSource

    parser = argparse.ArgumentParser(
        description="Show live ML monocular depth estimation side by side with the source video"
    )
    parser.add_argument("--video", required=True, help="Video file path or integer device index")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    parser.add_argument("--model", required=True, help="Path to the depth model ONNX checkpoint")
    parser.add_argument("--depth-stride", type=int, default=20,
                         help="Only recompute the depth map every N frames; frames in "
                              "between reuse the most recent depth map. CPU inference is "
                              "far slower than frame decode, so this keeps the demo from "
                              "stalling on every single frame")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable the live source+depth window")
    args = parser.parse_args()

    source = args.video
    if source.isdigit():
        source = int(source)

    estimator = DepthEstimator(args.model)
    last_depth = None
    n_updates = 0

    with CalibratedVideoSource(source, args.calibration) as frames:
        fps = frames.cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay_ms = max(1, int(1000 / fps))

        for frame in frames:
            is_update = last_depth is None or frame.index % args.depth_stride == 0
            if is_update:
                t0 = time.perf_counter()
                last_depth = estimator.estimate(frame.image)
                infer_ms = (time.perf_counter() - t0) * 1000
                status = f"depth updated ({infer_ms:.0f}ms)"
                n_updates += 1
            else:
                status = "depth cached"

            if not args.no_display:
                frame_vis = frame.image.copy()
                cv2.putText(
                    frame_vis, f"frame {frame.index}  {status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if is_update else (0, 200, 0), 2,
                )

                depth_vis = colorize_depth(last_depth)
                combined = np.hstack([frame_vis, depth_vis])

                # Same screen-fit scaling as the mapping demo - portrait phone
                # footage doubled side by side is often wider/taller than a
                # typical screen.
                max_w, max_h = 1600, 900
                display_scale = min(max_w / combined.shape[1], max_h / combined.shape[0], 1.0)
                if display_scale < 1.0:
                    combined = cv2.resize(
                        combined, None, fx=display_scale, fy=display_scale,
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow("Depth Anything V2 Small - RGB vs depth", combined)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

    if not args.no_display:
        cv2.destroyAllWindows()

    print(f"\n{n_updates} depth updates over {frame.index + 1} frames "
          f"(stride={args.depth_stride})")


if __name__ == "__main__":
    _demo()
