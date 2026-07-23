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
from scipy.optimize import least_squares

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


def render_colorbar(d_min, d_max, height, width=70, colormap=cv2.COLORMAP_INFERNO, n_ticks=5):
    """
    Vertical legend for colorize_depth's colormap. Oriented to match
    render_scanline_profiles' near-at-bottom/far-at-top convention: bright
    (near, larger raw value) at the bottom, dark (far, smaller raw value)
    at the top.
    """
    bar_width = 30
    gradient = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(np.repeat(gradient, bar_width, axis=1), colormap)

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    canvas[:, :bar_width] = bar

    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        row = int(frac * (height - 1))
        value = d_min + frac * (d_max - d_min)
        cv2.line(canvas, (0, row), (bar_width, row), (0, 0, 0), 1)
        cv2.putText(canvas, f"{value:.1f}", (bar_width + 2, min(row + 4, height - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


# Distinct BGR colors, one per scanline - shared between the line drawn on
# the depth panel and its matching curve in the profile plot.
_SCAN_COLORS = [(0, 0, 220), (0, 170, 0), (220, 120, 0), (0, 180, 220)]


def to_pseudo_depth(raw_depth, eps=0.05):
    """
    Depth Anything's raw output is inverse-depth-like (larger = closer);
    real pinhole Z is approximately its reciprocal (disparity ~ a/Z + b),
    not just a re-ordering of it - inverting reproduces the compression far
    objects have (small disparity changes = large real distance changes)
    and the expansion near objects have, which a simple negate/shift does
    not. Still not metric (that needs the scale/shift calibration against
    geometric ORB points, planned separately) - just a better-shaped
    relative ordering.

    Can't invert the raw values directly: they aren't guaranteed positive
    (small/negative values show up for far background in practice), so
    they're normalized to (0, 1] first (near=1, far=0), then inverted with
    a small floor `eps` so the farthest point doesn't blow up toward
    infinity - eps effectively caps the represented near:far distance ratio
    at 1/eps.
    """
    d_min, d_max = float(raw_depth.min()), float(raw_depth.max())
    normalized = (raw_depth - d_min) / (d_max - d_min + 1e-6)
    return 1.0 / (normalized + eps)


def scanline_rows(height, n_rows, top_frac=0.35, bottom_frac=0.85):
    """
    Evenly spaced row indices between top_frac and bottom_frac of the image
    height - stays clear of the sky/ceiling and the extreme bottom edge,
    where depth estimates tend to be least reliable.
    """
    return [int(f * (height - 1)) for f in np.linspace(top_frac, bottom_frac, n_rows)]


def backproject_row(pseudo_depth, row, camera_matrix, stride=4):
    """
    Back-projects one horizontal row of a *pseudo-depth* map (see
    to_pseudo_depth - larger = farther, like a real pinhole Z) into
    camera-frame (X, Z) via X = (u - cx)/fx * Z. Sweeping u this way traces
    real scene geometry - a flat wall stays flat, a near object steps the
    curve toward the camera - because X and Z both vary independently with
    u. A *vertical* sweep would not work for this: u is fixed, so X/Z is a
    constant regardless of depth, and every point collapses onto a single
    ray through the camera origin no matter what the scene looks like.
    """
    fx, cx = camera_matrix[0, 0], camera_matrix[0, 2]
    us = np.arange(0, pseudo_depth.shape[1], stride)
    z = pseudo_depth[row, us].astype(np.float64)
    x = (us - cx) / fx * z
    return x, z


def backproject_pixels(us, vs, z_cam, camera_matrix):
    """
    Full pinhole back-projection of pixel coordinates (us, vs) with known
    camera-frame depth z_cam into camera-frame 3D points (X, Y, Z) - unlike
    backproject_row, which only returns (X, Z) for the top-down profile
    plot, this keeps Y so the result is usable as an actual 3D map point.
    """
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    x = (us - cx) / fx * z_cam
    y = (vs - cy) / fy * z_cam
    return np.stack([x, y, z_cam], axis=1)


def fit_disparity_scale_shift(disparity, inv_depth):
    """
    Robust least-squares fit of disparity ~= a * inv_depth + b - the
    standard MiDaS/Depth-Anything relative-depth calibration - against a set
    of points with known camera-frame depth (e.g. the map's own confirmed,
    already-scaled points). This has to be refit fresh each time it's used,
    not treated as a fixed constant: the map's own scale drifts slowly over
    a sequence (see NOTES.md), so the "correct" (a, b) drifts along with it.

    An ordinary least-squares solve seeds the fit, then a Huber-loss
    refinement (same rationale as bundle_adjustment.py's use of it) keeps a
    handful of noisy or mismatched points from dominating the result.

    Returns (a, b), or None if there isn't enough data for a stable fit.
    """
    disparity = np.asarray(disparity, dtype=np.float64)
    inv_depth = np.asarray(inv_depth, dtype=np.float64)
    if len(disparity) < 10:
        return None

    A = np.stack([inv_depth, np.ones_like(inv_depth)], axis=1)
    x0, *_ = np.linalg.lstsq(A, disparity, rcond=None)

    residual0 = A @ x0 - disparity
    mad = np.median(np.abs(residual0 - np.median(residual0)))
    f_scale = max(1.4826 * mad, 1e-3)

    result = least_squares(
        lambda p: p[0] * inv_depth + p[1] - disparity,
        x0, loss="huber", f_scale=f_scale,
    )
    a, b = result.x
    if abs(a) < 1e-8:
        return None
    return float(a), float(b)


def detect_object_mask(depth, edge_percentile=90, max_background_frac=0.15):
    """
    Very simple region segmentation to tell discrete objects apart from
    smoothly-varying background (floor/walls) - motivated by scanline
    densification otherwise producing a repeated "streak" every keyframe
    wherever a scanline just crosses open floor, which carries no real
    object information and is pure clutter.

    Method: threshold the depth gradient magnitude to get edges, then treat
    connected regions of non-edge pixels as candidate flat surfaces. A
    continuous floor or wall dominates the frame by sheer pixel area in a
    way a discrete object doesn't, so the largest region(s) are assumed to
    be background and everything else is assumed to be "object."

    edge_percentile is relative (not an absolute gradient magnitude), since
    the model's raw disparity has no fixed scale - the steepest
    edge_percentile% of gradient magnitudes in this frame are treated as
    edges. max_background_frac is the minimum fraction of the frame a flat
    region must cover to count as background rather than an object.

    Returns a boolean mask, True where a pixel belongs to a small ("object")
    region rather than the dominant background or an edge itself.
    """
    depth = depth.astype(np.float32)
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    edge_threshold = np.percentile(grad_mag, edge_percentile)
    non_edge = (grad_mag <= edge_threshold).astype(np.uint8)

    n_labels, labels = cv2.connectedComponents(non_edge, connectivity=4)
    sizes = np.bincount(labels.ravel(), minlength=n_labels)
    total = labels.size
    # Label 0 is the edge pixels themselves (cv2 treats 0-valued input as
    # background) - never "object", same as an oversized flat region.
    background_labels = set(np.where(sizes > max_background_frac * total)[0]) | {0}

    return ~np.isin(labels, list(background_labels))


def render_scanline_profiles(rows_xz, size=600, margin=40, n_ticks=5):
    """
    Renders each scanline's (X, Z) profile as a distinctly colored curve in
    a shared, auto-scaled plot - same top-down (X, Z) convention as
    pose.render_trajectory (near/small-Z at the bottom, far/large-Z at the
    top), so these curves are directly comparable to the geometric map's
    own top-down point cloud. A horizontal gridline + Z value is drawn at
    each tick so the curves can be read as an actual depth axis, not just a
    schematic shape.
    """
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)

    all_xz = np.hstack([np.vstack([x, z]) for x, z in rows_xz if len(x)])
    if all_xz.size == 0:
        return canvas
    min_xy = all_xz.min(axis=1)
    max_xy = all_xz.max(axis=1)
    span = np.maximum(max_xy - min_xy, 1e-3)
    scale = (size - 2 * margin) / span.max()

    def to_canvas(x, z):
        cx = int((x - min_xy[0]) * scale + margin)
        cy = int((z - min_xy[1]) * scale + margin)
        return cx, size - cy  # flip so near (small Z) is at the bottom

    for z_val in np.linspace(min_xy[1], max_xy[1], n_ticks):
        _, py = to_canvas(min_xy[0], z_val)
        cv2.line(canvas, (margin, py), (size - margin, py), (220, 220, 220), 1)
        # Ticks are in to_pseudo_depth's output unit (1/(normalized disparity
        # + eps)), not metric distance - labeled as such, not as "Z", so
        # this isn't mistaken for a calibrated depth axis.
        cv2.putText(canvas, f"{z_val:.2f}", (4, min(max(py, 12), size - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    for i, (x, z) in enumerate(rows_xz):
        color = _SCAN_COLORS[i % len(_SCAN_COLORS)]
        pts = np.array([to_canvas(xi, zi) for xi, zi in zip(x, z)], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

    cv2.putText(canvas, "scanline depth profiles (X, 1/(z+eps))", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return canvas


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
    parser.add_argument("--scan-rows", type=int, default=3,
                         help="Number of horizontal scanlines to back-project into (X, Z) "
                              "depth profiles")
    parser.add_argument("--scan-stride", type=int, default=4,
                         help="Column stride when sampling each scanline")
    parser.add_argument("--no-display", action="store_true",
                         help="Disable the live source+depth window")
    args = parser.parse_args()

    source = args.video
    if source.isdigit():
        source = int(source)

    estimator = DepthEstimator(args.model)
    last_depth = None
    rows_xz = []
    n_updates = 0

    with CalibratedVideoSource(source, args.calibration) as frames:
        K = frames.camera_matrix_undistorted
        fps = frames.cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay_ms = max(1, int(1000 / fps))
        rows = None

        for frame in frames:
            if rows is None:
                rows = scanline_rows(frame.image.shape[0], args.scan_rows)

            is_update = last_depth is None or frame.index % args.depth_stride == 0
            if is_update:
                t0 = time.perf_counter()
                last_depth = estimator.estimate(frame.image)
                infer_ms = (time.perf_counter() - t0) * 1000
                status = f"depth updated ({infer_ms:.0f}ms)"
                n_updates += 1
                pseudo_depth = to_pseudo_depth(last_depth)
                rows_xz = [
                    backproject_row(pseudo_depth, row, K, stride=args.scan_stride)
                    for row in rows
                ]
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
                for i, row in enumerate(rows):
                    color = _SCAN_COLORS[i % len(_SCAN_COLORS)]
                    cv2.line(frame_vis, (0, row), (frame_vis.shape[1], row), color, 2)
                    cv2.line(depth_vis, (0, row), (depth_vis.shape[1], row), color, 2)

                colorbar_vis = render_colorbar(
                    float(last_depth.min()), float(last_depth.max()), frame_vis.shape[0]
                )
                profile_vis = render_scanline_profiles(rows_xz, size=frame_vis.shape[0])
                combined = np.hstack([frame_vis, depth_vis, colorbar_vis, profile_vis])

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
