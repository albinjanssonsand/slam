"""
Sanity-check a saved camera calibration by undistorting a sample frame.

Saves a side-by-side original-vs-undistorted image so you can visually check
for warping artifacts (a sign of an overfit distortion model, e.g. a large k3)
especially near the edges/corners of the frame.

Usage:
    python -m calibration.verify_undistort --video calibration/data/checkerboard.mp4 \
        --calibration calibration/phone_camera.yaml --frame 100 \
        --output calibration/data/undistort_check.png
"""

import argparse

import cv2
import numpy as np

from capture.video_source import load_calibration


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def draw_grid(image, spacing=50, color=(0, 255, 0)):
    vis = image.copy()
    h, w = vis.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(vis, (x, 0), (x, h), color, 1)
    for y in range(0, h, spacing):
        cv2.line(vis, (0, y), (w, y), color, 1)
    return vis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to a video to sample a frame from")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to sample (default: 0)")
    parser.add_argument("--output", default="calibration/data/undistort_check.png",
                         help="Output comparison image path")
    parser.add_argument("--grid", action="store_true",
                         help="Overlay a straight-line grid before undistorting, to make warping obvious")
    args = parser.parse_args()

    camera_matrix, dist_coeffs, calib_size = load_calibration(args.calibration)
    frame = read_frame(args.video, args.frame)

    h, w = frame.shape[:2]
    if (w, h) != calib_size:
        print(f"WARNING: frame size ({w}x{h}) does not match calibration size "
              f"{calib_size}. Results will be invalid unless these match.")

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0
    )
    undistorted = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    if args.grid:
        # Drawn independently on each panel AFTER undistortion, as a plain
        # reference grid. Drawing it once on the raw frame before undistorting
        # would bake it into the pixels and cv2.undistort would warp the grid
        # lines themselves, which looks like distortion but isn't.
        frame = draw_grid(frame)
        undistorted = draw_grid(undistorted)

    cv2.putText(frame, "original", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(undistorted, "undistorted", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    comparison = np.hstack([frame, undistorted])
    cv2.imwrite(args.output, comparison)
    print(f"Saved comparison image to {args.output}")
    print("Check the undistorted half for warping near the edges/corners "
          "(bowing, stretching) which would indicate an overfit distortion model.")


if __name__ == "__main__":
    main()