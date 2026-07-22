"""
Camera calibration from a checkerboard video.

Usage:
    python calibration/calibrate.py --video calibration/data/checkerboard.mp4 --pattern-size 7x7 --square-size 26 --output calibration/phone_camera.yaml

--pattern-size is the number of INTERNAL corners (cols x rows), not squares.
--square-size is the physical size of one checkerboard square, in mm.
"""

import argparse
import sys

import cv2
import numpy as np
import yaml


def parse_pattern_size(s):
    cols, rows = s.lower().split("x")
    return int(cols), int(rows)


def extract_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def find_corners_in_video(video_path, pattern_size, every_n_frames, show):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_points = []
    image_size = None
    frame_idx = 0
    used_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % every_n_frames != 0:
            frame_idx += 1
            continue
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            image_points.append(corners)
            used_count += 1

            if show:
                vis = frame.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                cv2.putText(
                    vis, f"used frames: {used_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                cv2.imshow("calibration", vis)
                cv2.waitKey(1)

    cap.release()
    if show:
        cv2.destroyAllWindows()

    return image_points, image_size, used_count


def calibrate(object_points_list, image_points, image_size):
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_list, image_points, image_size, None, None
    )
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def compute_reprojection_error(object_points_list, image_points, rvecs, tvecs,
                                camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0
    for i in range(len(object_points_list)):
        projected, _ = cv2.projectPoints(
            object_points_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        observed = np.asarray(image_points[i], dtype=np.float64).reshape(-1, 2)
        projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        error = np.linalg.norm(observed - projected, axis=1).mean()
        total_error += error
        total_points += 1
    return total_error / total_points


def save_calibration(path, camera_matrix, dist_coeffs, image_size, reproj_error):
    data = {
        "image_width": image_size[0],
        "image_height": image_size[1],
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "reprojection_error": float(reproj_error),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to checkerboard calibration video")
    parser.add_argument("--pattern-size", required=True,
                         help="Internal corners as COLSxROWS, e.g. 9x6")
    parser.add_argument("--square-size", type=float, required=True,
                         help="Checkerboard square size in mm")
    parser.add_argument("--every-n-frames", type=int, default=5,
                         help="Sample every Nth frame from the video (default: 5)")
    parser.add_argument("--output", default="calibration/phone_camera.yaml",
                         help="Output YAML path")
    parser.add_argument("--show", action="store_true",
                         help="Show detected corners while processing")
    args = parser.parse_args()

    pattern_size = parse_pattern_size(args.pattern_size)
    objp = extract_object_points(pattern_size, args.square_size)

    print(f"Scanning {args.video} for {pattern_size[0]}x{pattern_size[1]} checkerboard...")
    image_points, image_size, used_count = find_corners_in_video(
        args.video, pattern_size, args.every_n_frames, args.show
    )

    if used_count < 10:
        print(
            f"WARNING: only {used_count} usable frames found. "
            "Aim for 20-40+ good detections covering different positions/angles/distances "
            "for a reliable calibration.",
            file=sys.stderr,
        )
    if used_count == 0:
        print("ERROR: no checkerboard detected in any frame. Check --pattern-size "
              "and that the board is well-lit and fully visible.", file=sys.stderr)
        sys.exit(1)

    object_points_list = [objp] * used_count

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate(
        object_points_list, image_points, image_size
    )
    reproj_error = compute_reprojection_error(
        object_points_list, image_points, rvecs, tvecs, camera_matrix, dist_coeffs
    )

    print(f"Used {used_count} frames")
    print(f"RMS calibration error (OpenCV): {ret:.4f}")
    print(f"Mean reprojection error: {reproj_error:.4f} px")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.flatten())

    if reproj_error > 1.0:
        print(
            "WARNING: reprojection error is high (>1.0 px). Consider recapturing "
            "with more coverage/sharper focus, or check --square-size is correct.",
            file=sys.stderr,
        )

    save_calibration(args.output, camera_matrix, dist_coeffs, image_size, reproj_error)
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()