"""
Frame sources for the SLAM pipeline.

Yield undistorted frames from a video file, a live capture device, or a
TUM-RGB-D-style image-sequence folder through the same interface, so the
rest of the pipeline (feature tracking, pose estimation, etc.) doesn't need
to know which one it's reading from. Swapping a recorded test clip, a live
phone feed, or a benchmark dataset for another later is just a different
`source` argument.
"""

import os
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np
import yaml


@dataclass
class Frame:
    index: int
    timestamp: float  # seconds
    image: np.ndarray  # undistorted BGR image, cropped to the valid region


class UndistortParams(NamedTuple):
    """
    Undistort/crop parameters for one calibrated resolution: the new camera
    matrix and valid-region ROI cv2.getOptimalNewCameraMatrix produces for
    it, plus the resulting intrinsics for the image *after* cropping to that
    ROI. Bundled together since every consumer needs all three in lockstep.
    """
    new_camera_matrix: np.ndarray
    roi: tuple
    camera_matrix_undistorted: np.ndarray


def load_calibration(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    image_size = (data["image_width"], data["image_height"])
    return camera_matrix, dist_coeffs, image_size


def _check_resolution(width, height, calib_size):
    if (width, height) != calib_size:
        raise ValueError(
            f"Source resolution {(width, height)} does not match calibration "
            f"resolution {calib_size}. Recalibrate at this resolution, "
            f"or open the source at the calibrated resolution."
        )


def _build_undistort_params(camera_matrix, dist_coeffs, width, height):
    """
    Computes the new camera matrix/ROI for undistorting+cropping frames of
    size (width, height), plus the resulting intrinsics for the image
    *after* cropping to the valid (non-black) region - shared by every frame
    source so undistort/crop behavior stays identical across them.
    """
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), alpha=0
    )
    x, y, w, h = roi
    camera_matrix_undistorted = new_camera_matrix.copy()
    camera_matrix_undistorted[0, 2] -= x
    camera_matrix_undistorted[1, 2] -= y
    return UndistortParams(new_camera_matrix, roi, camera_matrix_undistorted)


def _undistort_and_crop(frame, camera_matrix, dist_coeffs, undistort_params):
    undistorted = cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, undistort_params.new_camera_matrix
    )
    x, y, w, h = undistort_params.roi
    return undistorted[y:y + h, x:x + w]


class _CalibratedSource:
    """
    Shared scaffolding for the calibrated frame sources below: calibration
    loading, resolution validation, undistort-param setup, and the
    iterator/context-manager protocol every source needs identically.

    A subclass opens its own underlying source, determines its
    (width, height), then calls super().__init__(calibration_path, width,
    height) to run this shared setup - after which it only needs to
    implement _read_raw_frame() (-> (frame, timestamp), or None at end of
    stream) and, if it holds a resource that needs closing, override
    release().
    """

    def __init__(self, calibration_path, width, height):
        self.camera_matrix, self.dist_coeffs, self.calib_size = load_calibration(
            calibration_path
        )
        _check_resolution(width, height, self.calib_size)

        self.undistort_params = _build_undistort_params(
            self.camera_matrix, self.dist_coeffs, width, height
        )
        self.camera_matrix_undistorted = self.undistort_params.camera_matrix_undistorted

        self._index = 0

    def _read_raw_frame(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        result = self._read_raw_frame()
        if result is None:
            self.release()
            raise StopIteration

        raw_frame, timestamp = result
        undistorted = _undistort_and_crop(
            raw_frame, self.camera_matrix, self.dist_coeffs, self.undistort_params
        )

        frame = Frame(index=self._index, timestamp=timestamp, image=undistorted)
        self._index += 1
        return frame

    def release(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class CalibratedVideoSource(_CalibratedSource):
    """
    Iterates undistorted Frames from a video file or live capture device.

    `source` is passed directly to cv2.VideoCapture: a file path for a
    recorded video, or an integer device index for a live camera/webcam.
    """

    def __init__(self, source, calibration_path):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        super().__init__(calibration_path, width, height)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def _read_raw_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        return frame, timestamp

    def release(self):
        self.cap.release()


class CalibratedImageSequenceSource(_CalibratedSource):
    """
    Iterates undistorted Frames from a TUM-RGB-D-style image-sequence folder:
    an `rgb.txt` file listing "timestamp filename" (one line per frame,
    `#`-prefixed comment lines ignored) alongside the referenced PNGs.

    Preserves each frame's real timestamp as given in `rgb.txt` rather than
    approximating it (e.g. by muxing to video and reading back an assumed
    frame rate) - ground-truth trajectory evaluation (RPE in particular) is
    timestamp-sensitive. Shares undistort/crop behavior with
    CalibratedVideoSource so the two sources are interchangeable everywhere
    the pipeline consumes a Frame iterator.
    """

    def __init__(self, folder_path, calibration_path):
        rgb_list_path = os.path.join(folder_path, "rgb.txt")
        self._entries = self._load_rgb_list(rgb_list_path, folder_path)
        if not self._entries:
            raise RuntimeError(f"No frames listed in {rgb_list_path}")

        # Decoded once here to read the resolution for the calibration check
        # below; cached and handed to _read_raw_frame() on the first call
        # instead of decoding this same PNG from disk a second time.
        first_path = self._entries[0][1]
        self._first_image = cv2.imread(first_path)
        if self._first_image is None:
            raise RuntimeError(f"Could not read image: {first_path}")
        height, width = self._first_image.shape[:2]

        super().__init__(calibration_path, width, height)

        timestamps = [t for t, _ in self._entries]
        deltas = np.diff(timestamps)
        self.fps = float(1.0 / np.median(deltas)) if len(deltas) > 0 else 30.0

    @staticmethod
    def _load_rgb_list(rgb_list_path, folder_path):
        entries = []
        with open(rgb_list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                timestamp_str, filename = line.split(None, 1)
                entries.append((float(timestamp_str), os.path.join(folder_path, filename)))
        return entries

    def _read_raw_frame(self):
        if self._index >= len(self._entries):
            return None

        timestamp, image_path = self._entries[self._index]
        if self._first_image is not None:
            frame, self._first_image = self._first_image, None
        else:
            frame = cv2.imread(image_path)
            if frame is None:
                raise RuntimeError(f"Could not read image: {image_path}")
        return frame, timestamp


def open_calibrated_source(source, calibration_path):
    """
    Dispatches to CalibratedImageSequenceSource for a folder path, or
    CalibratedVideoSource for a video file / integer device index string -
    the single place the pipeline needs to check to accept either a
    recorded/live video or an image-sequence dataset like TUM RGB-D.
    """
    if os.path.isdir(source):
        return CalibratedImageSequenceSource(source, calibration_path)
    if source.isdigit():
        source = int(source)
    return CalibratedVideoSource(source, calibration_path)


def _demo():
    import argparse

    parser = argparse.ArgumentParser(description="Play back undistorted frames from a source")
    parser.add_argument("--video", required=True,
                         help="Video file path, integer device index, or image-sequence folder "
                              "(containing rgb.txt)")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    args = parser.parse_args()

    with open_calibrated_source(args.video, args.calibration) as frames:
        print("Effective camera matrix (post-undistort, post-crop):")
        print(frames.camera_matrix_undistorted)

        for frame in frames:
            cv2.putText(
                frame.image, f"frame {frame.index}  t={frame.timestamp:.2f}s",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.imshow("undistorted", frame.image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()
