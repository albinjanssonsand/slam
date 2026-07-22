"""
Frame source for the SLAM pipeline.

Yields undistorted frames from a video file or a live capture device through
the same interface, so the rest of the pipeline (feature tracking, pose
estimation, etc.) doesn't need to know which one it's reading from. Swapping
a recorded test clip for a live phone feed later is just a different
`source` argument.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import yaml


@dataclass
class Frame:
    index: int
    timestamp: float  # seconds
    image: np.ndarray  # undistorted BGR image, cropped to the valid region


class CalibratedVideoSource:
    """
    Iterates undistorted Frames from a video file or live capture device.

    `source` is passed directly to cv2.VideoCapture: a file path for a
    recorded video, or an integer device index for a live camera/webcam.
    """

    def __init__(self, source, calibration_path):
        self.camera_matrix, self.dist_coeffs, self.calib_size = self._load_calibration(
            calibration_path
        )

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (width, height) != self.calib_size:
            raise ValueError(
                f"Source resolution {(width, height)} does not match calibration "
                f"resolution {self.calib_size}. Recalibrate at this resolution, "
                f"or open the source at the calibrated resolution."
            )

        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (width, height), alpha=0
        )
        self._new_camera_matrix = new_camera_matrix
        self._roi = roi

        x, y, w, h = roi
        # Intrinsics for the image *after* cropping to the valid (non-black) region.
        self.camera_matrix_undistorted = new_camera_matrix.copy()
        self.camera_matrix_undistorted[0, 2] -= x
        self.camera_matrix_undistorted[1, 2] -= y

        self._index = 0

    @staticmethod
    def _load_calibration(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
        image_size = (data["image_width"], data["image_height"])
        return camera_matrix, dist_coeffs, image_size

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            self.release()
            raise StopIteration

        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        undistorted = cv2.undistort(
            frame, self.camera_matrix, self.dist_coeffs, None, self._new_camera_matrix
        )
        x, y, w, h = self._roi
        undistorted = undistorted[y:y + h, x:x + w]

        result = Frame(index=self._index, timestamp=timestamp, image=undistorted)
        self._index += 1
        return result

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


def _demo():
    import argparse

    parser = argparse.ArgumentParser(description="Play back undistorted frames from a source")
    parser.add_argument("--video", required=True, help="Video file path or integer device index")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    args = parser.parse_args()

    source = args.video
    if source.isdigit():
        source = int(source)

    with CalibratedVideoSource(source, args.calibration) as frames:
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