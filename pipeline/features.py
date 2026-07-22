"""
ORB feature detection and matching between frames.
"""

import cv2


def create_orb(n_features=2000):
    return cv2.ORB_create(nfeatures=n_features)


def detect_and_compute(orb, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors


def detect_and_compute_gridded(image, n_features=2000, grid=(4, 4), pad=16):
    """
    ORB detection with a per-cell feature quota, instead of one global budget.

    A single cv2.ORB_create(nfeatures=N) call returns whatever N strongest
    corners exist in the WHOLE image - if one region (e.g. a richly textured
    foreground object) has much stronger corners than another (e.g. a distant
    background), that region can consume nearly the entire budget, leaving
    the other region with few or no features. That starves the map of points
    in the under-represented region, which becomes a real problem once the
    over-represented region leaves the frame later in a sequence. Splitting
    the image into a grid and enforcing a quota per cell guarantees spatial
    coverage regardless of texture disparity.

    Each cell is cropped with a `pad`-pixel margin so keypoints near the
    nominal cell boundary still have full context available for their
    descriptor - a hard, unpadded crop truncates that context, producing
    corrupted descriptors (and the false matches that come with them) right
    at every internal grid line. Detections that fall inside the padding
    margin are discarded (not just clipped) so neighboring cells don't each
    report the same feature twice.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    rows, cols = grid
    h, w = gray.shape[:2]
    per_cell = max(1, n_features // (rows * cols))
    cell_orb = cv2.ORB_create(nfeatures=per_cell)

    all_keypoints = []
    all_descriptors = []

    for r in range(rows):
        y0, y1 = (h * r) // rows, (h * (r + 1)) // rows
        for c in range(cols):
            x0, x1 = (w * c) // cols, (w * (c + 1)) // cols

            py0, py1 = max(0, y0 - pad), min(h, y1 + pad)
            px0, px1 = max(0, x0 - pad), min(w, x1 + pad)
            cell = gray[py0:py1, px0:px1]

            kp, desc = cell_orb.detectAndCompute(cell, None)
            if not kp:
                continue

            kept_kp = []
            kept_idx = []
            for i, k in enumerate(kp):
                gx, gy = k.pt[0] + px0, k.pt[1] + py0
                if x0 <= gx < x1 and y0 <= gy < y1:
                    k.pt = (gx, gy)
                    kept_kp.append(k)
                    kept_idx.append(i)

            if not kept_kp:
                continue
            all_keypoints.extend(kept_kp)
            all_descriptors.append(desc[kept_idx])

    if not all_keypoints:
        return [], None

    import numpy as np
    return all_keypoints, np.vstack(all_descriptors)


def match_descriptors(desc1, desc2, ratio=0.75):
    """
    Brute-force Hamming matching with Lowe's ratio test.
    Returns a list of cv2.DMatch, sorted by distance (best first).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    good.sort(key=lambda m: m.distance)
    return good


def _demo():
    import argparse

    from capture.video_source import CalibratedVideoSource

    parser = argparse.ArgumentParser(description="Visualize ORB matches between consecutive frames")
    parser.add_argument("--video", required=True, help="Video file path or integer device index")
    parser.add_argument("--calibration", required=True, help="Path to calibration YAML")
    parser.add_argument("--n-features", type=int, default=2000)
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold")
    args = parser.parse_args()

    source = args.video
    if source.isdigit():
        source = int(source)

    orb = create_orb(args.n_features)

    prev_image = None
    prev_kp = None
    prev_desc = None

    with CalibratedVideoSource(source, args.calibration) as frames:
        fps = frames.cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay_ms = max(1, int(1000 / fps))

        for frame in frames:
            kp, desc = detect_and_compute(orb, frame.image)

            if prev_desc is not None:
                matches = match_descriptors(prev_desc, desc, args.ratio)
                vis = cv2.drawMatches(
                    prev_image, prev_kp, frame.image, kp, matches[:200], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                cv2.putText(
                    vis, f"frame {frame.index}  matches={len(matches)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                cv2.imshow("orb matches", vis)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break

            prev_image, prev_kp, prev_desc = frame.image, kp, desc

    cv2.destroyAllWindows()


if __name__ == "__main__":
    _demo()