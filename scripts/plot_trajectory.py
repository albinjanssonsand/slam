"""
Plots an estimated trajectory against a TUM ground-truth trajectory and saves
a PNG. Timestamps are associated by nearest match (like evo/TUM's own
association), then the estimate is Umeyama-aligned (rotation + scale +
translation) onto the ground-truth frame before plotting, since the
pipeline's monocular estimate has no absolute scale or frame - the same
alignment `evo_ape ... -a -s` performs for scoring (see NOTES.md).

Usage:
    python scripts/plot_trajectory.py \
        --estimate results/estimate.txt \
        --groundtruth datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
        --output results/trajectory.png
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)


def load_tum(path):
    """Loads a TUM-format file (timestamp tx ty tz qx qy qz qw), skipping
    blank/comment lines. Returns an (N, 8) array."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append([float(v) for v in line.split()])
    return np.array(rows)


def associate(est, gt, max_diff=0.02):
    """Matches each estimate row to its nearest ground-truth row in time,
    dropping pairs further apart than max_diff seconds. Returns the matched
    (N, 3) position arrays."""
    gt_times = gt[:, 0]
    est_positions, gt_positions = [], []
    for row in est:
        idx = np.searchsorted(gt_times, row[0])
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(gt_times)]
        best = min(candidates, key=lambda i: abs(gt_times[i] - row[0]))
        if abs(gt_times[best] - row[0]) <= max_diff:
            est_positions.append(row[1:4])
            gt_positions.append(gt[best, 1:4])
    return np.array(est_positions), np.array(gt_positions)


def umeyama_alignment(src, dst):
    """Least-squares similarity transform (rotation, scale, translation)
    mapping src onto dst: dst ~= s * R @ src + t."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    src_var = (src_c ** 2).sum() / len(src)
    scale = np.sum(D * np.diag(S)) / src_var
    t = dst_mean - scale * R @ src_mean
    return R, scale, t


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", default="results/estimate.txt")
    parser.add_argument(
        "--groundtruth",
        default="datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt",
    )
    parser.add_argument("--output", default="results/trajectory.png")
    parser.add_argument("--max-diff", type=float, default=0.02,
                         help="max timestamp gap (s) for association")
    args = parser.parse_args()

    est = load_tum(args.estimate)
    gt = load_tum(args.groundtruth)

    est_matched, gt_matched = associate(est, gt, args.max_diff)
    if len(est_matched) < 3:
        raise SystemExit(
            f"Only {len(est_matched)} timestamp-matched pairs found - "
            "not enough to align. Check --max-diff or the input files."
        )

    R, scale, t = umeyama_alignment(est_matched, gt_matched)
    est_aligned = (scale * (R @ est[:, 1:4].T).T) + t

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(gt[:, 1], gt[:, 2], gt[:, 3], label="ground truth", linewidth=1.5)
    ax.plot(est_aligned[:, 0], est_aligned[:, 1], est_aligned[:, 2],
             label="estimate (aligned)", linewidth=1.5)
    ax.scatter(*est_aligned[0], c="green", s=60, label="start")
    ax.scatter(*est_aligned[-1], c="red", s=60, label="end")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Estimated vs. ground-truth trajectory")
    ax.legend()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved trajectory plot to {args.output}")


if __name__ == "__main__":
    main()
