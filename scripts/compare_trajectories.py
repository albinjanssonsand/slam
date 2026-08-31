"""
Compares two estimated trajectories (e.g. geometric-only vs. + ML depth)
against one TUM ground-truth trajectory. Each estimate is independently
Umeyama-aligned (rotation + scale + translation) onto ground truth - the
same Sim(3) alignment `evo_ape ... -a -s` performs (see NOTES.md's
"Ground-truth evaluation" section) - reusing `plot_trajectory.py`'s
`load_tum`/`associate`/`umeyama_alignment` helpers. Both aligned estimates
are plotted overlaid against ground truth on one figure, and ATE
(translation RMSE over the timestamp-matched, aligned pairs) is printed for
each.

ATE is computed directly in Python here rather than shelling out to
`evo_ape`, since `evo` (`pip install evo`) isn't a hard dependency of this
repo - see NOTES.md for a full `evo_ape`/`evo_rpe` metrics report once `evo`
is installed.

Usage:
    python scripts/compare_trajectories.py \
        --estimate-a results/geometric_estimate.txt \
        --estimate-b results/ml_fusion_estimate.txt \
        --groundtruth datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
        --output results/compare_trajectories.png
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from plot_trajectory import load_tum, associate, umeyama_alignment


def align_and_score(est, gt, max_diff):
    """Umeyama-aligns est onto gt (matched by timestamp) and returns the
    full aligned estimate plus its ATE - the RMSE of the aligned position
    residuals at the matched pairs, computed the same way `evo_ape -a -s`
    reports `rmse`."""
    est_matched, gt_matched = associate(est, gt, max_diff)
    if len(est_matched) < 3:
        raise SystemExit(
            f"Only {len(est_matched)} timestamp-matched pairs found - "
            "not enough to align. Check --max-diff or the input files."
        )

    R, scale, t = umeyama_alignment(est_matched, gt_matched)
    est_aligned = (scale * (R @ est[:, 1:4].T).T) + t
    aligned_matched = (scale * (R @ est_matched.T).T) + t
    ate_rmse = np.sqrt(np.mean(np.sum((aligned_matched - gt_matched) ** 2, axis=1)))
    return est_aligned, ate_rmse


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--estimate-a", required=True)
    parser.add_argument("--estimate-b", required=True)
    parser.add_argument("--label-a", default="geometric-only")
    parser.add_argument("--label-b", default="+ ML depth")
    parser.add_argument(
        "--groundtruth",
        default="datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt",
    )
    parser.add_argument("--output", default="results/compare_trajectories.png")
    parser.add_argument("--max-diff", type=float, default=0.02,
                         help="max timestamp gap (s) for association")
    args = parser.parse_args()

    gt = load_tum(args.groundtruth)
    est_a = load_tum(args.estimate_a)
    est_b = load_tum(args.estimate_b)

    aligned_a, ate_a = align_and_score(est_a, gt, args.max_diff)
    aligned_b, ate_b = align_and_score(est_b, gt, args.max_diff)

    print("ATE (translation RMSE), Sim(3)-aligned:")
    print(f"  {args.label_a}: {ate_a:.6f}")
    print(f"  {args.label_b}: {ate_b:.6f}")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    ax.plot(gt[:, 1], gt[:, 2], gt[:, 3], label="ground truth", linewidth=1.5)
    ax.plot(aligned_a[:, 0], aligned_a[:, 1], aligned_a[:, 2],
             label=f"{args.label_a} (aligned)", linewidth=1.5)
    ax.plot(aligned_b[:, 0], aligned_b[:, 1], aligned_b[:, 2],
             label=f"{args.label_b} (aligned)", linewidth=1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Trajectory comparison vs. ground truth")
    ax.legend()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved comparison plot to {args.output}")


if __name__ == "__main__":
    main()
