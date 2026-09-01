# Evaluation Results: Pipeline vs. Ground Truth (TUM RGB-D)

Tracks ATE/RPE against the TUM RGB-D `freiburg1` sequences, one section per
pipeline version (append, don't overwrite, so a regression/improvement is a
diff between sections). Currently holds the geometric-only baseline
(pre-ML); a section per `V2_INTEGRATION_PLAN.md` ML-depth-fusion phase gets
added once that work lands.

**Method:** `pipeline.mapping --trajectory-output` writes each accepted
keyframe's pose in TUM format; `evo_ape`/`evo_rpe ... -a -s` (Sim(3)
alignment - required since this pipeline's monocular scale isn't metric)
score it against each sequence's `groundtruth.txt`. `scripts/compare_trajectories.py`
isn't used here since it compares two estimates against one ground truth at
once and there's only one pipeline version per section so far - it becomes
the right tool once a later phase needs plotting against this baseline.

**"Too hard" criterion:** a dataset is too hard if the pipeline never
produces a usable trajectory (crash, or too few keyframes for `evo`'s
Umeyama fit), **or** if it does produce ATE/RPE but at low **trajectory
coverage** (estimate duration / ground-truth duration, from each file's
first/last timestamp). The second case matters because this pipeline has no
relocalization: a run that loses tracking early just stops producing
keyframes, and `evo`'s Sim(3) alignment then fits well to a small,
temporally-clustered handful of early poses - a deceptively low ATE for a
trajectory that covers almost none of the real motion. Coverage catches
this; ATE/RPE alone don't.

**Reproduction** (for any sequence `<seq>` in `xyz`, `desk`, `room`, `rpy`):

```bash
python -m pipeline.mapping \
  --video datasets/tum/rgbd_dataset_freiburg1_<seq> \
  --calibration calibration/tum_freiburg1.yaml \
  --trajectory-output results/geometric_<seq>_estimate.txt \
  --plot-output results/geometric_<seq>_trajectory.png \
  --no-display

evo_ape tum datasets/tum/rgbd_dataset_freiburg1_<seq>/groundtruth.txt results/geometric_<seq>_estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg1_<seq>/groundtruth.txt results/geometric_<seq>_estimate.txt -a -s

python scripts/plot_trajectory.py \
  --estimate results/geometric_<seq>_estimate.txt \
  --groundtruth datasets/tum/rgbd_dataset_freiburg1_<seq>/groundtruth.txt \
  --output results/geometric_<seq>_vs_groundtruth.png
```

Trajectory coverage isn't computed by any script - read the first/last
timestamp (column 1) of the estimate and of `groundtruth.txt`, divide.

---

## Geometric-only baseline (pre-ML)

**Version:** `main` @ `303e8965233817a558a849a6b2eeac8755f66f9d` (no
`V2_INTEGRATION_PLAN.md` phase implemented yet; `--depth-densify` not
passed). `freiburg1_xyz` numbers match `NOTES.md`'s existing "Ground-truth
evaluation" record exactly, confirming a faithful reproduction.

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Status |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` | 798 | 355 | 18644 / 26406 | 88.1% (26.5s / 30.1s) | 0.0796 | 0.0402 | OK |
| `freiburg1_desk` | 613 | 20 | 540 / 1142 | 5.3% (1.2s / 23.4s) | 0.0350 | 0.0415 | **Too hard** (tracking loss) |
| `freiburg1_room` | 1362 | 74 | 1476 / 2500 | 6.8% (3.3s / 48.9s) | 0.0727 | 0.0333 | **Too hard** (tracking loss) |
| `freiburg1_rpy` | 723 | 0 | 0 / 0 | 0% | - | - | **Too hard** (bootstrap failure) |

**Notes:**

- `freiburg1_xyz` is the only sequence where ATE/RPE score the whole run
  (88% coverage) - the only one usable as a baseline today.
- `freiburg1_desk`/`freiburg1_room` looked "OK" on ATE alone (both *lower*
  than `xyz`'s, despite far sparser maps) until plotted against ground
  truth (`results/geometric_{desk,room}_vs_groundtruth.png`): the estimate
  is a small scribble at the start point while ground truth sweeps through
  the whole room. The per-keyframe `TRACK` log lines
  (`results/geometric_<seq>_run.log`) show why: a steady PnP-inlier-ratio
  decline (`NOTES.md`'s known early drift/failure signal) ending in
  permanent loss with zero recovery -
  `desk` declines ~83%→~50% and dies for good at **frame 37/613**; `room`
  declines 99%→~30% and dies for good at **frame 100/1362**. Reclassified
  as too hard despite `evo` producing a number for both - see root-cause
  analysis below.
- `freiburg1_rpy` (near-pure rotation) never leaves bootstrap: 0 keyframes,
  a single identity-pose row saved. Cause: bootstrap is two-view
  essential-matrix pose + triangulation, which needs a translational
  baseline a rotation-only sequence never provides - `evo_ape` confirms
  there's nothing to score (`Degenerate covariance rank`). Expected
  limitation of monocular two-view bootstrap, not a bug; out of scope here.
- **Net result: only `freiburg1_xyz` is usable as a baseline today.**
  `desk`/`room` (tracking loss) and `rpy` (bootstrap failure) are distinct
  failure modes - a future fix might resolve one without the other.

### Root cause of `desk`/`room`'s tracking loss: a self-reinforcing death spiral

The live demo view shows the mechanism directly: match lines between the
reference keyframe and the current frame
([mapping.py:750-753](pipeline/mapping.py#L750-L753)) turn visibly
non-parallel/diagonal well before tracking dies. Three compounding gaps,
filed as issues:

1. **No cross-check/geometric filter in matching.**
   `match_descriptors` ([features.py:83-98](pipeline/features.py#L83-L98))
   is pure Hamming NN + Lowe's ratio test - no mutual-NN check, no
   epipolar/homography filter. Repetitive texture (desk edges, room
   corners) passes the ratio test easily even on a healthy frame.
   [#14](https://github.com/albinjanssonsand/slam/issues/14).
2. **Tracking runs in bursts, not continuously, so a stale reference
   compounds the problem.** PnP is only attempted once parallax against
   the *last accepted keyframe* crosses `--min-parallax`
   ([mapping.py:527-530](pipeline/mapping.py#L527-L530)); the reference
   only updates on a successful keyframe
   ([mapping.py:815-818](pipeline/mapping.py#L815-L818)). Once an attempt
   fails, the next is against the same reference at an even larger
   baseline. Narrowly filed as
   [#12](https://github.com/albinjanssonsand/slam/issues/12) (refresh the
   reference more often); **superseded by the foundational fix,
   [#15](https://github.com/albinjanssonsand/slam/issues/15) (high
   priority)** - track pose every frame via motion-predicted guided
   matching, decoupling pose estimation from keyframe insertion so there's
   no accumulating-baseline window left. #12 likely folds into #15.
3. **No relocalization once tracking is lost.** The confirmed map also
   stops growing the moment tracking fails, so PnP's candidate pool is
   frozen too - there's no wide-search recovery attempt, only the same
   narrow matching that already failed. Filed as
   [#13](https://github.com/albinjanssonsand/slam/issues/13) - deprioritized
   relative to #15: if #15 prevents most full losses, relocalization is
   only needed for residual edge cases (occlusion, extreme blur), not the
   primary mitigation.

**#15 is the primary planned fix.** `desk` (frame 37/613) and `room` (frame
100/1362) are the concrete before-state for #12/#13/#14/#15 - re-run and
compare against these rows once fixed, don't overwrite them.
