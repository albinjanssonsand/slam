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

**Update:** #15 has landed - see the "#15: per-frame motion-predicted guided
tracking" section below for results. **It does not fix `desk`/`room`'s
death spiral** - death point moves 3-4 frames out of 600-1300+, coverage is
still 5-7%, and once dead, >98% of remaining frames fail even frame-only
tracking (not just keyframe promotion). The mechanism this section blamed
(stale reference, accumulating baseline) turned out not to be the dominant
cause on these two sequences: a window-size test (below) rules out "search
radius too small" and points instead at total loss of view overlap with the
map - nothing in the confirmed pool is visible at all, so guided matching
has nothing to guide toward. **This reverses the priority call below: #13
(relocalization) should be re-elevated, not treated as a residual edge
case** - it looks like the dominant failure mode for this kind of motion,
not a rare one. #15 is still a real, worthwhile win on sequences that don't
fully lose tracking (`xyz`'s ~2x ATE improvement at equal coverage), just
not the fix for the problem that motivated filing it.

---

## #15: per-frame motion-predicted guided tracking

**Version:** `issue-15-per-frame-motion-predicted-guided-tracking` branch,
on top of the geometric-only baseline above (`--depth-densify` not passed).
Implements the scope of
[#15](https://github.com/albinjanssonsand/slam/issues/15): a constant-
velocity motion prediction (`pose.predict_constant_velocity`) plus windowed/
guided descriptor matching (`Map.match_against_guided`, a `scipy.spatial.
cKDTree` radius query around each confirmed point's predicted projection,
new `--guided-window` px, default 60) attempted every frame, decoupled from
keyframe insertion - `--min-parallax` now gates only whether a frame's
already-estimated pose also becomes a keyframe (triangulation/confirmation/
BA), not whether a pose is attempted at all.

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Status |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` | 798 | 216 | 11854 / 16061 | 88.1% (26.5s / 30.1s) | 0.0403 | 0.0270 | OK |
| `freiburg1_desk` | 613 | 25 | 748 / 1583 | 5.8% (1.4s / 23.4s) | 0.0392 | 0.0431 | **Too hard** (tracking loss) |
| `freiburg1_room` | 1362 | 80 | 1726 / 3017 | 7.0% (3.4s / 48.9s) | 0.0487 | 0.0418 | **Too hard** (tracking loss) |

**Reproduction:** same commands as above, run from this branch.

**`freiburg1_xyz`: clear win, same coverage.** Tracks the full sequence at
identical 88.1% coverage to the baseline, but both error metrics improve
substantially - ATE RMSE 0.0796m -> 0.0403m (~2x better), RPE RMSE 0.0402m ->
0.0270m. Keyframe count is lower (216 vs. 355): expected, not a regression -
many frames that previously had to become a keyframe just to get *any* pose
estimate now get tracked frame-only instead (260 such frames this run,
`n_tracked_only` in the run log) without needing the full triangulation/
confirmation/BA cost of a keyframe. Per-frame PnP inlier counts stay high
through the *entire* sequence (frame 792: 634/1694; frame 797: 647/1577) -
no terminal decline, unlike the baseline log's later keyframes. Full run
took 7m54s wall-clock (CPU-only, 798 frames including bootstrap/BA/depth
setup) - not directly compared against a baseline timing (none was recorded
when that run produced `results/geometric_xyz_run.log`), so this is a
measured absolute number, not a verified "guided is cheaper" delta; per
`match_against_guided`'s design (a `cKDTree` radius query replacing an
all-pairs brute-force search), it should be, but that comparison wasn't
directly profiled here.

**`freiburg1_desk`/`freiburg1_room`: #15 did not fix this - the death point
barely moves, and what happens after it is worse than the frame numbers
alone suggest.** `desk` survives to frame 41/613 (was 37, +4 frames) with
25 keyframes (was 20); `room` to frame 103/1362 (was 100, +3 frames) with
80 keyframes (was 74). Trajectory coverage is nearly unchanged: 5.3% -> 5.8%
(`desk`), 6.8% -> 7.0% (`room`). The run log's new frame-accounting (which
now distinguishes "still tracked frame-only" from "lost tracking entirely"
among frames not promoted to a keyframe) makes the severity clear: on
`desk`, only 5 of the 587 post-keyframe-41 frames get even a frame-only
pose - **582 lose tracking entirely**; on `room`, 1264 of 1281 do. This
isn't "keyframe insertion stayed strict while rough tracking continued" -
the per-frame tracker itself gives up almost immediately past the same
point the baseline died at.

Diagnosed the cause by testing whether it's a guided-window sizing problem:
re-running `desk` with `--guided-window 200` (vs. the default 60px - both
scale up further via the same adaptive growth on consecutive per-frame
failures, capped at 5x, so up to 1000px effective radius in this test)
still dies permanently at frame 39, *one frame earlier* than the
default-window run. Window size is not the bottleneck - this rules out
"points are there but outside the search radius" and points instead at
**zero confirmed map points in view at all**. No amount of guided-search
tuning recovers that, since there is nothing in the confirmed pool to guide
toward - continued mapping/relocalization when genuinely lost is #13's job,
not #15's (see #15's own non-goals). Reclassified as still "too hard" by
the same coverage criterion as the baseline; the keyframe-count/death-point
movement is real but not meaningful at this magnitude.

**Two implementation bugs found and fixed during this work, both worth
recording:**

- Moving the running pose update (`R_pos, t_pos = R_new, t_new`) to fire on
  every accepted per-frame pose (needed so motion prediction has a current
  estimate) broke the *keyframe*-insertion triangulation call, which still
  read `R_pos`/`t_pos` expecting it to hold the *reference keyframe's* pose
  (triangulation needs two distinct views). By the time that call ran,
  `R_pos` already equaled the new frame's own pose - a zero-baseline,
  degenerate triangulation producing points with reprojection errors in the
  hundreds to tens of thousands of pixels, silently poisoning every
  provisional-point re-observation check (always 0 re-observed, confirmed
  pool frozen at the bootstrap count forever). Fixed by tracking the
  reference keyframe's own pose separately (`ref_R`/`ref_t`, refreshed
  alongside `ref_kp`/`ref_desc` on every keyframe promotion).
- Constant-velocity prediction extrapolated whatever relative motion was
  last observed between two accepted poses, even when several frames had
  failed to track in between - so the "one frame's worth of motion" it
  computed actually spanned multiple frames, and extrapolating it another
  single frame ahead overshot badly right after any recovery. Fixed by
  tracking the frame index each accepted pose came from and only trusting
  the constant-velocity prediction when both the previous-to-current and
  current-to-this-frame gaps are exactly 1; otherwise predicting "no
  motion" (repeat the last pose) instead.

**Net result:** #15 does not fix what it was filed to fix. `desk`/`room`'s
death spiral was hypothesized (in the root-cause analysis above) to be
mostly a stale-reference/accumulating-baseline problem that continuous
guided tracking would resolve; it wasn't - both sequences still die
permanently within a handful of frames of the original baseline, and once
dead, tracking is fully lost (not just under-promoted) for essentially all
remaining frames. The actual dominant cause is zero view overlap with the
confirmed map, which is #13's (relocalization) territory, not #15's - and
on this evidence, #13 looks like the primary fix needed for this failure
mode, not a residual safety net as originally assessed.

#15 is still worth keeping: it's a real, non-regressing accuracy
improvement on sequences that don't fully lose tracking (`xyz`'s ~2x
ATE / ~33% RPE improvement at equal coverage), it fixed two genuine
correctness bugs along the way (below), and its motion-prediction /
per-frame-tracking infrastructure is very likely what #13's relocalization
trigger (detecting sustained per-frame tracking failure) should build on
top of. It just isn't, by itself, the fix for tracking loss.
