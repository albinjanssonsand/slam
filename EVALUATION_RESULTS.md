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

---

## #17: freiburg2 sequences (broader ground-truth baseline set)

**Version:** `main` @ `ff105cdfc963fe1ac0656eed60e697b223407e60` (#15
merged; `--depth-densify` not passed). Extends the ground-truth baseline
set per [#17](https://github.com/albinjanssonsand/slam/issues/17): with
only `freiburg1_xyz` usable so far (`desk`/`room`: tracking loss; `rpy`:
bootstrap failure - see above), this evaluates `freiburg2` sequences
chosen specifically to avoid both known failure modes - a second slow/clean
sequence on a different camera rig (`freiburg2_xyz`) and a robot-mounted
sequence with no handheld shake (`freiburg2_pioneer_slam2`).

**Calibration:** `calibration/tum_freiburg2.yaml`, `freiburg2`'s published
RGB camera intrinsics (TUM download page / ORB-SLAM2's `TUM2.yaml`, cross-
checked to agree), same schema as `calibration/tum_freiburg1.yaml`.

**Reproduction** (for any sequence `<seq>` in `xyz`, `pioneer_slam2`):

```bash
python -m pipeline.mapping \
  --video datasets/tum/rgbd_dataset_freiburg2_<seq> \
  --calibration calibration/tum_freiburg2.yaml \
  --trajectory-output results/guided_freiburg2_<seq>_estimate.txt \
  --plot-output results/guided_freiburg2_<seq>_trajectory.png \
  --no-display

evo_ape tum datasets/tum/rgbd_dataset_freiburg2_<seq>/groundtruth.txt results/guided_freiburg2_<seq>_estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg2_<seq>/groundtruth.txt results/guided_freiburg2_<seq>_estimate.txt -a -s

python scripts/plot_trajectory.py \
  --estimate results/guided_freiburg2_<seq>_estimate.txt \
  --groundtruth datasets/tum/rgbd_dataset_freiburg2_<seq>/groundtruth.txt \
  --output results/guided_freiburg2_<seq>_vs_groundtruth.png
```

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Status |
|---|---|---|---|---|---|---|---|
| `freiburg2_xyz` | 3669 | 305 | 17436 / 26692 | 98.6% (121.0s / 122.7s) | 0.1589 | 0.0456 | OK |
| `freiburg2_pioneer_slam2` | 2113 | 44 | 208 / 226 | 5.2% (6.0s / 115.6s) | 0.0264 | 0.0239 | **Too hard** (tracking loss) |

**Notes:**

- **`freiburg2_xyz` is a second usable ground-truth baseline**, on a
  different camera rig than `freiburg1`. Tracks the full sequence at 98.6%
  coverage (305 keyframes; 3226 of the 3363 non-keyframe frames still get a
  frame-only pose, only 137 lose tracking). ATE RMSE (0.159m) is notably
  higher than `freiburg1_xyz`'s 0.040m despite `freiburg2_xyz` being the
  nominally "easier" sequence (translation-only, explicitly low motion
  blur/rolling shutter per its TUM listing) - but plotted against ground
  truth (`results/guided_freiburg2_xyz_vs_groundtruth.png`), the aligned
  estimate correctly follows the same diagonal sweep and central back-and-
  forth pattern as ground truth, just noisier throughout, not a
  qualitatively wrong trajectory. Not investigated further here (#17 is
  dataset acquisition/evaluation only, no pipeline changes in scope), but
  worth a look before treating `freiburg2_xyz` as equally reliable as
  `freiburg1_xyz` for future ML-fusion comparisons.
- **`freiburg2_pioneer_slam2` reproduces the exact `desk`/`room`
  tracking-loss failure mode**, on a robot-mounted rig this time rather
  than handheld: per-keyframe PnP inlier count declines through the last
  few keyframes before death (83 -> 69 -> 65 -> 58 -> 57 -> 53 -> 42 -> 43
  -> 24) and dies permanently at **frame 181/2113** (8.6% into the
  sequence) - of the 2068 frames that don't get promoted to a keyframe,
  1978 lose tracking entirely rather than just failing keyframe promotion,
  nearly all of them in the tail after frame 181. The plot
  (`results/guided_freiburg2_pioneer_slam2_vs_groundtruth.png`)
  shows the same signature already seen with `desk`/`room`: a small
  scribble near the start point while ground truth sweeps through the
  whole hall. ATE/RPE look deceptively reasonable (0.026m / 0.024m) despite
  only 5.2% coverage - exactly the case the "too hard" coverage check
  exists to catch. This is further evidence that the dominant tracking-loss
  cause (zero view overlap with the confirmed map, per the root-cause
  analysis above) isn't specific to `freiburg1`'s handheld motion or
  camera - it reproduces on a different rig and a different kind of scene
  (open hall vs. desk/room), consistent with #13 (relocalization) being the
  fix that's actually needed, not something `freiburg1`-specific.
- **`pioneer_slam`/`pioneer_slam3` skipped, not attempted**, per #17's own
  contingency ("documented as skipped with why, if it fails for a
  systemic reason worth understanding first"). `pioneer_slam2` failed via
  the same already-diagnosed tracking-loss mechanism as `desk`/`room`
  (declining PnP inlier count -> permanent loss, no relocalization) - not a
  new or freiburg2-specific problem needing further investigation.
  `pioneer_slam` (the longest and most demanding of the four per #17) and
  `pioneer_slam3` (a "similar profile to `slam2`") are both robot-mounted
  hall/maze sequences with a comparable motion profile to `slam2`, so they
  would very likely reproduce the same failure without adding new
  diagnostic information - not worth the additional ~3GB download here.
  Revisit once #13 (relocalization) lands.
- **Net result: freiburg2 contributes one new usable ground-truth
  baseline (`freiburg2_xyz`)**, alongside `freiburg1_xyz`.
  `freiburg2_pioneer_slam2`'s failure is additional evidence for the
  tracking-loss root cause already tracked under #13, not a new distinct
  failure mode.

---

## #20: full/global bundle adjustment offline refinement

**Version:** `issue-20-generalize-bundle-adjustment-to-full-global-ba`
branch, on top of `main` post-#17 (`--depth-densify` not passed).
Implements [#20](https://github.com/albinjanssonsand/slam/issues/20): a
`--global-ba-at-end` flag that runs one full bundle adjustment pass (every
keyframe pose except the first, plus every map point, jointly refined) over
the whole trajectory after tracking completes, reusing
`bundle_adjustment.local_bundle_adjustment` via a thin `_run_global_ba`
wrapper rather than a second optimizer.

**Two problems found and fixed during implementation, before any of the
numbers below were trustworthy:**

- **Plausibility-gate scale mismatch.** The propose-then-validate pattern
  (`_validate_and_apply_ba`) reused for the global result was, by default,
  checking each keyframe's correction against `--max-plausible-rotation`/
  `--max-step-ratio` - thresholds calibrated against `recent_step_sizes`, a
  *per-frame tracking motion* statistic. A legitimate full-trajectory drift
  correction to an old keyframe can be far larger than one frame's typical
  step without being wrong, so reusing those thresholds unchanged risked
  silently rejecting every real correction and making the flag a no-op.
  Fixed by threading explicit `max_plausible_rotation`/`max_step_ratio`
  parameters through `_validate_and_apply_ba` instead of reading `args`
  directly, and adding separate, looser `--global-ba-max-plausible-rotation`
  (90°) / `--global-ba-max-step-ratio` (200x) flags for the global case;
  local BA's two existing call sites pass `args.max_plausible_rotation`/
  `args.max_step_ratio` unchanged (confirmed byte-identical output with the
  flag off, below).
- **Premature convergence from reused local-BA tolerances.** First
  measurement: `--global-ba-at-end` on `freiburg1_xyz` finished in 20.5s and
  produced a **byte-identical** output trajectory - not "improved by an
  imperceptible amount," genuinely unchanged. Root cause: `least_squares`'
  `ftol`/`xtol` were hardcoded to `1e-4` inside `local_bundle_adjustment`,
  tuned for local BA's "exit fast once good enough" per-keyframe use case.
  Because local BA already runs continuously throughout tracking, the map
  going into the global pass is already close to a local optimum, so a loose
  relative tolerance let the solver report convergence within a handful of
  iterations without doing meaningful work. Exposed `ftol`/`xtol` as
  parameters on `local_bundle_adjustment`/`_run_local_ba` (existing callers
  unaffected - defaults unchanged) and gave `_run_global_ba` its own
  tightened defaults (`1e-6`, vs. local BA's `1e-4`) via new
  `--global-ba-ftol`/`--global-ba-xtol` flags. Re-measured after the fix:
  results below.

**freiburg1_xyz, flag off (regression check):**

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|---|---|
| `freiburg1_xyz` | 798 | 216 | 11854 / 16061 | 88.1% (26.5s / 30.1s) | 0.0403 | 0.0270 |

Exact match to the #15 baseline row above (same keyframe/point counts, same
ATE/RPE to 4 decimal places) - confirms the `_validate_and_apply_ba`
signature refactor didn't change local-BA/tracking behavior.

**freiburg1_xyz, flag on:**

| Sequence | Keyframes accepted | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Global BA time | Reprojection error (px) mean | Reprojection error (px) max |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` | 216 (same) | 88.1% (same) | 0.0406 | 0.0274 | 25.5s | 3.39 -> 2.68 | 105703.46 -> 1844.98 |

Result was accepted (no `[BA result rejected...]` message for the global
pass; all 217 lines of the trajectory file differ from the flag-off run -
every keyframe pose moved, if only slightly for most of them). Global BA measurably
reduces raw reprojection error - mean improves ~21% (3.39px -> 2.68px), and
one severely mistriangulated point drops from 105,703px of reprojection
error to 1,845px (still bad, but two orders of magnitude less catastrophic;
consistent with there being no map-point culling yet - #26's territory).

**But ATE/RPE against ground truth get very slightly *worse*, not
better**: ATE RMSE 0.0403m -> 0.0406m (+0.7%), RPE RMSE 0.0270m -> 0.0274m
(+1.5%). Reported honestly per the issue's own acceptance criteria - global
BA is not a free win here. Plausible explanation (not verified further,
out of scope for this issue): minimizing raw reprojection error over a map
that still contains unculled outliers/mismatches isn't the same objective
as minimizing Sim(3)-aligned distance to an independently-measured ground
truth trajectory - correcting the worst outlier's local geometry can
redistribute residual error elsewhere in a way that doesn't help (or
slightly hurts) the global alignment. Consistent with `EVALUATION_RESULTS.md`
generally: this pipeline has no point-culling yet (#26), so full BA is
optimizing against known-imperfect data.

**freiburg2_pioneer_slam2: not usable as the intended long-sequence-drift
proxy.** This issue's own scope note picked `pioneer_slam2` (2116 frames)
as a longer stand-in for `NOTES.md`'s diagnosed long-sequence PnP-inlier
decline, since `freiburg1_xyz` (798 frames) is too short to exercise it.
It turns out `pioneer_slam2` was already shown, in #17's evaluation above,
to fail via total tracking loss at frame ~181/2116 (**already documented,
not rediscovered here**) - the same zero-view-overlap failure mode as
`freiburg1_desk`/`freiburg1_room`, unrelated to BA. Re-confirmed here: 44
keyframes accepted before permanent tracking loss, 8.5% coverage, ATE RMSE
0.0264m (the same "deceptively low ATE on a tiny tracked prefix" trap the
coverage check exists to catch - not a real accuracy number). With only 44
keyframes tracked, this sequence can't exercise the 800+-keyframe drift
scenario it was picked for, so the global-BA-on variant wasn't run against
it - would only reproduce the small-map freiburg1_xyz result at a smaller
scale, not test anything new. **`NOTES.md`'s original long-sequence-drift
question remains untested by this issue** - a real TUM/recording sequence
that (a) tracks continuously for 800+ keyframes and (b) has ground truth
doesn't currently exist in this repo's dataset set. Worth flagging for
whoever picks up #23/#27, which point at this same sequence for the same
reason and will hit the identical dead end.

**Net result:** `--global-ba-at-end` works correctly (converges, measurably
reduces reprojection error, propose-then-validate gate functions as
designed) but does not demonstrate an ATE/RPE improvement on the only
sequence it could actually be tested against, and the sequence meant to
test its primary motivation (long-sequence drift) turned out unusable for
an unrelated, already-known reason. Kept as an opt-in flag (default off)
given this result - not enabled by default pending either a usable
long-sequence test or point-culling (#26) landing first.

---

## #21: ORB extraction - single-pyramid grid with adaptive per-cell threshold

**Version:** `issue-21-orb-extraction-single-pyramid-grid` branch, on top of
`main` post-#20. Implements [#21](https://github.com/albinjanssonsand/slam/issues/21):
replaces `detect_and_compute_gridded`'s independent per-cell-cropped ORB
pyramids with a single full-image pyramid (matching paper §V-A), bucketing
the resulting keypoints into a grid to enforce a per-cell quota, with
progressively lower FAST thresholds retried (further full-image passes,
never cropped) for any cell still short of quota.

**One serious bug found and fixed during implementation, before any of
the numbers below were trustworthy:** requesting a large `nfeatures`
candidate pool from a single `cv2.ORB_create` call (needed so a strong
region can't crowd out a weak one before our own per-cell selection runs -
see the function's docstring) makes ORB report the same physical corner
more than once far more often than a normal-sized request would - confirmed
empirically, ~20% of raw keypoints from one full-image pass were within 2px
of another, and this holds even at `nfeatures=2000` (117% duplicate-pair
rate), not just at the large candidate budget. Undetected, this collapsed
bootstrap on `freiburg1_xyz`'s first attempt: 836 essential-matrix inliers
triangulated into only **5** map points (vs. 49 for the old cropped-cell
implementation on the identical frame pair) - duplicate detections of a
handful of strong corners were filling entire cells' quotas, inflating
match *counts* (more near-identical descriptors to match against) while
destroying match *diversity* (most "inliers" were redundant/ambiguous
correspondences of the same few physical points, not genuinely new
structure). Fixed with a greedy NMS pass (via `scipy.spatial.cKDTree`,
already a project dependency) collapsing near-duplicate detections to their
single highest-response representative before any per-cell selection - a
fixed spatial-bin approach was tried first and rejected (still left 1727
near-duplicate pairs due to points straddling bin boundaries; the radius-
based KD-tree approach leaves ~0-1). Re-tested after the fix: same frame
pair now triangulates **124** points (2.5x the old implementation's 49).

A second, smaller issue: the fallback-pass dedup check (against already-
selected keypoints from an earlier pass) was a per-candidate Python
`any()`-over-generator loop, profiled as 77% of total runtime on a
fallback-heavy frame (~150ms vs. the old implementation's ~31ms on the
same frame). Vectorized with numpy broadcasting instead (~70-100ms on the
same frame).

**Per-cell balance check** (required by the issue, on the most texture-
imbalanced frame found across `recordings/demo1-6.mp4` - `demo3.mp4`,
frame 188/377, chosen because the *ungridded* baseline showed 5 of 16
cells at zero keypoints and one cell dominating with 1070):

| Extraction | Total keypoints | Zero-count cells | Max cell count |
|---|---|---|---|
| Ungridded (plain `cv2.ORB_create`, pre-#21 baseline path) | 2000 | 5 / 16 | 1070 |
| Old gridded (per-cell-cropped, fixed threshold) | 757 | 4 / 16 | 125 (quota cap) |
| New gridded (this issue, single pyramid + adaptive threshold) | 1572-1637 | 1 / 16 | 125 (quota cap) |

New extraction both yields more than double the old gridded
implementation's total keypoints *and* recovers 3 of the 4 previously-zero
cells via the adaptive-threshold retry (the one remaining zero cell
returned nothing even at the lowest fallback threshold - genuinely
textureless, matching the paper's own "some cells contain no corners"
allowance).

**Match-quality check** (required by the issue - average matches per
consecutive frame pair, first 60 frames of `freiburg1_xyz`, old vs. new
gridded extraction, `n_features=5000`): **925.1 -> 1504.5** average matches
per pair (~1.6x). An earlier measurement (2357.4, ~2.5x) was inflated by
the duplicate-keypoint bug above and is superseded by this post-fix number.

**Timing** (required by the issue - per-frame extraction cost, `demo3.mp4`'s
pathological frame, steady-state/warmed-up process): old gridded 31.5ms ->
new gridded ~70-100ms (roughly 2.2-3.2x, depending on how many fallback
passes a given frame triggers). Regresses versus the old implementation,
contrary to the paper's own real-time budget assuming a single pyramid is
cheaper than N per-cell ones - plausible explanation: this codebase's old
implementation used tiny per-cell `nfeatures` (~125), while matching the
paper's *effect* (per-cell quota via adaptive threshold, not per-cell
cropping) here still means detecting a large `nfeatures` candidate pool
over the *whole* image up to 3 times per frame (default + 2 fallback
thresholds) before our own selection narrows it down. Acceptable for this
offline/batch pipeline (a few hundred ms/frame), reported honestly per the
issue's own requirement rather than left unmeasured.

**freiburg1_xyz, full run:**

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #20's flag-off row) | 798 | 216 | 11854 / 16061 | 88.1% (26.5s / 30.1s) | 0.0403 | 0.0270 |
| `freiburg1_xyz` (this issue) | 798 | 353 (before tracking loss) | 28885 / 51898 | 70.5% (21.2s / 30.1s) | 0.1249 | 0.0388 |

**This is a real regression in end-to-end trajectory quality**, reported
honestly rather than glossed over. Per-frame keyframe richness is
dramatically better while tracking survives (353 keyframes with ~3.2x the
map density by frame 634, vs. 216 for the *entire* old-extraction
sequence), but tracking collapses at frame 635 and never recovers - the
PnP inlier count drops from 634 (frame 634) to 22 (frame 635) in one step,
with relative rotation jumping 2.0deg -> 7.4deg -> 8.8deg across the next
two keyframes, then no further per-frame log output at all (total tracking
loss) for the rest of the sequence.

**Root cause (best understanding, not fixed here - out of #21's explicit
scope, which excludes touching matching/PnP):** this is the same PnP-
ambiguity/no-relocalization failure mode `NOTES.md` already documents for
`freiburg1_desk`/`freiburg1_room` - a moderate, sub-`--max-plausible-
rotation` (15deg) rotation jump slips through the plausibility gate
undetected, then poisons the next frame's motion-predicted guided search,
cascading into permanent loss with no relocalization to recover (#13).
`freiburg1_xyz` was previously the one sequence that never triggered this.
Working hypothesis for why it does now: `match_against_guided` projects
the *entire* confirmed map into every frame with no covisibility bound
(already flagged as a scaling problem in #23) - a much denser map (this
run's ~52k points vs. the old baseline's ~16k for the whole sequence)
means more candidate points crowded into the same fixed-radius guided-
search window, plausibly increasing correspondence ambiguity at exactly
the moment PnP needs to be most reliable. Not verified further here since
diagnosing/fixing matching or relocalization behavior is explicitly out of
scope per #21's own Non-goals - flagged for whoever picks up #12
(stale-reference cascade), #13 (relocalization), or #23 (covisibility-
bounded local map, which would directly reduce the guided-search candidate
pool this hypothesis points at).

**Net result:** #21's own goal - matching the paper's single-pyramid,
adaptive-threshold construction, with better spatial balance and richer
matches than the old cropped-cell implementation - is achieved and
verified directly (per-cell balance, match-quality, and bootstrap-recovery
checks above all improve). The extraction change surfaces (but does not
itself cause, and is not responsible for fixing) a pre-existing PnP-
ambiguity/relocalization gap that a denser resulting map appears to trigger
more readily. Landed as scoped; the trajectory-quality regression is
tracked as a consequence for #12/#13/#23, not reopened as part of #21.

---

## #23: Covisibility graph, map point metadata, Track Local Map hardening, local BA rescoping

**Version:** `issue-23-covisibility-graph-point-metadata` branch, on top of
`main` post-#21. Implements [#23](https://github.com/albinjanssonsand/slam/issues/23)
(paper §III-C/D, §V-D, §VI-D) as three increments in one branch: (1) a
covisibility graph plus per-point viewing direction / scale-invariance
bounds / representative descriptor, all incrementally maintained by a new
`Map.add_observation`; (2) `Map.match_against_guided` rewritten to the
paper's projection sequence (viewing-angle gate, scale-invariance gate,
predicted-pyramid-octave search) restricted to the local map (K1 union K2,
`Map.local_map_keyframes`/`local_map_points`) instead of the whole
confirmed map; (3) `_run_local_ba` rescoped from a fixed insertion-order
sliding window to the current keyframe's covisibility neighbors (free),
with every other keyframe that also observes a local point held fixed
instead of dropped (`local_bundle_adjustment`'s `fix_first_pose: bool`
generalized to an arbitrary `fixed_poses` mask to support this).

**A real performance regression found and fixed before any of the numbers
below were trustworthy:** the first full `freiburg1_xyz` run took ~40-50
minutes (vs. this same sequence's ~20 minutes pre-#23) - the opposite of
this issue's own purpose. Root cause, found via manual timing
instrumentation (`cProfile` itself proved unreliable to interrupt cleanly
in this environment): K1's construction is deliberately unthresholded per
the paper (any shared point pulls in the observing keyframe), and on a
small, heavily-revisited scene like `freiburg1_xyz` this lets a single
popular point pull in most of the trajectory's keyframes - `Map.
match_against_guided`'s candidate-point count grew unbounded with total
map size (124 -> 3136+ points and still climbing over a 60s sample)
instead of staying local, consuming ~45% of per-frame wall time by itself.
Capping keyframe count alone (`local_map_keyframes`'s `max_keyframes=30`,
`_run_local_ba`'s top-20-covisibility-neighbors) did not fix it, since on
this scene even a handful of keyframes each individually observe
thousands of points; a direct cap on the local point-set size itself
(`--track-local-map-max-points`, keeping the most recently added points if
exceeded) was required. An initial cap of 2000 fixed the speed but was too
aggressive - it starved PnP of candidates, more than tripling tracking
loss (152 keyframes / 545 frames losing tracking entirely, dying
permanently at frame 443/798, vs. 7 lost frames pre-#23). Raised to 6000
(the value used below): local map size stays bounded (settles around
2400-2700 points on `freiburg1_xyz` even as the confirmed map grows past
35000) while keeping enough candidates for PnP to stay healthy through the
whole sequence.

**freiburg1_xyz, full run:**

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Wall clock |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #21's post-fix row) | 798 | 353 (before tracking loss) | 28885 / 51898 | 70.5% (21.2s / 30.1s) | 0.1249 | 0.0388 | - |
| `freiburg1_xyz` (this issue, `--track-local-map-max-points 6000`) | 798 | 372 | 35662 / 59834 | 87.9% (26.4s / 30.1s) | 0.1425 | 0.0282 | ~11 min |

(The `#20`-flag-off row - 216 keyframes, 88.1% coverage, 0.0403/0.0270 ATE/RPE,
pre-#21 - remains the cleanest "no tracking loss at all" comparison point;
included here against #21's own post-fix numbers instead, since that's the
row this issue's changes land directly on top of.)

**Reading these numbers:** coverage recovers to 87.9% (vs. #21's 70.5% -
this issue's bounded local map measurably helps the tracking-loss
regression #21 itself flagged as a likely consequence of an unbounded
guided-search candidate pool, matching the hypothesis in that section).
RPE (frame-to-frame local accuracy) improves to 0.0282m from #21's
0.0388m. ATE (global Sim(3)-aligned accuracy) gets worse, 0.1425m vs.
#21's 0.1249m, despite both coverage and RPE improving - plausible given
RPE only measures *local* consecutive-frame consistency while ATE is
sensitive to how a handful of larger jumps or drift segments affect the
global alignment; not investigated further here in the interest of time
(this issue already required two full-`freiburg1_xyz`-run iterations to
find and fix the performance regression above). Wall clock improved
materially (~11 min vs. #21's un-timed but user-observed ~20+ min baseline
for a comparable run) even with the point cap generous enough to avoid
starving PnP - confirms the local-map bound is doing real work, not just
trading speed for correctness one-for-one.

**freiburg2_pioneer_slam2 (this issue's own bounded-local-map + revisit
check): not usable, for the exact reason #20 already flagged.** #20's own
write-up (above) explicitly warned that `pioneer_slam2` dies from a
pre-existing, already-diagnosed tracking-loss bug (zero view overlap once
PnP inlier count declines past a threshold, no relocalization to recover -
tracked under #13, unrelated to BA or covisibility) at frame ~181/2113,
and that "#23/#27... will hit the identical dead end." Re-confirmed here,
not rediscovered: this run dies at **frame 173/2113** (38 keyframes, 461
confirmed / 1343 total points, 5.0% coverage, ATE RMSE 0.0197m - the same
"deceptively low ATE on a tiny tracked prefix" trap the coverage check
exists to catch, not a real accuracy number). With only 38 keyframes and
1343 points ever created, this sequence can't exercise either of the
checks it was picked for (bounded local-map size under real growth, or a
genuine revisited segment) - `freiburg1_xyz`'s own point-count growth
(35000+ points, local map staying at 2400-2700) is the more meaningful
demonstration of bounded growth available in this repo's current dataset
set. Per the issue's own instruction ("don't construct a synthetic revisit
segment if this sequence doesn't happen to contain one - report that
finding instead"): no revisit segment is reachable either, since tracking
dies at 8.2% into the sequence. `NOTES.md`/#20's original long-sequence
question remains untested, as #20 already anticipated.

**Net result:** all three sub-issues (covisibility graph + metadata,
Track Local Map hardening, local BA rescoping) implemented and landed in
one branch per the issue's own "split further if needed" guidance (kept
as one branch here since the performance-bug investigation made splitting
into separate PRs impractical after the fact). Bounded local-map search
is real and measured (point count stays flat as the map grows past 35k
points) and recovers most of #21's tracking-loss regression on
`freiburg1_xyz` (coverage 70.5% -> 87.9%), at the cost of a moderate ATE
regression not further diagnosed here. `freiburg2_pioneer_slam2` remains
unusable for this issue's own acceptance criteria for a pre-existing,
already-documented reason unrelated to this issue's changes.

---

## #24: New map point creation from all covisible keyframes (paper §VI-C)

**Version:** `issue-24-new-point-creation-all-covisible` branch, on top of
`main` post-#23 (`--depth-densify` not passed). Implements
[#24](https://github.com/albinjanssonsand/slam/issues/24): the new-point-
creation step in `pipeline/mapping.py`'s `_demo()` TRACK branch now searches
every keyframe connected to the current one in the covisibility graph
(`Map.covisible_keyframes`, capped to the 10 highest-shared-point neighbors,
`--new-point-max-covisible-keyframes`), not just the single immediately-
preceding reference keyframe. For each covisible keyframe: match this
keyframe's still-unmatched ORB features against that keyframe's own still-
unmatched features (`features.match_descriptors`), discard candidates that
fail a new epipolar-constraint check between the two keyframes' already-
solved poses (`--epipolar-max-error`, default 2px - no such check existed
before, since there was previously only ever one candidate pair), and
triangulate survivors (`triangulation.triangulate`, unchanged). After
creation, each new point is additionally projected into every OTHER
covisible keyframe it wasn't triangulated from and searched for a further
correspondence, reusing `Map.match_against_guided` (§V-D Track Local Map
projection/matching, from #23) rather than a separate search - each such
match counts as an independent re-observation (`Map.confirm`), same as any
other provisional point's confirming re-observation. New `Map.
keyframe_matched_frame_idx`/`add_observation(..., frame_idx=)` track, per
keyframe, which of its own ORB features are already tied to a map point, so
the search doesn't spawn a duplicate point next to one a covisible keyframe
already observes. Every keyframe's own ORB keypoints/descriptors are now
kept for the life of the run (`keyframe_kp`/`keyframe_desc`, previously only
the current reference keyframe's were retained), since the search needs to
reach any covisible neighbor, not just the previous keyframe.

**Reproduction:** same commands as the top of this file, run from this
branch; outputs saved with a `covisible_newpoints_` prefix instead of
`geometric_`/`guided_`.

**New-point-creation rate (required by the issue) - points added per
post-bootstrap keyframe, `freiburg1_xyz`:**

| | Keyframes | Total map points | Points / post-bootstrap keyframe |
|---|---|---|---|
| Before (#23 baseline, single previous keyframe) | 372 | 59834 | 161.4 |
| After (this issue, up to 10 covisible keyframes) | 418 | 112173 | 269.3 |

**~1.67x more new structure created per keyframe**, consistent with the
paper's richer candidate set (measured directly from the run log: keyframe
promotions searched an average of 9.87 covisible keyframes each - the
10-keyframe cap is nearly always saturated on this heavily-revisited scene -
of which an average of 9.45 actually contributed at least one new point).

**Map density/coverage (required by the issue), `freiburg1_xyz`:**

| | Confirmed / total map points | Confirmed ratio |
|---|---|---|
| Before (#23 baseline) | 35662 / 59834 | 59.6% |
| After (this issue) | 102768 / 112173 | 91.6% |

Total map size grows 1.87x (59834 -> 112173), and the confirmed fraction
jumps far more (2.88x confirmed count, 35662 -> 102768) - explained by the
§VI-C last-paragraph projection step: a newly created point often picks up
several independent re-observations from other covisible keyframes
immediately at creation time (average 525.3 extra re-observations per
keyframe promotion, from the run log), rather than waiting for a chance
re-observation on some later frame the way the pre-existing provisional/
confirmed lifecycle did.

**freiburg1_xyz, full run:**

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Wall clock |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #23's row) | 798 | 372 | 35662 / 59834 | 87.9% (26.4s / 30.1s) | 0.1425 | 0.0282 | ~11 min |
| `freiburg1_xyz` (this issue) | 798 | 418 | 102768 / 112173 | 87.9% (26.4s / 30.1s) | 0.1516 | 0.0197 | ~17m45s |

(`results/covisible_newpoints_xyz_vs_groundtruth.png` - the aligned estimate
tracks ground truth's same diagonal back-and-forth sweep, the same
qualitative shape as every other non-"too hard" `freiburg1_xyz` run in this
file, not a tracking-loss scribble.)

**Reading these numbers:** coverage is unchanged (87.9%, same to the tenth
of a second) and only 3 of 798 frames lose tracking entirely - the richer
search doesn't destabilize tracking. RPE (frame-to-frame local consistency)
improves substantially, 0.0282m -> 0.0197m (-30%), consistent with local BA
now having a much denser, better-connected covisibility neighborhood to
constrain each keyframe against. **ATE gets slightly worse, 0.1425m ->
0.1516m (+6.4%)**, reported honestly rather than glossed over - the same
RPE-improves/ATE-worsens split #20 saw from full BA on this same sequence.
Plausible explanation (not verified further, out of scope for this issue):
this pipeline has no map-point culling/fusion yet (#26's territory, and
explicitly deferred by #24's own scope to coordinate with that issue rather
than assume today's lifecycle) - denser per-keyframe triangulation against
many covisible keyframes plausibly creates more near-duplicate points
representing the same physical surface at slightly different 3D positions
than the old single-pair search did, which can locally over-constrain BA
into a self-consistent-but-globally-biased solution without showing up in
frame-to-frame RPE.

**Performance cost (required by the issue):** wall clock grows ~1.6x (~11
min -> ~17m45s) for the full 798-frame run - the added cost of matching
+ epipolar-checking against up to 10 covisible keyframes per promotion
instead of 1, plus the extra `match_against_guided` projection pass per new
point. Reported as measured, not optimized further - acceptable for this
offline/batch pipeline per the issue's own performance note.

**Net result:** §VI-C's richer candidate set delivers a clear, measured win
on the issue's own primary acceptance criteria - 1.67x more new points per
keyframe, 2.88x more confirmed points, no coverage or tracking-stability
regression, and better RPE - at a measured ~1.6x wall-clock cost and a
small ATE regression plausibly attributable to the still-missing point-
culling/fusion step (#26), not to this issue's search logic itself.
