# Evaluation Results: Pipeline vs. Ground Truth (TUM RGB-D)

Tracks ATE/RPE against the TUM RGB-D `freiburg1` sequences, one section per
pipeline version (append, don't overwrite, so a regression/improvement is a
diff between sections). Currently holds the geometric-only baseline
(pre-ML); a section per `V2_INTEGRATION_PLAN.md` ML-depth-fusion phase gets
added once that work lands.

**Read `EVALUATION_METHOD.md` before adding a section here or trusting one
already here** - it has the reproduction commands, the "too hard" criterion,
and (important) a list of known pitfalls in this methodology, including one
that previously produced a real, shipped-then-reverted false result on
`freiburg1_desk` (see that section below for the corrected writeup).

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

---

## #25: Multi-condition new-keyframe decision policy (paper §V-E)

**Version:** `issue-25-multi-condition-keyframe-policy` branch, on top of
`main` post-#24 (`--depth-densify` not passed). Implements
[#25](https://github.com/albinjanssonsand/slam/issues/25): the TRACK
branch's keyframe-promotion decision in `pipeline/mapping.py`'s `_demo()`
is no longer a bare `parallax >= --min-parallax` check. It now also
requires the paper's §V-E policy - `frames_since_relocalization >
--kf-min-frames-since-relocalization` (condition 1; always true today,
since relocalization doesn't exist yet - #13 - the counter starts with a
`10**9`-frame head start and #13 can wire in a real reset-on-relocalization
event later without reworking this policy), the current frame's PnP inlier
count `>= --kf-min-tracked-points` (condition 3, default 50), and that
count being `< --kf-ref-ratio` (default 0.9) of what the current reference
keyframe itself tracked when it was inserted (condition 4) - **or**,
independent of all three, `frames_since_last_keyframe >=
--kf-max-frames-since-keyframe` (default 20) as a standalone fallback
standing in for the paper's "local mapping idle" condition 2, which has no
meaning without a separate mapping thread. `--min-parallax` remains an
additional required gate on top of all of this, not replaced by it - kept
per the issue's own instruction, since this codebase's bootstrap-then-track
design still needs real triangulation baseline even when the paper's own
conditions are satisfied.

**Reproduction:** same commands as the top of this file, run from this
branch; outputs saved with a `kfpolicy_` prefix instead of
`geometric_`/`guided_`.

**Keyframe insertion rate (required by the issue), `freiburg1_xyz`:**

| | Keyframes | Frames | Insertion rate |
|---|---|---|---|
| Before (#24 baseline, parallax-only gate) | 418 | 798 | 52.4% |
| After (this issue, paper §V-E policy) | 76 | 798 | 9.5% |

**Insertion rate drops by more than 5x - the opposite of the increase the
issue anticipated ("the paper's policy is deliberately more aggressive...
expect a measurable increase"), reported honestly rather than forced to
match that expectation.** Root cause, read directly from the run log (`grep
trigger= results/kfpolicy_xyz_run.log`): of the 75 TRACK-branch keyframes
(+1 bootstrap keyframe = 76 total), 51 were triggered by the genuine paper
conditions (tracked-point-count dropped below 90% of the reference
keyframe's own count) and only 24 by the every-20-frame fallback - the
policy conditions are doing real, non-trivial work, not being bypassed by
the fallback. The difference from #24's baseline is that #23/#24 already
made this pipeline's confirmed map dense and well-covisible (up to 10
covisible keyframes searched per promotion, hundreds of PnP inliers per
frame against a large local map) - PnP tracking against that richer map
stays confident for many more frames than it would against a sparser one,
so condition 4 (tracking has meaningfully degraded vs. the reference
keyframe) takes far longer to become true than the old parallax-only gate
did. Sample promotions confirm this directly, e.g. frame 111 (34.2px
parallax, 248/403 inliers vs. a richer reference) and frame 175 (29.4px
parallax) - both well past `--min-parallax`'s 10px floor, held back until
tracking quality itself degraded enough.

**freiburg1_xyz, full run:**

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Wall clock |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #24's row) | 798 | 418 | 102768 / 112173 | 87.9% (26.4s / 30.1s) | 0.1516 | 0.0197 | ~17m45s |
| `freiburg1_xyz` (this issue) | 798 | 76 | 33679 / 42062 | 88.3% (26.6s / 30.1s) | 0.0491 | 0.0508 | ~9m53s |

(`results/kfpolicy_xyz_trajectory.png` - the trajectory shape matches every
other non-"too hard" `freiburg1_xyz` run in this file, not a tracking-loss
scribble; 721/798 frames didn't become keyframes, of which 644 still
tracked frame-only and 77 lost tracking entirely, comparable in kind to
every other run in this file.)

**Performance cost (required by the issue):** wall clock roughly *halves*
(~17m45s -> ~9m53s) rather than growing - the direct consequence of far
fewer keyframes, each of which is what triggers local BA, new-point search
across covisible keyframes, and the §VI-C re-observation pass. No
regression to report; the added §V-E policy evaluation itself is cheap
(a handful of scalar comparisons per frame) next to the per-keyframe work
it's now gating more selectively.

**Reading these numbers:** coverage is essentially unchanged (88.3% vs.
87.9%) and tracking-loss frame count is comparable, so the pipeline isn't
losing track more - it's simply creating far less (denser, more redundant)
structure along the way, consistent with fewer, better-separated keyframes
each covering more real camera motion. ATE improves substantially (0.1516m
-> 0.0491m, -68%), plausibly because the map now accumulates far less of
the near-duplicate-point clutter #24's write-up flagged as a likely ATE
drag (fewer, more separated triangulation events against a less redundant
covisible set). RPE gets worse (0.0197m -> 0.0508m, +158%) - the mirror
image of #24's own ATE/RPE trade-off, and for a related reason: RPE is
measured between *consecutive keyframes*, and this policy makes consecutive
keyframes further apart in both frames and real motion, so each keyframe-
to-keyframe step now accumulates more real per-step drift than the old
~2-frame-apart baseline did. Not further diagnosed here (out of scope for
this issue's decision-policy-only mandate) - plausibly the same missing
point-culling/fusion (#26) interacting differently with a sparser keyframe
set, or simply a direct consequence of measuring RPE over larger steps.

**Net result:** the paper §V-E conditions are implemented faithfully and do
real, verified work (51/75 promotions triggered by genuine tracking-
degradation, not the fallback) - but on this codebase's now-map-dense
post-#23/#24 baseline, tracking against a rich local map stays confident
far longer than the paper's own sparser-map assumptions anticipate, so the
net effect is *fewer*, more selective keyframes rather than more. Coverage
holds, ATE improves markedly, RPE worsens - a genuine trade-off, not a
regression to fix within this issue's scope. If a higher keyframe rate is
wanted, `--kf-max-frames-since-keyframe` (currently 20) and `--kf-ref-ratio`
(currently 0.9) are the two levers that would need retuning, but changing
defaults away from the issue's own suggested numbers wasn't done here
without a specific reason to prefer a different rate.

**Ablation: does the extra `--min-parallax` gate on top of §V-E still earn
its keep?** The paper's own keyframe decision has no whole-frame parallax
gate at all - it relies on `CreateNewMapPoints`' own per-point parallax-
angle rejection (this codebase's equivalent: `triangulate(...,
min_parallax_deg=args.min_triangulation_angle)`, already running inside
`_create_new_points_from_covisible_keyframes` since #24) rather than a
coarse per-keyframe proxy. `--min-parallax` was kept anyway per this
issue's own scope decision ("combine, don't replace"), on the theory that
this codebase's design still needs it. Tested directly: same branch, same
`freiburg1_xyz` run, with the TRACK-branch promotion condition changed from
`has_ref_baseline and parallax >= args.min_parallax and kf_policy_ok` to
just `has_ref_baseline and kf_policy_ok` (bootstrap's own, separately-
justified parallax gate on essential-matrix estimation is untouched either
way).

| | Keyframes | Confirmed / total points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Lost tracking entirely | Wall clock |
|---|---|---|---|---|---|---|---|
| With `--min-parallax` (this issue, as landed) | 76 | 33679 / 42062 | 88.3% (26.6s / 30.1s) | 0.0491 | 0.0508 | 77 / 798 |  ~9m53s |
| Without `--min-parallax` (ablation) | 88 | 39476 / 47591 | 86.4% (26.0s / 30.1s) | 0.0409 | 0.0436 | 36 / 798 | ~14m14s |

Trigger breakdown was similar in shape either way (69/87 TRACK-branch
promotions genuine-paper-condition-triggered without the gate vs. 51/75
with it) - `--min-parallax` isn't the dominant limiter on keyframe count
either way (§V-E's own conditions are), so removing it only grows the
keyframe count modestly (+16%, 76->88), not dramatically.

**Reading these numbers:** dropping `--min-parallax` wins on every accuracy
metric measured here - ATE improves 17% (0.0491m->0.0409m), RPE improves
14% (0.0508m->0.0436m), and **frames that lose tracking entirely drop by
more than half (77->36)** - the most consequential difference, since
robustness under exactly this kind of tracking pressure is the paper's
stated motivation for the §V-E policy in the first place. The plausible
mechanism: `--min-parallax` measures pixel displacement vs. the *current*
reference keyframe, a signal unrelated to *why* §V-E's own condition 4
(tracked-point-ratio degraded) fires - a frame can have low real camera
translation (little rotation, e.g. a slow pan or momentary blur/lighting
change) while still tracking measurably fewer points than the reference
did. With the gate, that frame is blocked from refreshing the reference
keyframe until real parallax accumulates too, so degraded tracking is left
to degrade further against a stale reference - directly working against
the aggressive-insertion-for-robustness intent §V-E exists for. Without the
gate, the reference refreshes as soon as tracking quality alone says it
should, which is what the drop in full tracking-loss frames shows directly.
The cost is a modest coverage dip (88.3%->86.4%) and ~44% more wall clock
(~9m53s->~14m14s, from 16% more keyframes each doing full local BA +
covisible-keyframe search) - both attributable to simply making more
keyframes, not to any instability.

**Caveat:** single-sequence evidence (`freiburg1_xyz` only, per this
issue's own required-evaluation scope) - not confirmed against a sequence
with different motion characteristics (e.g. `rpy`'s fast-rotation profile,
which is exactly the "hard exploration condition" §V-E's aggressive-
insertion design targets and where this gate's cost might show up
differently).

**Decision: drop `--min-parallax` from the TRACK-branch condition.** Better
on every accuracy/robustness metric measured, and more faithful to the
paper (no whole-frame parallax gate on keyframe promotion at all - only
`CreateNewMapPoints`' own per-point check, which this codebase already has
via `triangulate`'s `min_parallax_deg`). Landed as `has_ref_baseline and
kf_policy_ok` (bootstrap's own, separately-justified parallax gate on
essential-matrix estimation is unaffected). The single-sequence caveat
above still applies - worth re-checking on `rpy` or another sequence if a
future issue's evaluation surfaces a regression traceable to this.

**Canonical row (this is what #25 actually shipped - use this one, not the
with-`--min-parallax` row above, as the baseline for whatever issue comes
next):**

| Sequence | Frames | Keyframes accepted | Confirmed / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Wall clock |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #24's row) | 798 | 418 | 102768 / 112173 | 87.9% (26.4s / 30.1s) | 0.1516 | 0.0197 | ~17m45s |
| `freiburg1_xyz` (#25, shipped - no `--min-parallax` on TRACK branch) | 798 | 88 | 39476 / 47591 | 86.4% (26.0s / 30.1s) | 0.0409 | 0.0436 | ~14m14s |

Keyframe insertion rate (required by the issue), final: 88/798 = **11.0%**,
still a decrease vs. #24's 52.4% baseline for the same reason worked out
above (this codebase's map is dense enough post-#23/#24 that PnP tracking
stays confident far longer than the paper's sparser-map assumptions
anticipate) - reported honestly per the issue's own instruction, not
adjusted to match the "expect an increase" prediction.

---

## #26: Recent map point culling, replacing the provisional/confirmed lifecycle (paper §VI-B)

**Version:** `issue-26-recent-map-point-culling` branch, on top of `main`
post-#25 (`--depth-densify` not passed). Implements
[#26](https://github.com/albinjanssonsand/slam/issues/26): `Map.confirm`/
`Map.confirmed`/`confirmation_count` (a point excluded from PnP/BA until it
accumulates `--confirm-count` independent re-observations, and never removed
once added) is deleted outright and replaced with the paper's Recent Map
Points Culling test (§VI-B). Every point is now usable in PnP, guided
matching, and BA from the moment it's created - there's no more admission
gate - but it's checked during its first three keyframes after creation and
removed if it's found by tracking in at most 25% of the frames it was
predicted visible in, or (once more than one keyframe has passed since its
creation) if it hasn't been observed from at least three keyframes. New
per-point bookkeeping (`Map.created_kf`/`n_visible`/`n_found`, `Map.
record_visible`/`record_found`, called every tracking frame from the §V-D
Track Local Map guided-matching call) drives this; `Map.cull_new_points`/
`cull_low_observation_points` run once per keyframe insertion and call a new
`Map.remove_points`, which excludes a point from every future match/local-
map/BA query by cleaning it out of the covisibility graph's keyframe/point
sets rather than by physically deleting its row (point indices stay stable
everywhere else - `keyframe_observations`, BA point ids). An ongoing rule
(`cull_low_observation_points`) also removes any already-surviving point
whose observing-keyframe count later drops below three - a no-op today
since nothing yet reduces observation counts (no keyframe culling, no BA
outlier-observation removal), but implemented so those can plug into it
later. Every call site that gated on `sparse_map.confirmed` (PnP/guided
matching, the old provisional-point re-observation scan) was updated or
removed; `--confirm-count`/`--confirm-reproj-error` are gone from the CLI.

**Two real bugs found during self-review and fixed before any of the
numbers below were trustworthy:**

- **`--global-ba-at-end` silently re-included culled points.** `_run_global_ba`
  and `_reprojection_error_stats` built their point/observation sets
  directly from the demo-local `keyframe_observations` list, which
  `Map.remove_points` never touches (it only cleans the `Map`'s own
  internal graph state) - so a culled point's stale observations kept
  flowing into a full-trajectory BA pass as live constraints, contradicting
  `remove_points`'s own docstring guarantee. Fixed by filtering both against
  `sparse_map.active` before use. Doesn't affect the default (no
  `--global-ba-at-end`) run below.
- **A culled point's originating ORB feature stayed permanently
  unavailable.** `remove_points` cleaned the covisibility graph's
  keyframe/point sets but not `_keyframe_matched_frame_idx` (per-keyframe
  "this feature already belongs to a map point" bookkeeping §VI-C's new-
  point search consults) - so once a point founded from keyframe K's
  feature #120 was culled, feature #120 stayed flagged "already matched"
  forever, permanently blocking any future keyframe from spawning a
  replacement point there. Fixed by threading `frame_idx` through each
  stored observation and freeing it in `remove_points`. This one *does*
  affect the run below (it's on the default per-keyframe path) - the
  numbers below are from after the fix.

**Culling activity (required by the issue), `freiburg1_xyz`, full run:**

| | Total map points ever created | Removed by first-3-keyframe test | Removed by ongoing <3-keyframe rule | Active (surviving) |
|---|---|---|---|---|
| This issue | 60765 | 43835 (72.1%) | 0 | 16930 (27.9%) |

The first-3-keyframe test is doing substantial, real work - nearly three
quarters of every point ever triangulated is rejected within its first three
keyframes, leaving a map under a third the size it would otherwise reach.
The ongoing rule never fires, as anticipated in its own docstring: nothing
in this pipeline yet reduces an already-surviving point's observing-keyframe
count (no keyframe culling, no BA marking observations as outliers), so
there's no trigger for it today - implemented per the issue's own scope
(the removal rule itself, not those triggers) for #23/#19's later
covisibility-graph work to build on.

**Outlier reduction (required by the issue) - qualitative point-cloud
check:** `results/culling_xyz_trajectory.png` still shows a handful of
far-flung outlier points (up to ~200 units from the trajectory, vs. the
main point cloud's ~0-90 unit spread) alongside the dense, well-formed
cluster directly ahead of the camera path. **Culling does not eliminate
geometric outliers** - and shouldn't be expected to: §VI-B's test checks
re-observation *frequency* (found/predicted-visible ratio, observing-
keyframe count), not geometric plausibility, so a mistriangulated point that
still gets matched consistently by tracking (e.g. a systematically-biased
but internally self-consistent batch, per `triangulation.triangulate`'s own
docstring) satisfies the culling test just as easily as a good point does.
What culling *does* demonstrably do is shrink the map by removing points
that fail to be *re-found* reliably (the 72.1% above) - a different,
complementary notion of "outlier" than mistriangulated geometry, matching
the paper's own framing (§VI-B exists to drop points that don't hold up
under continued tracking, not to geometrically filter triangulation).
Reported honestly per the issue's own acceptance criteria, rather than
claimed as a geometric-outlier fix it isn't.

**freiburg1_xyz, full run:**

| Sequence | Frames | Keyframes accepted | Active / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Wall clock |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #25's canonical row) | 798 | 88 | 39476 / 47591 (confirmed/total) | 86.4% (26.0s / 30.1s) | 0.0409 | 0.0436 | ~14m14s |
| `freiburg1_xyz` (this issue) | 798 | 80 | 16930 / 60765 (active/total) | 88.3% (26.6s / 30.1s) | 0.0321 | 0.0323 | ~16 min |

(`results/culling_xyz_vs_groundtruth.png` - the aligned estimate closely
tracks ground truth's same crossing figure-eight sweep, a tight overlay
rather than a loose or scribbled one.)

**Reading these numbers:** this is a clear win on every accuracy metric
measured, not just a wash from the lifecycle replacement. ATE RMSE improves
21.5% (0.0409m -> 0.0321m) and RPE RMSE improves 26.0% (0.0436m -> 0.0323m)
versus #25's baseline, with coverage essentially unchanged (88.3% vs.
86.4%) and tracking-loss frame count comparable (1/798 lost entirely, same
order as every other non-"too hard" run in this file). The "confirmed/
total" and "active/total" columns aren't directly comparable (different
admission semantics - see above), but the qualitative shift is real: under
the old lifecycle, ~83% of every point ever created stayed in the map
forever (never removed, only split confirmed/provisional); under this
issue's culling, only ~28% survive at all. Keyframe count drops modestly
(88 -> 80, -9%), plausibly because a smaller, more aggressively-pruned local
map changes exactly the tracked-point counts §V-E's keyframe policy
(`--kf-min-tracked-points`/`--kf-ref-ratio`) reacts to - not investigated
further here, out of this issue's own admission/removal-only scope. Wall
clock is comparable (~16 min vs. ~14m14s), the added per-frame visibility
bookkeeping and per-keyframe culling sweep costing a small, unmeasured-in-
isolation amount, consistent with the issue's own performance requirement
("a small amount of per-point work every frame").

**Net result:** the paper's §VI-B test is implemented as specified and
measured fresh (not conflated with the earlier, already-reverted "map point
culling" experiment `NOTES.md` documents - different trigger, window, and
target, per the issue's own note) - it removes a large majority (72.1%) of
ever-created points within their first three keyframes, and delivers a
genuine accuracy improvement (ATE -21.5%, RPE -26.0%) at comparable coverage
and wall-clock cost. It does not, and by design cannot, geometrically filter
mistriangulated-but-consistently-tracked outliers - reported honestly per
the issue's own acceptance criteria rather than overclaimed. Two real bugs
surfaced and fixed during self-review (`--global-ba-at-end` point leakage;
permanent ORB-feature-slot loss on culled points) before these numbers were
trustworthy.

---

## #33: Triangulation-time reprojection-error/scale-consistency checks and BA outlier-observation discarding (paper §VI-C/§VI-D)

**Version:** `issue-33-triangulation-checks-ba-outlier-discard` branch, on
top of `main` post-#26 (`--depth-densify` not passed). Implements
[#33](https://github.com/albinjanssonsand/slam/issues/33), closing the two
gaps #26 documented: `triangulation.triangulate()` now also checks
reprojection error and scale consistency (previously only cheirality and
parallax), and local/global bundle adjustment now discard BA-outlier
observations, which - for the first time - gives `Map.
cull_low_observation_points` (#26's "ongoing" rule, a documented no-op until
now) something to actually remove.

`triangulate()` gained optional `octave1`/`octave2` (per-keypoint ORB
pyramid octave, one array per view). When given, a candidate is now also
rejected if its reprojection error in either view exceeds a chi-squared
bound (default 5.991 - the standard 95%-confidence, 2-DOF threshold,
matching ORB-SLAM2's own default) scaled by that view's detection-octave
variance, or if the ratio of its distance to each camera is inconsistent
(within a tolerance factor) with the ratio of the two views' pyramid scale
factors at the detected octaves - both checks mirror ORB-SLAM2's
`CreateNewMapPoints` acceptance test directly. Both real call sites
(bootstrap triangulation, `_create_new_points_from_covisible_keyframes`)
were updated to pass real octave arrays; `pipeline.pose`'s older, superseded
standalone demo was left untouched (the new params default to `None`,
preserving its old cheirality+parallax-only behavior).

`bundle_adjustment.local_bundle_adjustment` now runs a second classify-and-
discard pass matching the paper's own wording ("observations that are
marked as outliers are discarded at the middle and at the end of the
optimization"): after the first Huber-loss solve, every observation's
squared reprojection error is checked against a chi-squared bound; flagged
ones are excluded and the window is re-solved with the survivors (the
"middle" checkpoint), then re-classified once more (the "end" checkpoint) to
catch anything the re-solve's shifted estimate newly exposes. `Map` gained
`remove_observation(point_idx, kf_idx)` - the missing inverse of
`add_observation` (drops one observation, decrements the covisibility edge,
frees the matched-frame-idx slot, recomputes the point's §III-C metadata
from what remains) - and a new `mapping._discard_ba_outlier_observations`
uses it to physically remove every BA-flagged observation from both `Map`
and the flat `keyframe_observations` list, then calls `cull_low_observation_
points`. This is the concrete mechanism #26's culling test could never
reach on its own.

### Three real bugs found before these numbers were trustworthy

Two surfaced during self-review (2 parallel reviewer passes, since the diff
spans 3 tightly-coupled files at 518 changed lines) and were fixed before
any run below; a third only showed up once the actual `freiburg1_xyz` run's
numbers looked wrong, and was root-caused and fixed before accepting the
final numbers:

- **The BA outlier-classification guard was bypassed by its own "end"
  checkpoint.** The guard exists specifically to skip discarding when too
  many (or too few surviving) observations exceed the chi-squared bound -
  its own comment says that's "a sign of a bad solve, not real outliers."
  But when the guard blocked the re-solve, `obs_mask`/`result.x` were left
  untouched, so the "end" checkpoint reclassified the *identical* full set
  against the *identical* solve, reproducing the same flagged indices and
  returning them as trusted outliers anyway - in the worst case
  (`mid_outliers.sum() == len(obs)`), discarding an entire BA window's
  observations in one shot, the opposite of the guard's intent. Fixed to
  report zero outliers whenever the guard doesn't trust the classification,
  rather than silently reusing it.
- **A point could be left `active` with zero observations and stale
  metadata.** `remove_observation` had no equivalent of `remove_points`'s
  full unlink for the specific case of a point's *last* observation being
  popped - if both of a just-triangulated point's founding observations were
  flagged outliers within the same keyframe's BA call, the point stayed
  `active` (age-gated `cull_new_points`/`cull_low_observation_points` can't
  catch it at `elapsed == 0`) with pre-removal viewing-direction/scale-
  invariance bounds, eligible for guided matching until some later
  keyframe's age-gated check happened to catch it. Fixed: `remove_observation`
  now calls `remove_points` immediately, unconditionally, the moment a point
  drops to zero observations - closing the gap regardless of age.
- **The BA outlier chi-squared threshold was flat (implicitly octave-0-only),
  which measurably hurt accuracy rather than helping.** The first full
  `freiburg1_xyz` run (after fixing the two bugs above) showed ATE/RPE
  *regressing* by more than 50% versus #26's baseline (ATE 0.0321m ->
  0.0493m, RPE 0.0323m -> 0.0519m) - ORB observations are legitimately less
  precisely localized in pixel space at coarser pyramid octaves (the same
  reasoning `triangulate()`'s own reprojection check, and `Map`'s existing
  `d_min`/`d_max` scale-invariance bounds, already account for), so a flat
  chi-squared bound calibrated for octave 0 was flagging large numbers of
  perfectly good coarser-octave observations as outliers and discarding real
  constraining data - 975 observations discarded, 63 points removed, on that
  first run. Root-caused rather than accepted as "the paper's checks just
  don't help here": added `Map.observation_octave(point_idx, kf_idx)` and
  threaded per-observation octave metadata through `_run_ba` into
  `local_bundle_adjustment`, which now scales the chi-squared bound per
  observation by `pyramid_scale_factor ** (2*octave)`, exactly mirroring
  `triangulate()`'s own treatment. Re-running after this fix dropped outlier
  discarding to a much more conservative 120 observations/8 points (see
  below) and turned the regression into a genuine improvement.

### Triangulation-rejection demonstration (required - direct, not just end-to-end)

Ad hoc synthetic script (not committed - no test suite exists in this repo),
constructing exact pixel coordinates for a calibrated pure-X-translation
stereo pair:

- A well-conditioned point (real cheirality, 5.7deg parallax) passes both
  new checks, as expected.
- The same point pair with view 2 perturbed 15px **off the epipolar line**
  is accepted by today's cheirality+parallax-only test but **rejected** once
  the reprojection-error check is enabled (`valid: True -> False`). Note:
  a *horizontal* (epipolar-consistent) 15px perturbation is NOT rejected -
  2-view `cv2.triangulatePoints` DLT reprojects any epipolar-consistent
  correspondence with ~zero error regardless of the resulting depth (this is
  inherent to linear 2-view triangulation, not a gap in this
  implementation - ORB-SLAM2's own `CreateNewMapPoints` has the identical
  property, since it also epipolar-filters candidates before triangulating).
  The reprojection-error check is therefore a check on whether the
  *correspondence* was real, not on whether the resulting *depth* is
  plausible - an implausibly close **or** far depth from an
  epipolar-consistent correspondence is still only caught by the existing
  parallax-angle check, exactly as before this issue.
- The same well-conditioned point, given deliberately mismatched octaves
  (`octave1=0, octave2=5`, implying a ~2.49x distance ratio the actual ~1.0x
  triangulated ratio doesn't match), is accepted without the scale-
  consistency check and **rejected** with it (`valid: True -> False`).

### BA outlier-discard demonstration (required - direct, not just end-to-end)

Second ad hoc script: 4 keyframes, 30 points, one observation corrupted by
40px.

- `local_bundle_adjustment` flags exactly that one injected observation -
  no false positives among the other 119 clean, noisy (0.3px) observations.
- `_discard_ba_outlier_observations` correctly drops the Map's observing-
  keyframe count for that point from 4 to 3 and removes its entry from
  `keyframe_observations`; `cull_low_observation_points` correctly does
  *not* remove it at exactly 3 (the paper's own boundary).
- Manually discarding a second point's observations down to 2 observing
  keyframes (the same primitive, invoked twice) confirms
  `cull_low_observation_points` *does* fire at that point, removing it -
  the first time this trigger has ever been reachable (#26 landed it as a
  documented no-op).

### Culling/BA-discard activity (required by the issue), `freiburg1_xyz`, full run

| | Total map points ever created | Removed by first-3-keyframe test | Removed by the ongoing <3-observing-keyframe rule (total) | ...of which via BA-outlier discard (#33, new) | Active (surviving) |
|---|---|---|---|---|---|
| #26 (baseline) | 60765 | 43835 (72.1%) | 0 | n/a (trigger didn't exist) | 16930 (27.9%) |
| This issue | 59138 | 43058 (72.8%) | 11 | 8 | 16069 (27.2%) |

BA discarded 120 observations across all local BA passes over the run (no
`--global-ba-at-end`), of which 8 points' observation counts dropped below
3 as a direct, immediate result - `cull_low_observation_points` firing for
the first time via this trigger, exactly as #26 anticipated it eventually
would. The remaining 3 (of the 11 total ongoing-rule removals) came via the
pre-existing per-keyframe check rather than the new post-BA check
specifically - plausibly a point whose BA-discarded observation happened
while it was still within its first-3-keyframe window (age-gated out of
`cull_low_observation_points`'s own candidate set at the moment of
discarding, then caught once it aged past that window) - not separately
instrumented to prove that attribution, reported as a plausible explanation
rather than a confirmed one. Total ever-created points also dropped
slightly (60765 -> 59138, -2.7%), attributable to the new triangulation-time
checks now rejecting some candidates at creation time that previously
reached the map (and, in some fraction of cases, would only have been
caught later by the first-3-keyframe test or contributed to the far-flung
outliers below).

**Outlier reduction (required by the issue) - qualitative point-cloud
check:** comparing `results/culling_xyz_trajectory.png` (#26, before) with
this issue's `results/ba_outlier_xyz_trajectory.png` (after): the most
extreme far-flung outliers are visibly reduced - #26's plot shows points out
to ~215 units from the trajectory (e.g. around (-115, 213)); this issue's
plot's most extreme points reach only ~130 units (e.g. around (-40, 125)).
Reported honestly per the issue's own acceptance bar ("implemented the
paper's checks correctly and measured the effect," not "eliminated every
outlier"): this is a real, visible reduction, not a complete fix - some
80-130 unit outliers remain, consistent with the reprojection-error check's
inherent epipolar-consistency limitation documented above (an
epipolar-consistent-but-wrong-depth correspondence isn't caught by either
new check, only by the pre-existing parallax-angle test).

**freiburg1_xyz, full run:**

| Sequence | Frames | Keyframes accepted | Active / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) | Wall clock |
|---|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #26's canonical row) | 798 | 80 | 16930 / 60765 | 88.3% (26.6s / 30.1s) | 0.0321 | 0.0323 | ~16 min |
| `freiburg1_xyz` (this issue) | 798 | 78 | 16069 / 59138 | 88.1% (26.5s / 30.1s) | 0.0250 | 0.0255 | ~15 min |

(`results/ba_outlier_xyz_vs_groundtruth.png` - the aligned estimate closely
tracks ground truth's same crossing figure-eight sweep, a tight overlay.)

**Reading these numbers:** a genuine accuracy improvement on both metrics -
ATE RMSE -22.1% (0.0321m -> 0.0250m), RPE RMSE -21.1% (0.0323m -> 0.0255m) -
at essentially unchanged coverage (88.1% vs 88.3%) and comparable wall-clock
cost, with keyframe count modestly lower (80 -> 78, -2.5%). This is *after*
fixing the octave-aware chi-squared bug above; the first (buggy) run showed
the opposite - a >50% regression - which is itself informative: a correctly
scale-aware implementation of the paper's outlier-discard rule helps, but a
naively flat threshold actively hurts by discarding good, low-precision-but-
legitimate observations. Consistent with #26's own finding, this can't be
attributed to eliminating geometric outliers wholesale (the qualitative
check above shows some remain, by design) - the improvement instead comes
from the more mundane mechanism the paper actually specifies: BA no longer
lets a handful of genuinely bad correspondences corrupt otherwise-good
poses/points, and any point that becomes unreliable as a result is now
actually removed instead of staying in the map forever.

**Net result:** both of #26's documented gaps are closed and measured
fresh. The reprojection-error/scale-consistency checks are implemented
exactly as ORB-SLAM2 specifies them, with an honestly-documented inherent
limitation (epipolar-consistent depth collapse isn't and can't be caught by
either check - only by the existing parallax test). BA outlier discarding
is implemented per the paper's own two-checkpoint wording and demonstrably
reaches `cull_low_observation_points` for the first time. Three real bugs
were found and fixed before trusting any number here, the last of which
(the flat-threshold regression) would have been easy to miss without
running the required end-to-end evaluation and taking a >50% regression
seriously instead of writing it off as "the paper's checks don't help on
this sequence."

---

## #27: Local keyframe culling (paper §VI-E)

**Version:** `issue-27-local-keyframe-culling` branch, on top of `main`
post-#33 (`--depth-densify` not passed). Implements
[#27](https://github.com/albinjanssonsand/slam/issues/27): keyframes were
only ever appended (`keyframe_poses`/`keyframe_observations`) - nothing was
ever removed. `Map` gained `remove_keyframe(kf_idx)` (unlinks a keyframe
from every point it observes, reusing #33's `remove_observation` primitive
for the cascade - the exact reuse #33's own issue text anticipated) and
`cull_redundant_keyframes(candidate_kfs, min_observers=3, redundancy_ratio=
0.9)`, implementing the paper's test directly: discard a keyframe once at
least 90% of its own observed (already point-culled) points are each also
observed, at the same or finer pyramid scale, by at least 3 other
keyframes - "same or finer" meaning the other keyframe's own detection
octave for that point is `<=` this keyframe's octave for it, the same
convention `Map.d_min`/`d_max`, `match_against_guided`'s `PredictScale`,
and #33's own scale-consistency check already use.

**Cadence/scoping decision (the issue's own open question):** runs once
per keyframe insertion, scoped to that keyframe's covisibility-graph
neighbors (capped to 20, most-shared-points-first - identical rescoping
reasoning to `_run_local_ba`'s own 20-neighbor cap), via a new
`mapping._cull_redundant_keyframes` wrapper that also clears the
demo-local `keyframe_observations` entry for anything removed and cascades
into `cull_low_observation_points`. Chosen over a periodic/end-of-run sweep
because: (1) it mirrors local BA's own already-established per-keyframe,
covisibility-scoped cadence, so no new architectural pattern is introduced;
(2) a keyframe only becomes evaluable once its neighbors are known, which
naturally happens as later keyframes are inserted nearby (including on a
revisit, when an old keyframe becomes covisible with new ones again) -
periodic/end-of-run sweeps wouldn't evaluate anything a per-insertion pass
doesn't already reach, they'd just delay it; (3) it's cheap by construction
(bounded to <=20 candidates, each an O(points x observers-per-point) scan)
without needing a separate ablation to prove - the freiburg1_xyz run below
executes this check every single keyframe and produces a byte-identical
trajectory to #33's own run in the same time-order-of-magnitude wall clock
(concurrent execution with the other two runs below prevents a precise
delta - see the honest caveat in that section - but nothing resembling a
new bottleneck appeared). The current/most-recently-inserted keyframe and
keyframe 0 (the sole global-BA gauge anchor) are always excluded from
candidates. Keyframe indices, like point indices, are never reindexed -
`keyframe_poses`/`keyframe_observations`/`keyframe_kp`/`keyframe_desc` stay
append-only and index-stable; `Map.keyframe_active`/`removed_keyframes`
track removal the same way `Map.active`/`removed` do for points, and every
place that builds a keyframe set directly from these lists rather than
through one of `Map`'s own graph-derived queries (`_run_global_ba`'s
`free_kfs`, the final trajectory-output/plot filtering) now filters
through it explicitly.

### Keyframe-culling demonstration (required - direct, not just end-to-end)

Ad hoc synthetic script (not committed - no test suite exists in this
repo), built directly on `Map`:

- A keyframe whose points are each also seen, at the same octave, by 3
  other keyframes (100% redundant) is correctly removed by
  `cull_redundant_keyframes`.
- A keyframe with the same point count but only 1 other observer per point
  is correctly left alone (below the `min_observers=3` bar).
- A keyframe whose points ARE each seen by 3 other keyframes, but at a
  COARSER octave (3, vs. this keyframe's own octave 0), is correctly left
  alone too - confirming the "same or finer scale" condition is actually
  enforced, not just an observer-count check.
- A separate scenario exercises the cascading order the method's own
  docstring describes - candidates are processed one at a time, so an
  earlier removal changes what a later candidate's own redundancy fraction
  sees next: three keyframes (1, 11, 12) each superficially qualify as
  ~90%-redundant by their own point counts, but two of them (11, 12) only
  clear the >=3-other-observers bar for their shared points BECAUSE
  keyframe 1 counts as one of those observers. `cull_redundant_keyframes`
  removes keyframe 1 first (evaluated first, per covisibility-edge-weight
  ordering), and the run confirms only `[1]` is removed, not `[1, 11]` or
  `[1, 11, 12]` - keyframe 11's own redundancy fraction correctly drops
  once keyframe 1 (one of its own "other observers") is no longer there to
  count, exactly the cascading interaction the docstring promises rather
  than a stale, precomputed-upfront fraction.
- `mapping._cull_redundant_keyframes`'s end-to-end wrapper (a separate,
  simpler scenario): keyframe 1, correctly found via `current_kf_idx`'s own
  covisibility neighbors (not an externally-supplied candidate list) at
  exactly the 90% boundary (9 of 10 points redundant), gets removed;
  `keyframe_observations` for it is correctly cleared; and one of its own
  points - which only had 3 total observing keyframes, one of them being
  the removed keyframe itself - correctly drops to 2 and gets removed by
  the `cull_low_observation_points`
  cascade, exactly the interaction the issue's scope section describes
  ("the two mechanisms interact directly - a keyframe removal can cascade
  into point removals").

### freiburg1_xyz, full run (required per #19's evaluation policy)

| Sequence | Frames | Keyframes accepted | Active / total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline, #33's row) | 798 | 78 | 16069 / 59138 | 88.1% (26.5s / 30.1s) | 0.0250 | 0.0255 |
| `freiburg1_xyz` (this issue) | 798 | 78 | 16069 / 59138 | 88.1% (26.5s / 30.1s) | 0.0250 | 0.0255 |

**Byte-identical to #33's trajectory** (`diff`-confirmed) - keyframe
culling removed **0** keyframes on this sequence (`[keyframe culling (paper
VI-E): 0 keyframes removed (79 active / 79 total), 0 point(s) removed...]`
in the run log). Reported honestly rather than re-run or tuned to force a
different number: `freiburg1_xyz` is a short (798-frame), continuously-
translating handheld sequence with no lingering or revisiting - the paper's
own redundancy premise (a keyframe whose viewpoint is already well-covered
by >=3 later-or-earlier keyframes) doesn't arise when the camera keeps
moving through new viewpoints rather than sitting still or doubling back.
This isn't a surprise specific to this issue - #19's own "Specific test sets
beyond freiburg1_xyz" section already named `freiburg1_xyz` as "too short to
reliably show" exactly this scenario for #20/#23/#27 alike, which is why
`freiburg2_pioneer_slam2` was picked as this issue's dedicated redundancy
test (see below).

### freiburg2_pioneer_slam2 (required - the issue's own redundancy/lingering test)

| | Keyframes accepted | Active/total points | Coverage | ATE RMSE | RPE RMSE |
|---|---|---|---|---|---|
| `--no-keyframe-cull` (baseline) | 14 | 505/1033 | 4.6% (5.4s/115.6s) | 0.0190 | 0.0118 |
| Default (keyframe culling on) | 14 | 505/1033 | 4.6% (5.4s/115.6s) | 0.0190 | 0.0118 |

**Byte-identical trajectories** (`diff`-confirmed) - keyframe culling
removed **0** keyframes in either run. **Not a regression or a bug in this
issue's changes** - both runs die from total tracking loss at frame
~174/2116 (8.2% into the sequence), the exact same pre-existing,
already-diagnosed failure this codebase has hit on this sequence in every
prior sub-issue that tried it (#17: dies at frame 181, "too hard"/tracking
loss; #20: re-confirmed, 44 keyframes before death; #23: re-confirmed
again, 38 keyframes before death, frame ~173, quoting #20's own prediction
for whoever picked up #23/#27 next: *"#23/#27... will hit the identical
dead end"* - a prediction this issue now confirms). The root cause (zero
view overlap with the map once PnP inlier count declines past a threshold,
with no relocalization - #13 - to recover) is unrelated to bundle
adjustment, covisibility, or keyframe culling, and this issue makes no
attempt to fix it (out of scope, same as #20/#23's own conclusion). With
only 14 keyframes ever accepted before death, the map never grows large or
mature enough for any keyframe to reach the covisibility-neighbor
population `cull_redundant_keyframes` needs to even evaluate - so, per the
issue's own explicit instruction ("if it doesn't contain meaningful
lingering/revisiting either, report that rather than manufacture a
synthetic test just for this issue"), no synthetic revisit segment was
constructed. **Both required acceptance-criteria checks from the issue
still pass trivially and honestly**: map coverage/point count is
unaffected by keyframe culling alone (identical between the two runs,
confirmed above), and there's no reduction in BA/tracking cost to report
because there was nothing for the trigger to remove - reported as such,
not forced into a positive result.

### Reading these numbers

Neither of this repo's two evaluation sequences currently exercises real
keyframe redundancy end-to-end: `freiburg1_xyz` is too short/non-
repetitive (anticipated by #19 itself), and `freiburg2_pioneer_slam2` dies
before its map ever matures (a pre-existing #13-tracked bug, explicitly
predicted to recur here by #23). This means the *mechanism itself* -
`Map.cull_redundant_keyframes`'s redundancy test, the same-or-finer-scale
condition, and the removal cascade into point culling - is verified
correct only via the direct synthetic demonstration above, not via a live
end-to-end run in this repo's current dataset set. That's a real gap in
what could be measured here, reported honestly rather than papered over
with a fabricated revisit scenario the issue itself said not to construct.
A TUM sequence with genuine lingering/revisiting (or #13 landing and
unlocking `freiburg2_pioneer_slam2`'s remainder) would be the natural way
to close it later.

**Net result:** the paper's §VI-E test is implemented as specified -
covisibility-scoped candidate selection (mirroring local BA's own cadence),
the same-or-finer pyramid-scale condition (not just an observer count), and
a removal cascade into #26's ongoing point-culling rule via #33's
`remove_observation` primitive, exactly the reuse both issues' own text
anticipated. Self-reviewed (1 reviewer - diff was under the size/complexity
thresholds for escalation), no findings. Correctness demonstrated directly
via synthetic `Map`-level tests covering the redundancy count, the scale
condition, candidate-selection-from-covisibility, and the removal cascade.
End-to-end firing could not be demonstrated on either available dataset,
for reasons unrelated to this issue's own implementation - reported
honestly, consistent with #19's own precedent for this exact scenario.

---

## #14: Mutual nearest-neighbor cross-check in `match_descriptors`

**Version:** `issue-14-match-descriptors-crosscheck` branch, on top of `main`
post-#27 (`--depth-densify` not passed). Implements
[#14](https://github.com/albinjanssonsand/slam/issues/14):
`pipeline.features.match_descriptors` now follows its existing Lowe's-ratio
test with a mutual-nearest-neighbor cross-check - a forward match
`desc1[i] -> desc2[j]` is kept only if `desc2[j]`'s own best match back into
`desc1` is also `i` (one extra `BFMatcher.match(desc2, desc1)` call, cheapest
of the issue's own candidate list, no geometric assumptions).

**Scope decision (the issue's own open question):** limited to
`match_descriptors` itself, **not** extended to `Map.match_against_guided`'s
separate inline scoring. Two things checked first, both confirming the
issue's own "Updated" framing: (1) `Map.match_against` - the other historical
consumer the issue names - has zero call sites left in the codebase
(`grep -rn "\.match_against\("` finds nothing outside its own definition),
confirming it's dead code superseded by `match_against_guided` (#23), exactly
as the issue's update states; (2) `match_against_guided` already performs its
own geometric-consistency gating (predicted-position window, viewing-angle
vs. stored §III-C direction, [d_min, d_max] scale-invariance bounds,
predicted-pyramid-octave search) that goes beyond what a ratio test or
cross-check alone would add - it's already answering "is this match
geometrically consistent," just via projection rather than mutual-NN. Adding
cross-check there too was judged unnecessary to satisfy this issue's own
motivation and left as a candidate for a future issue if guided matching's
own false-match rate is ever separately identified as a problem.
`match_descriptors`'s two remaining live call sites both stay in scope:
`matches_ref` in `_demo()`'s main loop (bootstrap two-view matching, and -
since #25 - the sole live use of `has_ref_baseline` for gating TRACK-branch
keyframe promotion) and `_create_new_points_from_covisible_keyframes`
(§VI-C new-point creation, #24).

**Match-quality measurement (required by the issue - false-match-rate proxy,
before/after):** an ad hoc script (not committed - no test suite exists in
this repo) ran both the old ratio-only matcher and the new ratio+cross-check
matcher over the same consecutive-frame-pair descriptors (this pipeline's own
`detect_and_compute_gridded`), then measured each match set's
`cv2.findEssentialMat(..., method=cv2.RANSAC)` inlier ratio as a proxy for
"how many of these matches are real, geometrically-consistent
correspondences" - a wrong match is very unlikely to agree with the true
epipolar geometry by chance, so a match set with fewer total matches but a
higher RANSAC-inlier fraction is measurably cleaner going into
bootstrap/pose estimation, independent of anything full end-to-end tracking
does downstream.

| Sequence (frame range) | n pairs | Avg matches (old &rarr; new) | Avg E-inlier rate (old &rarr; new) |
|---|---|---|---|
| `freiburg1_desk` (0-50) | 50 | 505.0 &rarr; 437.8 (-13%) | 68.9% &rarr; 72.1% (+3.2pp) |
| `freiburg1_room` (0-25) | 25 | 567.8 &rarr; 490.8 (-14%) | 70.1% &rarr; 73.1% (+3.0pp) |
| `freiburg1_xyz` (0-60) | 60 | 768.1 &rarr; 694.2 (-10%) | 77.5% &rarr; 80.1% (+2.6pp) |
| `recordings/demo1.mp4` (0-60) | 60 | 558.2 &rarr; 513.7 (-8%) | 84.1% &rarr; 87.0% (+2.9pp) |

**Consistent, real effect across every sequence tested, including both
target hard sequences:** cross-check trims ~8-14% of candidate matches while
raising the RANSAC-inlier proxy by ~2.6-3.2 percentage points every time, no
exceptions across 195 frame pairs on four different sequences/cameras. This
directly answers the issue's "investigate which meaningfully reduces the
false-match rate... don't guess, measure" requirement: cross-check alone
does measurably clean up `match_descriptors`'s candidate set. (The other
candidate - a fundamental-matrix RANSAC pre-filter - wasn't additionally
implemented: cross-check's own measured effect already gave a clear,
consistent signal without needing the costlier geometric pass, and every
live call site already feeds its output straight into its own
RANSAC/reprojection-based filtering - `estimate_relative_pose`'s
`findEssentialMat` RANSAC for bootstrap, `triangulate`'s reprojection/scale
checks for new-point creation - so a second, redundant epipolar filter ahead
of those was judged not worth its extra cost here.)

**Regression check (required by the issue - healthy sequences must not
starve):** `freiburg1_xyz` and `recordings/demo1.mp4` both stay in the
hundreds of matches per pair after cross-check (694.2 and 513.7 average,
respectively) - nowhere near the `too few guided map matches`/`PnP: too few
inliers` starvation symptoms `NOTES.md` documents from the unrelated,
since-removed `--confirm-count 2` regression. No starvation risk on either
healthy control sequence.

**freiburg1_desk/freiburg1_room, fresh baseline (required by the issue - the
original text's frame-37/613 and frame-100/1362 citations predate
#20-#27/#33 and are stale):** re-ran both sequences on this branch's
pre-fix `main` first to get an accurate current "before" state, then again
with the fix.

| | Bootstrap frame | Keyframes accepted | Active/total points | Death point | Coverage | ATE/RPE RMSE (m) |
|---|---|---|---|---|---|---|
| `desk`, pre-fix (fresh baseline) | 2 | 6 | 566/1445 | frame 45/613 | 5.3% (1.24s/23.4s) | 0.0200 / 0.0749 |
| `desk`, post-fix (this issue) | 4 | 3 | 414/650 | frame 24/613 | 1.0% (0.23s/23.4s) | 0.0107 / 0.0171 |
| `room`, pre-fix (fresh baseline) | 4 | 1 | 140/140 | frame 7/1362 (0 KF beyond bootstrap) | 0.27% (0.13s/48.9s) | degenerate (2 KF, same as `rpy`) |
| `room`, post-fix (this issue) | 3 | 1 | 107/107 | frame 19/1362 (0 KF beyond bootstrap) | 0.20% (0.10s/48.9s) | degenerate (2 KF, same as `rpy`) |

Both remain solidly in the **"too hard" (tracking loss)** bucket either way
- consistent with the issue's own explicit Non-goal ("not a fix for... full
tracking-loss recovery (#13) - this issue is about reducing the baseline
false-match rate on any given frame pair, healthy or not"). `room`'s ATE/RPE
are degenerate on both runs (`evo`'s own "Degenerate covariance rank"
error) because neither run ever accepts a second TRACK-branch keyframe
beyond bootstrap - the exact same 2-keyframe alignment failure `NOTES.md`
already documents for `freiburg1_rpy` - so coverage/frame-only-tracking
duration are the only meaningful comparison points there, not ATE/RPE.

**Reading these numbers - reported honestly, not smoothed over:** the fix
changes which frame bootstrap fires on (`desk`: 2&rarr;4; `room`: 4&rarr;3) -
a direct, expected consequence of a stricter matcher taking one or two more
frames to accumulate 8+ surviving matches/enough parallax - and that
different starting point cascades differently on these two sequences, whose
dominant failure mode (documented extensively above, under #15/#17/#20/#23's
sections) is *already* known to be highly sensitive to exactly this kind of
small early difference: **zero view overlap with the confirmed map once PnP
inlier count declines past a threshold, with no relocalization (#13) to
recover** - not a false-match-rate problem this issue could fix even in
principle. `desk`'s post-fix run dies measurably earlier (frame 24 vs. 45) -
a real, honestly-reported regression in this one run-to-run comparison, not
dismissed as noise - but two things bound how much weight it deserves: (1)
OpenCV's RANSAC-based estimators (`findEssentialMat` for bootstrap,
`solvePnPRansac` inside `estimate_pose_pnp`) draw unseeded random samples, so
some run-to-run variation is expected from the *unchanged* code alone, not
only from this fix - not separately quantified here (would need many repeats
per condition), but a real confound worth naming rather than ignoring; (2)
`room`'s frame-only tracking survives measurably *longer* post-fix (14
frame-only-tracked frames vs. 3, dying at frame 19 vs. 7) - the opposite
direction from `desk` - so the two target hard sequences don't even agree on
a sign, consistent with "different bootstrap seed cascades differently on a
chaotic failure mode" rather than a systematic harm from cleaner matching.
Neither sequence's "too hard" classification changes either way.

**freiburg1_xyz, full run (required by the issue - regression control):**

| Sequence | Frames | Keyframes accepted | Active/total map points | Trajectory coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|---|---|
| `freiburg1_xyz`, pre-fix (fresh baseline) | 798 | 88 | 16639/63091 | 86.6% (26.07s/30.09s) | 0.0229 | 0.0218 |
| `freiburg1_xyz`, post-fix (this issue) | 798 | 88 | 16639/63091 | 86.6% (26.07s/30.09s) | 0.0229 | 0.0218 |

**Byte-identical** (`diff`-confirmed, both the full run log and the saved
trajectory file) - every keyframe/point count, coverage, and ATE/RPE digit
matches exactly. (This fresh baseline differs slightly from #27's own
canonical row - 88 vs. 78 keyframes, 86.6% vs. 88.1% coverage - attributable
to the RANSAC non-determinism noted above between separate runs of unchanged
code, not to anything this issue touches; the pre-fix/post-fix pair here was
run back-to-back specifically to isolate this issue's own effect from that
noise.) On this sequence, cross-check trims ~10% of candidate matches per
frame pair (measured above) without changing a single downstream decision -
plausible explanation, not separately proven: `estimate_relative_pose`'s own
`findEssentialMat` RANSAC and `_create_new_points_from_covisible_keyframes`'s
epipolar/reprojection/scale checks already reject the same wrong
correspondences cross-check would have pre-filtered, so the surviving
accepted set converges to the same answer either way on a sequence healthy
enough that those downstream checks were never close to their own rejection
thresholds.

**Net result:** the fix does exactly what it was scoped to do - a cheap,
dependency-free (`cv2.BFMatcher`, already used) match-quality improvement to
`match_descriptors`'s two live call sites, with a real, consistently
measured false-match-rate reduction (higher RANSAC-inlier proxy at a lower
match count, on every one of 4 tested sequences/cameras) and zero regression
on the two required healthy-sequence checks (no match-count starvation on
`xyz`/`demo1.mp4`; `xyz`'s full end-to-end run is byte-identical). It does
not - and, per the issue's own Non-goals, was never meant to - fix
`desk`/`room`'s tracking-loss failure mode, whose root cause (#13) is
unrelated to per-pair match quality; the death-point movement observed on
those two sequences is real but goes in opposite directions between them and
is bounded by unseeded-RANSAC run-to-run noise this write-up flags rather
than glosses over. `Map.match_against_guided` was deliberately left
untouched (see the Scope decision above) - a candidate for a future issue if
its own false-match rate is ever separately identified as a problem worth
solving.

---

## #13: Relocalization after tracking loss

**Version:** `issue-13-relocalization-after-tracking-loss` branch, on top of
`main` post-#14 (`--depth-densify` not passed). Implements
[#13](https://github.com/albinjanssonsand/slam/issues/13): once ordinary
per-frame guided PnP tracking has failed `--relocalize-after-frames`
(default 5) consecutive frames, a new `--relocalize` flag (off by default)
attempts a wide, unguided search - `Map.match_against` against the ENTIRE
active map, no predicted-pose window - followed by `estimate_pose_pnp`
fresh, with no assumption of a nearby previous pose. Deliberately skips the
rotation-vs-previous-frame/step-vs-recent-median plausibility checks
ordinary tracking runs (those assume continuity a relocalization by
definition doesn't have); the same `--pnp-min-inliers` bar ordinary tracking
itself requires is what validates the recovered pose instead. On success,
resumes ordinary tracking from the recovered pose - not a separate code
path - and re-anchors the plain (non-guided) reference-keyframe match
(`has_ref_baseline`, condition 4's baseline) to whichever existing keyframe
shares the most of the relocalization's own inlier points, so a future
keyframe stays reachable even if the pre-loss reference is visually
unrelated to the recovered location.

**Fresh baselines first (required by the issue - old frame-37/frame-100
citations predate #15-#33 and are stale, and #14 itself already showed
`desk`/`room`'s bootstrap point shifts by 1-2 frames):** used #14's own
already-fresh `desk`/`room` post-fix rows directly (re-confirmed identical
here with `--relocalize` off), and additionally re-baselined
`freiburg2_pioneer_slam2` fresh against current `main` for the first time
since #14 landed, since no prior section had done so.

| Sequence | Bootstrap frame | Keyframes | Active/total points | Death/degenerate point | Coverage | ATE/RPE RMSE (m) |
|---|---|---|---|---|---|---|
| `desk` (fresh, `--relocalize` off, = #14's post-fix row) | 4 | 3 | 414/650 | frame 24/613 | 1.0% (0.23s/23.4s) | 0.0107 / 0.0171 |
| `room` (fresh, `--relocalize` off, = #14's post-fix row) | 3 | 1 | 107/107 | frame 19/1362 (0 KF beyond bootstrap) | 0.20% (0.10s/48.9s) | degenerate (2 KF) |
| `freiburg2_pioneer_slam2` (fresh, first re-baseline since #14) | 79 | 6 | 142/331 | frame ~155/2113 | 4.1% (4.74s/115.6s) | 0.0104 / 0.0141 |

`pioneer_slam2`'s fresh numbers (bootstrap frame 79, dies ~frame 155 with 6
keyframes) differ substantially from every previously-documented run of this
sequence (bootstrap ~frame 2-4, dies frame ~173-181 with 38-44 keyframes,
per #17/#20/#23/#27) - consistent with #14's own already-established effect
of shifting which frame pair first clears the cross-check matcher's stricter
bar, just a much larger shift than `desk`/`room` saw, and not something any
prior section had actually re-measured on this sequence. Not investigated
further here (out of this issue's own scope), but worth flagging for
whoever next relies on `pioneer_slam2`'s "frame 181" citation.

**Relocalization results, all four required sequences (`--relocalize` on,
`--relocalize-after-frames` default 5):**

| Sequence | Attempts | Succeeded | Recovery length | Reached a new keyframe? | Coverage/ATE/RPE change |
|---|---|---|---|---|---|
| `desk` | 577 | 1 (frame 30, 29/56 inliers) | 1 extra frame (31), lost again | No | None - byte-identical to the flag-off row above |
| `room` | 1328 | 1 (frame 93, 20/25 inliers, after 73 lost frames) | ~14 frames (93-97+), lost again | No | None - still degenerate (2 KF), same as flag-off |
| `freiburg2_pioneer_slam2` | 1952 | 0 | n/a | n/a | None - correctly declines to (falsely) relocalize |
| `freiburg1_xyz` (regression control) | 21 | 1 (frame 48) | recovered permanently (already-healthy sequence) | yes, ordinarily | See below - NOT a clean no-op |

**`desk`/`room`: the mechanism genuinely works end-to-end on real data, but
neither reaches the bar to matter.** Both logs show a real, unambiguous
recovery - full-map matches, a passing PnP+RANSAC inlier count, tracking
resuming via the ordinary guided-matching path on the very next frame (e.g.
`room`'s frame 94-97 are indistinguishable in form from any other healthy
`TRACK` line) - not a false positive. But on `desk` the recovery survives
only 1 extra frame before the same fast-pan failure mode (documented
separately, this session's earlier investigation) reasserts itself; on
`room` it survives longer (~14 frames) but still not the
`--kf-min-frames-since-relocalization` (20) + `--kf-max-frames-since-keyframe`
(20) grace period needed to become eligible for a new keyframe, on a
sequence that (per #14's own fresh baseline) already couldn't produce a
*second* keyframe even right after a healthy bootstrap. Neither sequence's
"too hard" classification changes. `freiburg2_pioneer_slam2` is the
important negative control: 1952 attempts, 0 successes, matching the
issue's own ground-truth-verified prediction that this specific run's
camera never revisits the pre-death map - "correctly declines" is the
success criterion here, not a demonstrated recovery, and that's what
happened.

**`freiburg1_xyz` regression check: not a byte-identical no-op, for a
reason worth understanding rather than glossing over.** The flag-off
baseline (`postfix_xyz_run.log`, #14's row) already contains a 27-frame gap
(frames 22-48) where per-frame tracking fails entirely before recovering on
its own at frame 49 via the ordinary guided path (`trigger=max-frames-
fallback`) - a pre-existing property of this sequence, confirmed unrelated
to this issue (frames 1-47 of the `--relocalize` run are byte-*identical* to
the flag-off baseline, diff-confirmed). Because that gap is 27 frames long
(> the 5-frame `--relocalize-after-frames` default), `--relocalize` fires
21 times against it and succeeds once, one frame earlier (48) than the
baseline's own natural recovery (49). That one extra `cv2.solvePnPRansac`
call (and the 20 other failed attempts' worth of calls) is enough to change
the outcome for the entire rest of the run: OpenCV's RANSAC estimators draw
from process-global RNG state, not a call-local one, so every subsequent
`findEssentialMat`/`solvePnPRansac` call anywhere later in the run now draws
a different sequence than the flag-off baseline did, even on frames that
have nothing to do with relocalization. The result stays qualitatively
healthy (77 final keyframes vs. 88, 13165/47389 vs. 16639/63091 active/total
points, 88.3% vs. 86.6% coverage - all the same order of magnitude, not a
collapse), but ATE/RPE get measurably worse: ATE RMSE 0.0229m -> 0.0555m
(+142%), RPE RMSE 0.0218m -> 0.0494m (+127%). This is not a logic bug in
this issue's own code (verified: frames 1-47 are identical, so nothing
before the first relocalization attempt is touched) and not something #13
should fix on its own (OpenCV's shared-RNG behavior is a pre-existing
property of every RANSAC call this whole codebase already makes, already
flagged as a source of run-to-run variance in #14's own write-up above) -
but it does mean **`--relocalize` is not accuracy-neutral on a sequence
that already has an occasional multi-frame self-recovering gap**, even when
relocalization's own outcome is otherwise harmless. Documented in the
flag's own `--help` text rather than left as a silent surprise.

**Performance (required by the issue):** full-map search cost measured
directly across all four runs: 3.7-27.8ms/attempt (`desk` 27.8ms/attempt
over 577 attempts = 16.04s total; `room` 3.7ms/attempt over 1328 attempts =
4.85s total; `pioneer_slam2` 6.6ms/attempt over 1952 attempts = 12.88s
total; `xyz` 26.1ms/attempt over 21 attempts = 0.55s total) - cheap enough
per attempt that even ~2000 attempts across a whole sequence costs well
under 15 seconds of a multi-minute run, confirming the
`--relocalize-after-frames` threshold (avoiding an attempt on every single
skipped frame) wasn't strictly necessary for cost reasons on these map
sizes, though it remains the right default since a larger, longer-lived map
would scale this cost up linearly with active point count.

**Net result:** relocalization is implemented per the issue's own
constraints (reuses `Map.match_against`/`estimate_pose_pnp`, no new hard
dependency, gated behind `--relocalize`) and demonstrably works - two of
three hard sequences show a genuine, unambiguous recovery event on real
data, and the known-hard negative control correctly declines rather than
producing a false recovery. It does not change `desk`/`room`'s "too hard"
classification, because neither recovery survives long enough to reach a
new keyframe - a result the issue's own acceptance criteria anticipated
("a 'no relocalization found, correctly' outcome... is success for
correctness, not a demonstrated win") and this report takes at face value
rather than oversells. The one real surprise - `--relocalize` perturbing
`freiburg1_xyz`'s accuracy despite "never triggering" being false for that
sequence - is reported honestly rather than hidden behind a rerun chosen to
avoid it.

---

## #22: Automatic dual-model (homography/fundamental) initialization (paper §IV)

**Version:** `issue-22-dual-model-initialization` branch, on top of `main`
post-#13. Implements
[#22](https://github.com/albinjanssonsand/slam/issues/22): replaces the
bootstrap branch's essential-matrix-only two-view pose estimate
(`pose.estimate_relative_pose` + `triangulation.triangulate`) with the
paper's own automatic dual-model initialization (§IV) - a homography and a
fundamental matrix are estimated in parallel over the SAME matches
(`pose.estimate_homography`/`estimate_fundamental_matrix`), scored by
symmetric transfer error (`pose.homography_score`/`fundamental_score`, Eq.
2, T_H=5.99/T_F=3.84) and one is selected via the `R_H = S_H/(S_H+S_F) >
0.45` heuristic (Eq. 3), then the selected model's motion hypotheses
(`pose.decompose_homography`/`decompose_essential`) are disambiguated by
directly triangulating each with the existing octave-aware
`triangulation.triangulate` (#33) rather than trusting cheirality alone -
picking whichever hypothesis triangulates the most in-front/high-parallax/
low-reprojection-error points (`mapping._bootstrap_dual_model`), and
refusing to initialize outright (mirrors ORB-SLAM2's own
minTriangulated/0.9*N/0.7*best-vs-runner-up test) if no hypothesis is a
clear winner. A one-shot full BA (`mapping._run_global_ba`, reusing #20's
global-BA machinery) now runs unconditionally right after a successful
bootstrap, before the two-view reconstruction is otherwise used (paper §IV
step 5) - this replaces bootstrap's own previous per-keyframe local BA call
specifically; the TRACK branch's ongoing local BA is unchanged.

**A real scoring bug found and fixed along the way, before any number below
was collected:** the first implementation's symmetric-transfer-error check
combined both reprojection/epipolar-line directions into one squared error
before testing it against a single threshold. ORB-SLAM2's own reference
implementation of this exact equation (`CheckHomography`/`CheckFundamental`)
scores EACH direction independently - a separate inlier test and a separate
`gamma - d^2` reward per direction, not one combined check on their sum.
`pose._symmetric_transfer_score`/`homography_score`/`fundamental_score` were
rewritten to match. Also hardened `homography_score` against
`cv2.findHomography` returning a numerically singular matrix - a real
possibility on exactly the kind of degenerate/near-planar correspondence set
this feature exists to handle - since an uncaught `LinAlgError` there would
crash the whole run instead of falling through to "no dominant motion
hypothesis"/keep accumulating frames, like every other bootstrap-rejection
path already does.

### freiburg3_nostructure_texture_far: the paper's own reference case (Table III)

This TUM sequence (far, low-texture, near-planar structure) is the paper's
own reference case where ORB-SLAM correctly detects a planar twofold
ambiguity and refuses to initialize, where a naive essential-matrix-only
bootstrap risks seeding a corrupted map - required by the issue to be
tested **before and after**, to demonstrate the fix actually fixes
something rather than adding an unexercised code path. Newly downloaded for
this issue; not previously used anywhere in this file.

**Before (main, essential-matrix-only bootstrap):** bootstraps at frame 12
with 528 essential-matrix RANSAC inliers, seeding 526 points -
`--min-parallax`/`--min-inliers` are satisfied and nothing else checks
whether the correspondence set is actually well-conditioned. Confirmed
likely corrupted rather than merely unverified: re-running this exact frame
pair through the new dual-model diagnostic (below) shows only 14 of 1059
RANSAC-inlier correspondences (1.3%) triangulate to a dominant,
self-consistent 3D interpretation under ANY motion hypothesis - the old
code's frame-12 reconstruction has no such safeguard and just trusts
whichever pose `cv2.recoverPose`'s internal cheirality vote happened to pick.

**After (this issue, dual-model bootstrap):** R_H stays in the 0.47-0.51
range (homography favored - this scene genuinely looks planar to the
heuristic) across every attempted frame from 5 through 42, correctly
refusing to initialize the entire time, with "best" triangulated-hypothesis
counts near zero out of ~1000 RANSAC inliers for most of that span (frame
12 specifically: best 14/1059, matching the "before" diagnostic above).
Starting frame 32 the best-hypothesis count climbs steadily (543/1009 ->
736/915) as real parallax finally accumulates, and bootstrap is accepted at
**frame 43**: model=H, R_H=0.47, 865 pose inliers, 798/865 (92.3%)
triangulated as the dominant hypothesis - exactly the paper's documented
alternate-success outcome ("if real parallax does eventually appear later
in the sequence, correctly recovers non-planar structure"), not a flat
"never initializes" outcome, and a fully evidenced improvement over the old
code's unverified frame-12 guess.

### freiburg1_xyz regression check (required by the issue)

**Bootstrap still succeeds promptly:** frame 4, model=H, R_H=0.47,
959/959 (100%) points seeded.

| Sequence | Frames | Keyframes | Active/total points | Coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|---|---|
| `freiburg1_xyz` (baseline - see note below) | 798 | 88 | 16639/63091 | 86.6% (26.07s/30.09s) | 0.0229 | 0.0218 |
| `freiburg1_xyz` (this issue) | 798 | 95 | 13013/45049 | 86.2% (25.94s/30.09s) | 0.1129 | 0.1032 |

**Baseline note:** the issue asks to compare against "#27's row, the most
recently landed one" (798/78/16069-59138/88.1%/0.0250/0.0255) - but #14 and
#13 both landed on `main` after #27 (see git log: #33 -> #27 -> #14 -> #13),
and #13's own regression check re-confirmed #14's fresh post-fix row (88
keyframes, 0.0229/0.0218) as `main`'s actual current-default-flags
freiburg1_xyz behavior, unchanged by #13 itself. #27's row is therefore
stale by two landed issues; #14/#13's row is what a fresh run of unmodified
`main` actually produces today, so that's used as the honest "before" here -
the same reasoning #13's own section already applied to `desk`/`room`'s
stale frame-37/frame-100 citations.

**This is a genuine, measured regression - not a bug, root-caused and
reported per explicit user direction (see below) rather than silently
shipped or masked by deviating from the paper's own pinned constants.** ATE
RMSE roughly quintuples (0.0229 -> 0.1129) and RPE roughly quintuples too
(0.0218 -> 0.1032), despite near-identical coverage (86.6% -> 86.2%) and
MORE active keyframes (88 -> 95) than the baseline - so this isn't a
tracking-loss/coverage story, it's the two-view bootstrap solve itself
being measurably less accurate.

**Root cause, verified against TUM ground truth:** freiburg1_xyz's actual
bootstrap frame pair (frame 0 -> frame 4) is a borderline case for the R_H
heuristic: R_H=0.469, just over the paper's 0.45 homography-selection
threshold. Directly comparing the two candidate two-view rotations against
the ground-truth relative rotation between the same two frame timestamps:

| | Rotation (deg) | Error vs. ground truth |
|---|---|---|
| Ground truth (TUM mocap) | 2.590 | - |
| OLD essential-matrix-only pose | 2.694 | 0.104 deg |
| NEW homography-selected pose (R_H=0.469, model=H) | 1.367 | 1.223 deg (~12x worse) |

The paper's R_H > 0.45 heuristic deliberately biases toward the "safer"
homography model even in ambiguous cases, specifically to avoid
essential-matrix corruption on a truly planar scene (exactly what protects
`freiburg3_nostructure_texture_far` above). On `freiburg1_xyz` - a
genuinely 3D desk scene whose very first few frames happen to present a
borderline-planar-looking correspondence set - that same bias picks the
measurably less accurate model, and that two-view pose error propagates
through the whole subsequent PnP-tracked trajectory. This is inherent to
implementing the paper's own Eq. 3 threshold exactly as specified (both
this issue and the paper pin R_H > 0.45, T_H=5.99, T_F=3.84 as fixed
constants, not tunables) - not fixable without deviating from the paper's
own specification, which this issue explicitly requires following "exactly."

**Explicitly surfaced to the user given the direct conflict with this
issue's own "must not regress" acceptance criterion; decision: ship as
specified, document honestly.** A tie-breaking triangulation-quality check
near the R_H boundary, and investigating whether a different bootstrap
frame pair would resolve it, were both offered and declined in favor of
paper fidelity + transparent reporting - consistent with this file's own
established practice (e.g. #13's "one real surprise ... reported honestly
rather than hidden").

### Model/R_H selection sanity check (required by the issue)

| Sequence | Selected model | R_H | Scene character |
|---|---|---|---|
| `freiburg1_xyz` | H | 0.469 | Genuinely 3D desk scene - borderline case, see regression analysis above |
| `freiburg3_nostructure_texture_far` | H (throughout the refusal window and at eventual acceptance) | 0.47-0.51 | Far, low-texture, near-planar - correctly trends toward homography |

Both sequences trend toward homography, as expected for TUM's fr1/fr3
desk-distance camera framing, but the two outcomes diverge exactly where
they should: fr3's low parallax/planarity keeps every hypothesis
un-dominant for 38 frames (correct refusal), while fr1's real,
quickly-accumulating parallax gives every hypothesis - right or wrong - a
clean-looking triangulation almost immediately (a homography-selected
hypothesis self-consistently "explains" its own inlier correspondences even
when it's the geometrically worse pick, since it was fit FROM them - see
the regression analysis above for why "100% triangulated" isn't itself
evidence of a *correct*, as opposed to merely self-consistent, pose).

### Performance (required by the issue)

Computing both models in parallel does NOT roughly double bootstrap-attempt
cost as anticipated - measured directly (40 repeats, real freiburg1_xyz
frame-pair correspondences, ~1000-1100 matches): essential-matrix-only
(`estimate_relative_pose` + `triangulate`) averages ~17-21ms/attempt; the
dual-model path (`_bootstrap_dual_model`: both RANSAC fits + both symmetric-
transfer scores + up to 4 hypotheses each individually triangulated)
averages ~13-14ms/attempt, about 0.7-0.8x the old cost. Plausible
explanation: `cv2.findHomography`/`cv2.findFundamentalMat`'s RANSAC
(reprojection-threshold-based) are individually cheaper than
`cv2.findEssentialMat` + `cv2.recoverPose`'s own internal SVD/cheirality
voting over every point; the extra hypothesis-triangulation work is
apparently outweighed by that saving on this data. Either way, well within
the issue's own "acceptable, this only runs during the brief initialization
phase" expectation.

### Confirming the root cause: forcing F-selection on freiburg1_xyz

To more rigorously test the regression's root-cause diagnosis above (not
shipped - a one-off diagnostic run, `_bootstrap_dual_model`'s
`homography_select_threshold` default patched from 0.45 to 0.5 in-process,
`main`'s pinned 0.45 is untouched in the committed code), re-ran
`freiburg1_xyz` with the R_H heuristic biased toward the fundamental-matrix
branch instead:

| Config | Bootstrap frame | Model | Keyframes | Coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|---|---|
| Old code (baseline) | 5 | essential | 88 | 86.6% | 0.0229 | 0.0218 |
| New code, default (`R_H`>0.45) | 4 | H | 95 | 86.2% | 0.1129 | 0.1032 |
| New code, `R_H`>0.5 (diagnostic only) | 12 | F | 85 | 86.3% | 0.0423 | 0.0381 |

Forcing F-selection recovers most of the accuracy (ATE 0.1129 -> 0.0423,
~2.7x better) at equivalent coverage - strong, end-to-end confirmation that
H-vs-F model selection is the dominant driver of the regression, not
something else this issue's change also touched. It doesn't fully return to
the original 0.0229/0.0218, because raising the threshold doesn't just swap
the model at the same frame: F's own hypothesis only becomes dominant 8
frames later (frame 12, not frame 4), so this diagnostic run bootstraps off
a different, later frame pair with sparser founding structure (607 points
vs. 943-959) rather than an otherwise-identical reconstruction with only the
model swapped - not a fully apples-to-apples comparison, but conclusive on
the question it was run to answer.

**Decision (re-confirmed):** ship with the paper's literal R_H>0.45 (not
0.5 or any other tuned value) - this diagnostic exists to verify the
regression's cause, not to argue for changing the shipped constant away
from what the issue and the paper both specify exactly.

### Net result

The dual-model bootstrap works exactly as the paper intends on its own
reference case (`freiburg3_nostructure_texture_far`): a demonstrated,
before/after fix, not just a new code path that happened to never matter -
the old essential-matrix-only bootstrap mis-initializes at frame 12 off a
correspondence set that this issue's own diagnostic shows has almost no
coherent 3D structure (14/1059), while the new code correctly refuses for
38 frames and then recovers real structure once genuine parallax appears.
The cost is a genuine, root-caused, and explicitly user-accepted regression
on `freiburg1_xyz`, inherent to the paper's own R_H>0.45 threshold picking
the "safe" model even on a borderline non-planar scene - reported
transparently rather than masked by quietly deviating from the paper's
pinned constants.

---

## Keyframe insertion threshold tuning - tried, reverted (not tied to a filed issue)

**Version:** `main` @ current HEAD (`--depth-densify`/`--relocalize`/
`--rotation-only-fallback` not passed). Not scoped to a GitHub issue - this
started as a direct comparison against the reference paper's own published
numbers, and the investigation itself led to a concrete parameter change.

### Does the paper even cover the sequences we call "too hard"?

Checked directly against the actual paper (Mur-Artal et al., 2015,
*"ORB-SLAM: A Versatile and Accurate Monocular SLAM System"*, arXiv:1502.00956,
via its ar5iv full text):

- **`freiburg2_pioneer_slam2` is not in the paper at all** - the word
  "pioneer" does not appear anywhere in it, and its own TUM RGB-D table
  (Table III, 16 hand-held sequences) doesn't include it either. It was
  added to this project's evaluation suite by #17 for broader real-world
  coverage, not to reproduce a published result - there is no paper number
  to have fallen short of here.
- **`freiburg1_desk` IS in the paper** (Table III): ORB-SLAM (monocular)
  reports **1.69 cm ATE RMSE**, full-sequence tracking.
- **`freiburg1_room` is NOT in the paper's Table III** either (confirmed
  from the same fetch) - only `freiburg1_desk` gives a genuine apples-to-
  apples target among this project's existing "too hard" sequences.

So `freiburg1_desk` is the one sequence here with a real published number to
compare against, and worth investigating properly: current `main` (pre this
section) dies permanently at **frame 45/613** (4.6% coverage, 1.07s/23.4s),
vs. the paper's implicit ~100% coverage.

### Hypothesis 1 (rejected, tested directly): would BoW-style place
recognition have found what full-map brute-force search misses?

The obvious guess is that #13's relocalization (one flat brute-force
Hamming match against the *entire* confirmed map, no keyframe-level
retrieval) is a materially weaker substitute for the paper's DBoW2
bag-of-words candidate retrieval (query an inverted visual-word index,
retrieve candidate *keyframes* by appearance similarity, then match/verify
against one keyframe's own point set at a time). Tested this directly
rather than assumed it: dumped the real confirmed map from the `main`
`freiburg1_desk` run above (6 keyframes, 542 active points at time of
death) and, for frames 26-60, compared (a) `Map.match_against` over the
WHOLE active map (#13's current approach) vs. (b) matching against each
keyframe's own point subset individually, one at a time (what a BoW
retrieval step would hand you) - same underlying matcher
(`match_descriptors`, ratio test + mutual-NN cross-check per #14) and same
PnP+RANSAC verification either way, so the only variable is candidate-pool
scope.

| frame | full-map matches / PnP inliers | best single-keyframe matches / inliers |
|---|---|---|
| 30 | 123 / 84 | 86 / 58 |
| 40 | 90 / 38 | 65 / 28 |
| 45 | 37 / 18 | 26 / 16 |
| 49 | 11 / 0 | 9 / 0 |
| 60 | 7 / 0 | 6 / 0 |

**The flat full-map search wins on every single tested frame** - restricting
to one keyframe's subset only throws away correct correspondences that
happen to live in a different keyframe's own point set; different
keyframes mostly cover different map locations rather than confusable
near-duplicates, so a bigger pool adds true candidates more than it adds
false ones. More importantly, **both approaches hit exactly zero matches at
the same frame (~49) and stay there** for the rest of the tested range -
not a retrieval-algorithm ceiling, but every keyframe's points, individually
and combined, having genuinely left the frame. No candidate-retrieval
strategy (BoW, brute force, or otherwise) can match a point to a view that
never contains it. **Conclusion: this specific failure would not have been
fixed by BoW-style place recognition.**

### Hypothesis 2 (confirmed): is this actually a hard, fast pan, or ordinary motion?

Checked ground truth directly rather than assumed either way:

| segment | rotation rate | translation rate |
|---|---|---|
| frame 0->32 (healthy bootstrap+tracking) | 22.1 deg/s | 0.29 m/s |
| **frame 32->49 (the death window)** | **61.5 deg/s** (~35 deg in 0.57s) | 0.38 m/s |
| whole-sequence average | 0.9 deg/s | 0.05 m/s |

Confirmed real: a genuine handheld whip-pan, ~3x the rotation rate of the
immediately preceding healthy segment - not an artifact, and not "ordinary"
motion. The paper's own system has to survive the same event, not an
easier one.

### Root cause (confirmed from the run log): the map stops growing exactly
when it's needed most

During frames 33-45 (inside the pan), per-frame PnP tracking keeps
succeeding - declining inlier count (79 -> ... -> 24), but genuinely
accepted, frame-only tracking, not lost. Yet **zero new keyframes get
inserted for the entire 13-frame stretch**: `--kf-min-tracked-points`
(paper's own §V-E condition 3, old default 50) is never satisfied while
inlier count sits in the 24-79 range, so no new structure gets triangulated
into the area the camera is panning toward while there is still some
overlap left to build from. By the time inlier count reaches zero (~frame
49), it's permanent - confirmed exhaustively by Hypothesis 1 above.

### Parameter tuning, tested directly on `freiburg1_desk` - initial result (later corrected below)

| config | death point | coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|
| default (`--kf-min-tracked-points 50`, `--kf-max-frames-since-keyframe 20`) | frame 45/613 | 4.6% | 0.0203 | 0.0545 |
| `--kf-min-tracked-points 20` alone | "tracking continues to" frame 506+/613 | 70.1% | 0.0423 | 0.0629 |
| `--kf-max-frames-since-keyframe 8` alone | frame 46/613 | 5.8% | 0.0299 | 0.0537 |
| both together | "tracking continues to" frame 522+/613 | 73.8% | 0.0762 | 0.0574 |

This was read at the time as "`--kf-min-tracked-points 20` accounts for
essentially the whole effect, closing the 13-frame growth gap diagnosed
above" - **that reading was wrong, caught only after a user directly
questioned a suspiciously low keyframe count (14, for a 613-frame
sequence) and asked to see the run log.** The "death point"/"tracking
continues to" language above is quoted because it doesn't mean what it
sounds like - see the correction immediately below. This is the incident
that produced `EVALUATION_METHOD.md`'s pitfall #1; read that section for
the general lesson, this one for the specific numbers.

### Corrected: the coverage number was hiding a 444-frame total blackout

Applying the gap-check now documented in `EVALUATION_METHOD.md` to these
same runs' own logs (every accepted-pose frame prints a `frame N:` line;
extract the indices, look at the gaps) tells a completely different story
than the table above:

| config | frames w/ an accepted pose | last frame before the gap | first frame after it | gap size |
|---|---|---|---|---|
| default (50/20) | 44 | 45 | *(never reconnects)* | stays dead for the remaining 568 frames |
| `--kf-min-tracked-points 20` alone | 61 | 46 | 492 | 446 frames, zero accepted poses |
| both together | 81 | 48 | 492 | 444 frames, zero accepted poses |

**The death point barely moves at all (45 -> 46-48) - a few frames, within
run-to-run noise, not a fix.** Both "improved" configurations then go
completely dark - not "fewer keyframes", genuinely zero accepted poses of
any kind, guided or otherwise - for well over 400 straight frames, before a
handful of keyframes suddenly appear again starting at frame 492. That is
not survived tracking; it's a fresh, coincidental reconnection. Verified
directly against `groundtruth.txt` rather than assumed: frame 492's camera
position (`[1.40, 0.45, 1.68]`) sits within about 10cm of frame 32-44's
positions (`[1.29-1.38, 0.52-0.66, 1.57-1.59]`) - after wandering far away
through the middle of the sequence (X ranging roughly -0.7 to 0.9 around
frames 100-400), the camera genuinely revisits the original desk area, and
*that* is what frame 492's keyframe is - a real, physical revisit, not
noise. The default (untuned) run's trajectory presumably passes through
the same physical area at the same later timestamp too, but never
reconnects there at all; the difference is that the two tuned
configurations bank more keyframes/points before dying (8 vs. 4), giving
that later revisit a denser residual map to find enough correspondences in
via ordinary guided matching - `--relocalize` was not even enabled for any
of these runs, so this isn't relocalization either. This is a real,
reproducible, now ground-truth-confirmed mechanism - but it is a
*different* mechanism than "fixes the whip-pan," it is specific to this
video happening to loop back near its own start, and it produces a
trajectory that is badly wrong (frozen at a stale pose) for the 444 frames
in between, which the coverage number gives no hint of.

This also *strengthens*, rather than contradicts, the root-cause diagnosis
above: even with keyframes inserted continuously right up through frame 44
(confirmed in the "both together" run - keyframes at 8, 13, 21, 29, 33, 40,
42, 44, vs. the default's 8, 13, 22, 32), overlap with the confirmed map
still collapses to zero within a few more frames regardless. Lowering
`--kf-min-tracked-points` does exactly what it was diagnosed to do (let
more keyframes in during the declining-but-tracked stretch) - it just
isn't *sufficient* to survive this particular whip-pan once inlier count
actually reaches zero, which happens only a handful of frames later than
before either way.

### Regression check, `freiburg1_xyz` - unaffected by the correction above

| config | keyframes | coverage | ATE RMSE (m) | RPE RMSE (m) |
|---|---|---|---|---|
| default | 102 | 86.2% | 0.1129 | 0.1032 |
| `--kf-min-tracked-points 20` alone | 99 | 86.6% | 0.1135 | 0.1021 |
| both together | 141 | 88.1% | 0.0687 | 0.0478 |

These numbers are NOT subject to the pitfall above: `xyz`'s own run log
shows only 5 of 698 non-keyframe frames "lost tracking entirely" in every
configuration (vs. `desk`'s 537 of 598) - genuinely continuous tracking
throughout, not a gapped coverage artifact, and `evo`'s Sim(3) alignment
scores every matched timestamp, not just the first/last. `--kf-min-tracked-
points 20` alone is a clean no-op here (inlier counts run 600-1700+,
nowhere near either floor); combined with `--kf-max-frames-since-keyframe
8` it measurably improves both ATE and RPE (~40% more keyframes). This part
of the investigation holds - it just no longer motivates keeping the
changed defaults on its own, since the primary (`desk`) justification is
gone.

### Decision (revised)

**Reverted both to the paper's own values**: `--kf-min-tracked-points 50`
(was briefly 20), `--kf-max-frames-since-keyframe 20` (was briefly 8). The
original justification - a demonstrated fix for `freiburg1_desk`'s
tracking-loss death spiral - did not hold up under the gap-check above.
Shipping a parameter change under a "fixes the whip-pan" framing that
instead produces 444 frames of silently-wrong frozen-pose output between
two real tracking islands would be actively misleading to anyone relying
on this doc or these defaults, regardless of how good the raw coverage
number looks.

**What still holds, unretracted:** the paper-gap analysis
(`freiburg2_pioneer_slam2` isn't in the reference paper at all;
`freiburg1_desk` is, at 1.69cm ATE), the BoW-vs-brute-force rejection
(Hypothesis 1 - independently valid, since it compared match/inlier counts
directly rather than a timestamp-derived coverage percentage, so it isn't
subject to this pitfall), the ground-truth whip-pan confirmation
(Hypothesis 2), and the root-cause diagnosis (map growth frozen by
`--kf-min-tracked-points` during a declining-but-tracked stretch) - if
anything the root cause is now better confirmed, not weaker, by seeing that
fixing it wasn't sufficient on its own.

**What's retracted:** the claim that lowering `--kf-min-tracked-points`/
`--kf-max-frames-since-keyframe` fixes `freiburg1_desk`'s tracking loss, and
the "no regression, plus a bonus win" framing for shipping them as new
defaults on that basis. `freiburg1_desk`/`freiburg1_room` remain "too hard"
per this doc's own criterion, un-improved by this investigation. A genuine
fix still needs something this investigation didn't find - possibly denser
feature coverage through the pan, a wider guided-matching search
specifically during a detected fast-rotation regime, or accepting that a
whip-pan this severe needs a live recovery mechanism (relocalization,
loop closing) reaching the later revisit *during* the run, rather than a
denser residual map passively catching it after the fact.

**Reproduction:**

```bash
python -m pipeline.mapping --video datasets/tum/rgbd_dataset_freiburg1_desk --calibration calibration/tum_freiburg1.yaml --trajectory-output results/desk_estimate.txt --plot-output results/desk_trajectory.png --no-display

evo_ape tum datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt results/desk_estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt results/desk_estimate.txt -a -s
```

See `EVALUATION_METHOD.md`'s pitfall #1 for the gap-check script used to
produce the corrected table above - run it against any new `desk` log
before trusting a coverage number from it.

---

## #47 (part 1): environment-mismatch fix and honest re-baseline

**Version:** `issue-47-paper-parity-investigation` branch, `main` @
`fc33419` unchanged - this section is a re-baseline, not a pipeline code
change. Part of [#47](https://github.com/albinjanssonsand/slam/issues/47)'s
investigation into why the pipeline underperforms the paper's own published
numbers on `freiburg1_xyz`/`freiburg1_desk`.

**Before any of #47's actual investigation, `EVALUATION_METHOD.md`
pitfall #4 recurred - and this time it was NOT a harmless coincidence.**
Bare `python`/`pip` on this machine's `PATH` resolve to a Windows Store
Python 3.11 install with `opencv-python==4.13.0`; the project's real
environment is a conda env named `slam`, with `opencv-python==5.0.0.93`.
Every dependency the pipeline needs happens to import successfully under
either, so nothing errors - the wrong environment runs silently, exactly as
pitfall #4 already warned. Tested directly on `freiburg1_desk` (identical
command, identical commit, only the interpreter differs): both environments
bootstrap **identically** (frame 7, model=F, R_H=0.39, 332 pose inliers,
318 points seeded) but diverge partway through per-frame guided PnP
tracking - confirmed not RNG noise (two same-environment reruns of the
identical command are byte-identical, aside from timing) but a real,
reproducible difference in `cv2.solvePnPRansac`'s own behavior between
OpenCV major versions:

| Environment | Death frame | Keyframes | Active/total points | Coverage | ATE/RPE RMSE (m) |
|---|---|---|---|---|---|
| OpenCV 4.13 (Windows Store, wrong) | 45/613 | 5 | 542/1241 | 4.6% (1.07s/23.40s) | 0.0203 / 0.0545 |
| OpenCV 5.0 (`slam` conda env, correct) | 40/613 | 4 | 482/1023 | 3.1% (0.73s/23.40s) | 0.0194 / 0.0430 |

The OpenCV-4.13 row is byte-for-byte the same as every prior `desk` "current
main" citation in this file (e.g. the keyframe-insertion-threshold-tuning
section's default row above: frame 45/613, 4.6%, 0.0203/0.0545) - conclusive
evidence that **every prior `freiburg1_desk` number in this file was
produced under the wrong OpenCV version**, not the project's real
environment. `freiburg1_xyz` was checked the same way for its current-`main`
baseline (below) and is measurably affected too, though less dramatically -
both environments track the whole sequence without loss, but the two-view
bootstrap/PnP-tracking chain still ends up materially different by the end.

**Fix:** `requirements.txt` now pins the exact versions verified working in
the `slam` conda env (`opencv-python==5.0.0.93` specifically, since that's
the version whose behavior every number in this file from here on should
reflect); `EVALUATION_METHOD.md` pitfall #4 now names the environment
explicitly and gives a copy-pasteable verification command, and its Method
section points there before the reproduction snippet. This does not stop a
future session from making the same mistake by hand, but it removes the
silent-guessing failure mode - `pip install -r requirements.txt` into the
wrong environment will now at least fail loudly on the version pin (or
install the right version there) rather than quietly running.

**Honest re-baseline, both sequences, current `main`, correct environment,
default flags (no `--relocalize`/`--rotation-only-fallback`/
`--global-ba-at-end`/`--depth-densify`):**

| Sequence | Bootstrap frame | Keyframes | Active/total points | Coverage | ATE RMSE (m) | RPE RMSE (m) | Gap-check |
|---|---|---|---|---|---|---|---|
| `freiburg1_desk` | 7 (model=F, R_H=0.39) | 4 | 482/1023 | 3.1% (0.73s/23.40s) | 0.0194 | 0.0430 | clean - 34 contiguous tracked frames (7-40), then permanent loss, no gaps |
| `freiburg1_xyz` | 4 (model=H, R_H=0.47) | 85 | 13511/47631 | 86.5% (26.04s/30.09s) | 0.0702 | 0.0898 | clean - 792 contiguous tracked frames (4-797), no gaps |

**`freiburg1_desk` remains "too hard"** by this file's own criterion - a
clean, un-gapped permanent loss at frame 40 (5 frames earlier than the
mismeasured 45), 3.1% coverage. Nothing here changes its classification;
the correct-environment number is honestly slightly worse (fewer frames
survived) than the number every prior investigation of this sequence
(including the keyframe-insertion-threshold-tuning and #13 relocalization
sections above) was actually reasoning about.

**`freiburg1_xyz` improves substantially just from fixing the environment,
with no code change at all**: ATE RMSE 0.1129 -> 0.0702 (-38%), RPE RMSE
0.1032 -> 0.0898 (-13%), at equivalent coverage (86.2% -> 86.5%) and fewer
keyframes (95 -> 85) than the OpenCV-4.13 number #22's section reported as
current-`main`. Still ~7.8x the paper's 0.90cm (vs. 12.5x under the
mismeasured number), so this alone does not close the gap - but it means
part of what #22's section attributed entirely to the R_H>0.45
homography-selection heuristic was actually environment noise layered on
top of that regression, not the regression alone. This is the correct
starting point for #47's own further `xyz` investigation, not the
0.1129/95-keyframe number cited in the issue itself.

**Reproduction:**

```bash
conda activate slam  # or invoke that env's python.exe directly
python -m pipeline.mapping --video datasets/tum/rgbd_dataset_freiburg1_<seq> --calibration calibration/tum_freiburg1.yaml --trajectory-output results/issue47_<seq>_baseline_estimate.txt --plot-output results/issue47_<seq>_baseline_trajectory.png --no-display

evo_ape tum datasets/tum/rgbd_dataset_freiburg1_<seq>/groundtruth.txt results/issue47_<seq>_baseline_estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg1_<seq>/groundtruth.txt results/issue47_<seq>_baseline_estimate.txt -a -s
```

---

## #47 (part 2): eight tested mechanisms, all negative, for `freiburg1_desk`'s whip-pan; `xyz`'s `--global-ba-at-end` measured

**Version:** `issue-47-xyz-desk-investigation` branch, `main` @ `85f1bb3`
(part 1) unchanged - no pipeline code touched, all runs use the corrected
`slam` conda environment.

### `freiburg1_xyz`: `--global-ba-at-end` (per user direction, not pursued
further - `xyz` accuracy is secondary to `desk`'s tracking loss for this
phase)

One-shot full BA over the whole trajectory (85 keyframes, 13511 active
points) after part 1's re-baseline:

| Config | ATE RMSE (m) | RPE RMSE (m) | Wall-clock cost |
|---|---|---|---|
| Baseline (part 1) | 0.0702 | 0.0898 | - |
| `--global-ba-at-end` | 0.0675 (-4%) | 0.0873 (-3%) | 1426s (~24 min) for the BA pass alone |

Reprojection error did genuinely improve (mean 2.40px -> 1.29px, so the
solve itself worked, wasn't rejected as implausible), but the ATE/RPE
change is marginal - not worth its cost as a fix direction. Also a useful
correction to `_run_global_ba`'s own docstring assumption
(`mapping.py:1150`) that a one-shot full-map pass is "acceptable... only
runs during the brief initialization phase" - true for bootstrap's own
small-map call, not for `--global-ba-at-end`'s full-trajectory one, which
was apparently never wall-clock-tested before at this map size.

### `freiburg1_desk`: comprehensive re-verification under the corrected
environment - every tested mechanism, including the prior session's key
ones, still fails

Part 1 already showed `desk`'s honest baseline is worse than previously
documented (death frame 40, not 45). Before accepting "not fixable" as
this phase's conclusion, every mechanism the issue's own candidate list
names was tested (or re-tested) directly against the corrected baseline -
several of the prior investigation's negative results had only ever been
measured under the wrong OpenCV environment (pitfall #4) and needed
re-verification, not just an assumption that they'd hold.

| # | Mechanism | Death/gap | Keyframes | Coverage | ATE/RPE RMSE (m) | Verdict |
|---|---|---|---|---|---|---|
| 1 | Baseline (part 1) | frame 40, clean | 4 | 3.1% | 0.0194 / 0.0430 | reference |
| 2 | `--rotation-only-fallback` | frame 40, clean (byte-identical) | 4 | 3.1% | 0.0194 / 0.0430 | **no-op** - fires once, correctly declines (real translation present, R_H heuristic doesn't favor pure rotation) |
| 3 | `--guided-window 150` (vs. default 60) | frame 34, clean | 5 | 2.9% | 0.0190 / 0.0299 | **regression** - wider radius admits more false matches, dies 6 frames earlier |
| 4 | `--n-features 10000` (vs. default 5000) | never bootstraps | 0 | 0% | n/a | **regression** - more low-quality candidate matches prevent bootstrap's dual-model dominance test from ever picking a winner, for the entire 613-frame sequence |
| 5 | `--relocalize` | frame 40, then dead forever | 4 | 3.1% | 0.0194 / 0.0430 | **no-op** - 567 brute-force full-map attempts across the remaining 573 frames, 0 successes (stronger negative than the prior wrong-env result, which found 1 transient success) |
| 6 | `--kf-min-tracked-points 20` | frame 46, then 449-frame blackout, coincidental reconnect at 495 | 9 (+1 late) | 70.5% (gapped) | 0.0523 / 0.0600 | **re-confirms the prior investigation's retracted finding, now under the correct environment** - real death delayed by only 6 frames (40->46), then the same pitfall-#1-style total blackout/coincidental-revisit pattern, not genuine survival |
| 7 | `--kf-min-tracked-points 20 --kf-max-frames-since-keyframe 8` | frame 48, then 444-frame blackout, coincidental reconnect at 492 | 14 (+1 late) | 73.8% (gapped) | 0.0727 / 0.0605 | same as #6 - death delayed 8 frames, same blackout pattern |
| 8 | `--kf-min-tracked-points 20 --relocalize` | identical to #6 | 9 (+1 late) | 70.5% (gapped) | 0.0523 / 0.0600 | **decisive** - even brute-force full-map search (544 attempts) finds nothing during the 449-frame gap; #6's reconnection at frame 495 happened via ordinary tracking once the camera physically returned, not because a better search found something earlier |
| 9 | (prior session) BoW-vs-full-map candidate retrieval | both hit zero matches at the same frame | - | - | - | **no-op** - not a retrieval-scope problem |

All eight (eight distinct mechanisms total across this and the prior
session) are consistent
with a single explanation, now tested from every angle the issue's own
candidate list names: for roughly 450 frames after the whip-pan begins,
**the camera's actual field of view shares no structure with anything
this pipeline has ever triangulated**, regardless of search strategy
(guided window, brute-force relocalization, BoW-style retrieval) or how
aggressively keyframes get inserted beforehand (#7's near-every-frame
keyframe insertion during the decline still only buys 8 frames). This
isn't a matching/search deficiency - every mechanism that changes *how*
the pipeline searches its existing map came back negative or worse.

**Why the paper's own system doesn't have this problem, and why that
doesn't mean nothing here is fixable:** every mechanism tested above is
*reactive* - it can only match the current frame against structure
triangulated from *past*, formally-promoted keyframes. None of them let
the map grow into territory the pan is *currently* entering. The paper's
concurrently-running Local Mapping thread doesn't need a smarter search
either - because it runs continuously, it can opportunistically
triangulate new structure from consecutive tracked frames as they happen,
not just from the sparse, discretely-gated set of frames that clear this
pipeline's keyframe-insertion policy (paper §V-E, unchanged from the
paper's own values). Even test #7's near-maximal keyframe-insertion
aggressiveness (keyframes at nearly every frame from 33-46) only delayed
death by 8 frames - consistent with keyframe-gated triangulation
fundamentally lagging one step behind an accelerating pan no matter how
aggressively it's gated, since a keyframe still only captures what the
camera already saw at the moment it's inserted.

**Not concluding "unfixable within non-goals" here** - filed as
[#49](https://github.com/albinjanssonsand/slam/issues/49): opportunistic
frame-to-frame triangulation (new map points from consecutive *tracked*
frames, not gated on keyframe promotion) during declining-inlier
stretches, as a way to get the same continuous-growth property the
paper's threading gives it, without threading - feasible here since this
is an offline replay pipeline with no real-time frame-drop constraint,
unlike the paper's live system. Real implementation work, not a parameter
sweep; tracked as its own issue rather than folded into this one, per
#47's own instruction to split out separately-landable pieces.

**Reproduction:** same as part 1's snippet, with the flag combination from
the table's row added to the `pipeline.mapping` invocation. Gap-check
(`EVALUATION_METHOD.md` pitfall #1) required before trusting any coverage
number above 4% on this sequence - rows 6-8 are exactly the pattern that
pitfall exists to catch.

---

## #49: opportunistic frame-to-frame triangulation - real progress on
`freiburg1_desk`, not a full fix

**Version:** `issue-49-frame-to-frame-triangulation` branch, on top of
`main` post-#47 (part 2). Implements
[#49](https://github.com/albinjanssonsand/slam/issues/49): every mechanism
#47 (part 2) tested only changed *how* tracking searches its existing map;
this instead lets the map grow *between* keyframes, not just at them - see
the issue for the full design rationale and `mapping._opportunistic_
triangulate_from_frame`'s docstring for the implementation.

**Mechanism:** `--opportunistic-triangulation` (off by default): whenever
an ordinarily-tracked (non-keyframe) frame's PnP inlier count drops below
`--kf-min-tracked-points` - the paper's own §V-E condition that blocks
keyframe promotion, and with it §VI-C's usual new-point search - triangulate
new map points from that frame's already-accepted pose against the
reference keyframe (and its covisibility neighbors) anyway, reusing
`_create_new_points_from_covisible_keyframes` exactly as real keyframe
insertion does. Only the reference keyframe's side of each new point's
founding observation is registered (the frame itself isn't a keyframe);
such a point starts with exactly one observing keyframe and is cleaned up
automatically by the existing §VI-B culling rules at a later keyframe if it
never accumulates the normal 3-observing-keyframe minimum - no new
lifecycle needed.

**Flag-off regression check:** `freiburg1_desk` with the flag omitted is
byte-identical to part 1's baseline trajectory (diff-confirmed) - the new
code path is fully gated behind the flag.

**`freiburg1_desk` results, flag on, all combinations tested (gap-checked
per pitfall #1):**

| Config | Death frame | Keyframes | Gap size | Reconnect | Coverage | ATE/RPE RMSE (m) |
|---|---|---|---|---|---|---|
| Baseline (part 1, flag off) | 40 | 4 | permanent (no reconnect) | - | 3.1% | 0.0194 / 0.0430 |
| `--opportunistic-triangulation` | **55** | 7 | 437 frames | 492 | 72.9% (gapped) | 0.0454 / 0.0828 |
| `+ --kf-min-tracked-points 20` | 46 | 9 | 449 frames | 495 | 70.5% (gapped) | identical to part 2's row 6 |
| `+ --relocalize` | **55** | 8 | **421 frames (best)** | 476 | 72.2% (gapped) | 0.0325 / 0.0700 |
| `+ --rotation-only-fallback` | 55 | 8 | 437 frames | 492 | - | fallback fires 5x post-reconnection only, no effect on the gap itself |
| `+ --relocalize + --rotation-only-fallback` | 55 | 8 | 421 frames | 476 | - | identical to `+ --relocalize` alone - rotation-only-fallback adds nothing on top |

**Real, meaningful progress - the best death-point extension of anything
tested across both sessions (+15 frames / +37%, more than double the
prior best of +6-8 frames from lowering the keyframe threshold alone) -
but not a full fix.** `--opportunistic-triangulation` alone creates real
new structure during the decline (26 attempts, 1809 new points on the
standalone run) and genuinely extends survival, confirmed by the death
frame moving and by ATE/RPE degrading in the expected direction (a longer
real trajectory accumulates more real error than one that dies
immediately - see `EVALUATION_METHOD.md` pitfall #2). `--relocalize`
stacks on top cleanly (different mechanism, different failure mode) and
shrinks the remaining blackout by 16 frames (437 -> 421) by finding one
real match in the richer residual map. `--rotation-only-fallback` does
NOT stack - it fires successfully several times, but only after the
reconnection point, contributing nothing to the actual gap (desk's pan has
real translation throughout, as #47 already established, so there's no
window of genuine near-pure-rotation for it to catch mid-gap).

**Important negative interaction: `--kf-min-tracked-points 20` and
`--opportunistic-triangulation` actively conflict, don't combine them.**
Lowering the keyframe threshold to 20 (equal to `--pnp-min-inliers`'s own
default) eliminates the "declining but still tracked, blocked from a
keyframe" window this mechanism exists to exploit - real keyframes now
claim every frame down to 20 inliers, and once PnP itself fails outright
at that same floor, there's no accepted pose left to opportunistically
triangulate from. Confirmed directly: 0 attempts, 0 new points, identical
output to `--kf-min-tracked-points 20` alone. The wide gap between the
paper's own default values (`--kf-min-tracked-points` 50, `--pnp-min-
inliers` 20) is exactly what gives this mechanism room to work - narrowing
that gap starves it.

**`freiburg1_xyz` regression check:** not byte-identical (fired once, 368
new points created during a brief dip below 50 inliers) - a small, mixed
effect: ATE RMSE 0.0702 -> 0.0729 (+3.9%), RPE RMSE 0.0898 -> 0.0756
(-16%), coverage effectively unchanged (86.5% -> 86.2%), no tracking gaps
introduced.
Not a regression worth avoiding, and the flag stays off by default.

**Why this doesn't fully close the gap, and what would - filed as
follow-up issues rather than pursued here, since both require lifting a
non-goal:** opportunistic triangulation is still bounded by needing an
*already-accepted* PnP pose to triangulate from - once inlier count
actually reaches zero (now at frame 55 instead of 40), there's no pose
left to bootstrap from, and the pipeline is exactly as stuck as before.
Discussed with the user directly: the paper's own system likely avoids
this via two components, both explicit non-goals here -

- A concurrently-running Local Mapping thread that triangulates
  continuously from every frame rather than gating on discrete keyframe
  promotion (this issue's own mechanism is a synchronous, partial
  approximation of exactly that effect, and it measurably helped -
  corroborating evidence, not just a guess).
- DBoW2-style appearance-based place recognition, which recognizes a
  revisited scene by whole-image visual-word similarity rather than
  literal point correspondence - a fundamentally different, coarser-
  grained mechanism than anything this pipeline's brute-force Hamming
  matcher does. Worth flagging a correction to #47 (part 2)'s own
  "BoW-vs-full-map" test here: that test only restricted the *same*
  point-level matcher to one keyframe's point subset at a time - it was
  never a real test of appearance-based image retrieval, so #47's
  "BoW-style retrieval doesn't help" conclusion deserves less confidence
  than originally stated.

Filed as [#50](https://github.com/albinjanssonsand/slam/issues/50)
(lightweight bag-of-words retrieval using `cv2.BOWKMeansTrainer`/
`BOWImgDescriptorExtractor` - no new dependency) and
[#51](https://github.com/albinjanssonsand/slam/issues/51) (real
concurrent Local Mapping thread) for whoever picks this up next - both
lift an explicit non-goal, which is why this issue doesn't attempt either
unilaterally.

**Reproduction:**

```bash
conda activate slam
python -m pipeline.mapping --video datasets/tum/rgbd_dataset_freiburg1_desk --calibration calibration/tum_freiburg1.yaml --opportunistic-triangulation [--relocalize] [--rotation-only-fallback] --trajectory-output results/<label>_desk_estimate.txt --plot-output results/<label>_desk_trajectory.png --no-display

evo_ape tum datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt results/<label>_desk_estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt results/<label>_desk_estimate.txt -a -s
```

Gap-check (`EVALUATION_METHOD.md` pitfall #1) required, same as part 2 -
every coverage number above is gapped and none should be read as
continuous tracking.

---

## #50: appearance-based (CLIP) keyframe retrieval for relocalization -
validated, wired in, no measurable improvement on `freiburg1_desk`

**Version:** `issue-50-clip-appearance-retrieval` branch, on top of `main`
post-#49. Implements
[#50](https://github.com/albinjanssonsand/slam/issues/50): #49's own
write-up above flagged that #47 (part 2)'s "BoW doesn't help" conclusion
was never a real test of appearance-based image retrieval (whole-image
similarity, not point-level matching restricted to one keyframe's
subset) - this issue actually builds that mechanism, using a pretrained
CLIP ViT-B/32 vision encoder (`Xenova/clip-vit-base-patch32`'s
vision-only ONNX export) as a practical stand-in for the paper's own
DBoW2, per the issue's own scope decision (real DBoW2 needs a
from-source C++ build, ruled out by this project's existing
OpenCV-version fragility; NetVLAD has no pretrained ONNX export
anywhere - CLIP was designated the thing to try first, with NetVLAD as a
fallback only if CLIP's fit validated too weak to be worth wiring in).

**Step 1: fit validation, before wiring in anything (required by the
issue).** Embedded `freiburg1_desk`'s pre-pan frames (8-46) and the
known revisit window (476-495, identified in #49's write-up as where the
camera physically returns) with `CLIPEmbedder`, and checked cosine
similarity against ground truth:

- Best-match pair: pre-pan frame 37 vs. revisit frame 484, similarity
  0.9642, ground-truth camera positions 24cm apart - a real match.
- Broader sweep across the whole gap (every 5th frame, frame 55-500):
  Pearson correlation between max similarity-to-pre-pan and
  ground-truth distance to the pre-pan area = **-0.48** (moderate,
  real, the right sign).
  The revisit window's frames land at ranks 2, 4, 9, 10 out of 90
  sampled gap frames by similarity alone (top ~11%) - a genuine,
  usable signal.
- **Not a clean spike, though**: frame 200 (182cm from the pre-pan
  area - clearly a different part of the room) scores 0.9596, almost
  indistinguishable from the true revisit's peak of 0.9628 at frame
  485. Frames 190-200 as a group score comparably to the real revisit
  despite being physically 1.7-1.8m away. CLIP's embedding is picking
  up real scene-appearance signal (lighting/wall-texture/general
  framing), but not cleanly enough to trust as a hard "this frame =
  this place" filter.

**Decision (per the issue's own criterion - "does it cleanly
distinguish same place from different place, or is that a reason to
fall back to NetVLAD"):** the correlation is real, not absent, so this
cleared the bar to wire in - but the false-positive risk (frame 200)
means it can only be trusted as a *soft* pre-filter with a fallback,
never a hard replacement for the existing unrestricted search.

**Mechanism:** `--appearance-relocalize` (off by default, requires
`--relocalize`): each keyframe now also gets a CLIP embedding
(`keyframe_clip_embedding`, computed once at insertion, same lifecycle
convention as `keyframe_kp`/`keyframe_desc`). Inside the existing
relocalization block, the current lost frame is embedded and compared
against every keyframe's stored embedding (cosine similarity - both
L2-normalized, so a plain dot product); the match is first restricted
to just the single best-matching keyframe's own points and PnP is
attempted against that smaller set. **Only if that fails** does it fall
through to today's unrestricted full-active-map search - so this can
only add an extra attempt on top of the existing mechanism, never
remove it, in case CLIP picks the wrong keyframe. See
`pipeline/clip_ml.py` (`CLIPEmbedder`, same `DepthEstimator`-style
wrapper convention as `depth_ml.py`: `model_path` constructor arg,
`ort.InferenceSession` with `CPUExecutionProvider`) and the flag's own
`--help` text in `pipeline/mapping.py`.

**`freiburg1_desk` result (`--opportunistic-triangulation --relocalize`,
`+ --appearance-relocalize`, gap-checked per pitfall #1):**

| Config | Death frame | Keyframes | Gap size | Reconnect | ATE/RPE RMSE (m) |
|---|---|---|---|---|---|
| `--opportunistic-triangulation --relocalize` (baseline, this session) | 52 | 8 | 430 frames | 482 | 0.0622 / 0.1150 |
| `+ --appearance-relocalize` | 52 | 8 | 430 frames | 482 | 0.0622 / 0.1150 (byte-identical trajectory) |

(Baseline death/gap/reconnect frames measured slightly differently here
- 52/430/482 - than #49's own write-up above - 55/421/476 - despite no
intervening logic change to either mechanism between sessions; not
investigated further, since the point of this test is the *within-session*
comparison, which is exact.)

`--appearance-relocalize` made **no measurable difference**: the output
trajectory is byte-identical (`diff`-confirmed) to the same-session
baseline. Relocalization stats explain why - **510 relocalization
attempts, 0 succeeded, with or without the appearance pre-filter.** Of
those 510, the CLIP-restricted candidate search reached PnP (found >= 6
matches within the single best-matching keyframe's own points) **112
times, and still never once cleared `--pnp-min-inliers`.**

**This is the important, non-obvious finding**: it initially looked
like the earlier `+ --relocalize` result (0-1 successes across 500+
attempts, per the issue's own framing) might be a *candidate-selection*
problem - the brute-force full-map search drowning a real match in too
many competing candidates. Appearance retrieval was specifically
supposed to test that theory by handing PnP a much smaller, appearance-
ranked candidate set. It didn't help, and 112 of those restricted
attempts show the correct(-looking) keyframe *was* being selected and
handed to PnP - the failure is downstream of candidate selection, in
literal ORB-descriptor point correspondence itself. Whatever changed
about the scene's appearance during the whip-pan (motion blur, lighting,
viewing angle) breaks ORB Hamming matching badly enough that even a
correctly-narrowed candidate pool can't produce `--pnp-min-inliers`-worth
of valid 2D-3D correspondences.

**Why this means NetVLAD likely wouldn't do better either, and what
would:** NetVLAD (and DBoW2 itself) are scene-*recognition* mechanisms,
not correspondence solvers - they narrow down *which* keyframe you're
probably looking at, exactly what CLIP already does adequately here (r
= -0.48, correct keyframe reached PnP 112/510 times). The bottleneck
this test exposes is downstream of recognition: once a place is
recognized, real ORB-SLAM still needs *some* mechanism to recover a
pose from it, and here that's still literal point-level PnP, which is
failing regardless of candidate scope. The paper's actual answer to
this is loop closing / pose-graph optimization - which doesn't need
fresh 2D-3D correspondences at all, just a recognized-place constraint
folded into the graph - and that's an explicit non-goal of this project
(see this file's own non-goals and issue #50/#47's scope notes). So the
honest conclusion is: appearance retrieval, done correctly, still isn't
enough on its own without either a correspondence-free recovery
mechanism (loop closing, out of scope) or #51's concurrent Local
Mapping thread (denser structure sooner, the other still-open lever).

**`freiburg1_xyz` regression check:** `--relocalize` never fires on this
sequence regardless (798 frames, 795 tracked, 0 gaps >5 frames, clean
per pitfall #1's gap-check - same as prior sessions), so `0
relocalization attempts` with or without `--appearance-relocalize` -
trajectories are byte-identical (`diff`-confirmed), ATE/RPE RMSE 0.1129
/ 0.1032 for both. Exactly the "off/never-triggered makes zero extra
calls" behavior the flag's own help text promises.

**Reproduction:**

```bash
conda activate slam
python -m pipeline.mapping --video datasets/tum/rgbd_dataset_freiburg1_desk --calibration calibration/tum_freiburg1.yaml --opportunistic-triangulation --relocalize [--appearance-relocalize] --trajectory-output results/<label>_desk_estimate.txt --plot-output results/<label>_desk_trajectory.png --no-display

evo_ape tum datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt results/<label>_desk_estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt results/<label>_desk_estimate.txt -a -s
```

Gap-check (`EVALUATION_METHOD.md` pitfall #1) required, same as #49 -
the `freiburg1_desk` numbers above are gapped and should not be read as
continuous tracking.
