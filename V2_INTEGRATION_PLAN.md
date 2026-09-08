# v2 Integration Plan: Fusing ML Depth Into the Tracked Map

This supersedes the previous version of this document. The actual work is
now tracked as GitHub issues (#6-#10) so each phase can be picked up
independently by a fresh `implement-issue` session; this file is the
one-page summary of the reasoning behind them.

## Current state

`pipeline/depth_ml.py` has a working `DepthEstimator` (Depth Anything V2
Small, ONNX, CPU) plus `fit_disparity_scale_shift`, which calibrates the
model's relative disparity against the map's own confirmed 3D points.
`pipeline/mapping.py`'s `_run_depth_densify` already runs this at every
accepted keyframe — but its own docstring says the result is **plotting
only**: "the caller must not feed the returned points into
sparse_map/PnP/BA." Nothing ML-derived currently affects tracking,
triangulation, point confirmation, or pose acceptance. NOTES.md's own v2
section already lays out roughly the right order to change that (bootstrap
scale → triangulation gaps → point confirmation → pose plausibility), tied
to specific code in `mapping.py`/`triangulation.py`.

**Baseline to build against ([#58](https://github.com/albinjanssonsand/slam/issues/58)):**
`main`'s default flags cost ~15-20 min/798 frames on `freiburg1_xyz` for
machinery (#21/#22/#24) whose accuracy payoff lands almost entirely on
`freiburg1_desk`/`freiburg1_room`/`freiburg2_pioneer_slam2`, not on the two
ground-truth sequences this comparison actually runs on. Use
`--orb-single-pass --essential-only-bootstrap --single-keyframe-point-creation`
instead: 4m48s on `freiburg1_xyz` (68 keyframes, 86.6% coverage, ATE/RPE
RMSE 0.0317/0.0302 - notably *better* than the full-machinery default's
0.0702/0.0898, an unexplained but real result, see `EVALUATION_RESULTS.md`'s
`#58` section) and 23m50s on `freiburg2_xyz` (227 keyframes, 99.6% coverage,
ATE/RPE RMSE 0.1015/0.0329). #7's own scope (closed `NOT_PLANNED` when the
#19 paper-alignment sprint took priority) should be re-opened against this
baseline, not the pre-#58 default-flags one.

## Blockers/dependencies before ML depth can actually help

1. **The model is relative depth, not metric.** Depth Anything V2 *Small*
   (the checkpoint actually in the repo) outputs disparity-like values with
   no real-world units — confirmed by `depth_ml.py`'s own docstrings. It
   cannot independently fix monocular scale ambiguity. The existing
   scale-fit also calibrates ML disparity *against the map's own
   already-established (arbitrary, drifting) scale* — circular for
   "anchoring" scale, and structurally unusable at/near bootstrap since
   there are no confirmed points yet to calibrate against. Real metric
   anchoring needs either a metric-depth checkpoint swap or an external
   reference (none exists yet; IMU is planned for v3). Scoped as an
   investigation (#10) rather than a blind implementation.
2. **CPU inference is slow (~1.3s/frame).** Fine paid once per sparse
   keyframe (already how `_run_depth_densify` uses it) — every phase must
   keep it that way, never move it into the per-frame loop.
3. **Nothing is measured yet.** NOTES.md's own history shows "looks
   plausible" is not the same as "actually helps" (e.g. the point-culling
   experiment that worked exactly as designed but didn't fix the real
   bottleneck, and was reverted). A baseline-vs-ML-fusion comparison tool
   (#6) has to exist before any fusion change can claim an improvement.
4. **Model checkpoint + TUM datasets are gitignored/local-only**, not
   committed to the repo. `implement-issue` runs in this same checkout
   (not an isolated worktree), so that's fine for local work — but these
   issues can't be handed to a remote/isolated-worktree agent without
   first getting that data there.

## Issues

| # | Title | Depends on | Summary |
|---|---|---|---|
| [#6](https://github.com/albinjanssonsand/slam/issues/6) | Build an ML-vs-geometric trajectory comparison workflow (Phase 0) | — | Comparison script: two TUM trajectories + ground truth → overlaid aligned plot + ATE/RPE for both. Required before any fusion change can be judged. |
| [#7](https://github.com/albinjanssonsand/slam/issues/7) | Fuse ML depth into triangulation gaps as provisional map points (Phase 2) | #6 | Feed ML-depth-backprojected points (from pixels geometric triangulation couldn't resolve) into `sparse_map` as **provisional** points, through the existing confirm/promote lifecycle — not trusted on arrival. |
| [#8](https://github.com/albinjanssonsand/slam/issues/8) | Corroborate provisional-point confirmation with ML depth agreement (Phase 3) | #6, #7 | When a provisional point's depth agrees with the ML fit, count it toward confirmation — targets the documented `--confirm-count 2` PnP-pool-starvation tradeoff without falling back to the weaker `--confirm-count 1`. |
| [#9](https://github.com/albinjanssonsand/slam/issues/9) | Add an ML-depth consistency check to pose-plausibility gating (Phase 4) | #6, #7, #8 | Reject a PnP/BA pose if its implied depths disagree with the ML depth map, as an *additional* guard alongside `--max-plausible-rotation`/`--max-step-ratio`. Riskiest phase — sequenced last, only after the depth fit's reliability is proven by #7/#8. |
| [#10](https://github.com/albinjanssonsand/slam/issues/10) | Investigate ML-depth scale-drift regularization across a sequence (Phase 1, exploratory) | #6 | Research spike, not an implementation: does periodic re-calibration reduce the long-sequence drift NOTES.md documents? Is a metric-depth checkpoint swap worth it? No pipeline code changes — a NOTES.md write-up with a recommendation. |

## Recommended order

```
#6 (harness)
 ├─→ #7 (map fusion) ─→ #8 (confirmation) ─→ #9 (pose gating)
 └─→ #10 (scale investigation, runs independently, doesn't block #7-#9)
```

Every fusion issue (#7-#9) is required to report an actual ATE/RPE delta
via #6's harness, not just "it ran" — matching how NOTES.md already treats
"looks right" as insufficient evidence on its own.
