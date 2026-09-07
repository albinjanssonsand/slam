# Evaluation Method

Read this before adding a new section to `EVALUATION_RESULTS.md`, and before
trusting any number already in it. This file holds the reusable method and
known pitfalls; `EVALUATION_RESULTS.md` holds the actual results log
(append-only, one section per pipeline version/issue - a regression or
improvement is a diff between sections there).

## Method

`pipeline.mapping --trajectory-output` writes each accepted keyframe's pose
in TUM format; `evo_ape`/`evo_rpe ... -a -s` (Sim(3) alignment - required
since this pipeline's monocular scale isn't metric) score it against each
sequence's `groundtruth.txt`. `scripts/compare_trajectories.py` compares two
estimates against one ground truth at once - use it once a change needs
plotting against a prior baseline, rather than re-deriving numbers by eye.

**Run `conda activate slam` (or invoke that env's `python.exe` directly)
first - see pitfall #4 below before running any of this on a bare `python`
from `PATH`.**

**Reproduction** (for any sequence `<seq>` in `xyz`, `desk`, `room`, `rpy` -
substitute the right `freiburg1`/`freiburg2`/`freiburg3` prefix and
`calibration/tum_freiburg<N>.yaml` for sequences outside `freiburg1`):

```bash
python -m pipeline.mapping --video datasets/tum/rgbd_dataset_freiburg1_<seq> --calibration calibration/tum_freiburg1.yaml --trajectory-output results/<label>_<seq>_estimate.txt --plot-output results/<label>_<seq>_trajectory.png --no-display

evo_ape tum datasets/tum/rgbd_dataset_freiburg1_<seq>/groundtruth.txt results/<label>_<seq>_estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg1_<seq>/groundtruth.txt results/<label>_<seq>_estimate.txt -a -s
```

`<label>` should identify what's being tested (e.g. the issue/branch or the
specific flags), since `results/` accumulates output across many
investigations and file names are the only thing distinguishing them.

## The "too hard" criterion

A dataset is too hard if the pipeline never produces a usable trajectory
(crash, or too few keyframes for `evo`'s Umeyama fit), **or** if it does
produce ATE/RPE but at low **trajectory coverage** (estimate duration /
ground-truth duration, from each file's first/last timestamp - read column 1
of the estimate and of `groundtruth.txt`, divide; no script computes this,
do it by hand). A run that loses tracking early just stops producing
keyframes, and `evo`'s Sim(3) alignment then fits well to a small,
temporally-clustered handful of early poses - a deceptively low ATE for a
trajectory that covers almost none of the real motion. Coverage catches
this; ATE/RPE alone don't.

## Known pitfalls (read before reporting a "coverage" or "ATE" number as an improvement)

### 1. Coverage (first/last keyframe timestamp) is blind to gaps in between

**This is the pitfall that produced a real, shipped-then-reverted false
result** (the keyframe-insertion-threshold-tuning investigation on
`freiburg1_desk` - see `EVALUATION_RESULTS.md`'s corrected writeup). Coverage
as defined above only looks at the *first* and *last* keyframe's timestamps -
it has no idea whether anything useful happened in between. A run that:

1. tracks normally for a while,
2. then loses tracking *completely* for hundreds of frames (a total,
   uninterrupted blackout - not just "no new keyframes", genuinely zero
   accepted poses of any kind),
3. then, purely because the camera happens to revisit somewhere it mapped
   early on, reconnects late in the sequence and produces a few more
   keyframes near the end,

reports a **high coverage number** identical in form to a run that tracked
continuously the whole time - even though the two are nothing alike. In the
incident that motivated this section, a parameter change was credited with
"fixing" `freiburg1_desk`'s tracking-loss death spiral (coverage 4.6% -> 73.8%)
when the actual, verified behavior was: dies at almost exactly the same frame
as before (a 3-frame difference - noise), stays completely dead for **444
straight frames**, then reconnects late only because a denser residual map
(an unrelated side effect of the same parameter change) happened to have
enough overlap with a later, coincidental revisit of the same physical desk
area (confirmed by checking ground-truth camera position at both points -
they were ~10cm apart). The 73.8% coverage number was real and reproducible,
but the story behind it - "fixes the whip-pan" - was false.

**Before reporting any coverage number as a genuine tracking improvement,
check for internal gaps.** Every frame that gets an accepted pose (keyframe,
frame-only TRACK, RELOCALIZED, or ROTATION-ONLY FALLBACK) prints one `frame
N: <STATUS> ...` line to the run log; a frame with no such line was lost
entirely. Extract every printed frame index and look at the gaps between
consecutive ones:

```python
import re
frames = []
with open("results/<run>.log") as f:
    for line in f:
        m = re.match(r"frame (\d+):", line)
        if m:
            frames.append(int(m.group(1)))

gaps = [(frames[i], frames[i+1], frames[i+1] - frames[i]) for i in range(len(frames) - 1)]
big_gaps = [g for g in gaps if g[2] > 5]  # tune the threshold to the sequence's frame rate
print(f"{len(frames)} frames tracked out of a {frames[-1] - frames[0]}-frame span")
print("large gaps:", big_gaps)
```

A handful of small gaps (a few frames, self-recovering) is normal and not a
concern. One or more gaps of tens-to-hundreds of frames means the headline
coverage number is not describing continuous tracking, and needs the honest,
more detailed story (see the incident writeup above for the tone/format this
should take) - not just the single percentage.

If a large gap's endpoint looks like a possible revisit, verify it against
ground truth directly rather than guessing - a short script comparing
camera position (translation columns of the nearest `groundtruth.txt` row to
each frame's timestamp) at both ends of the gap settles it either way, same
as the incident above.

### 2. ATE looks best exactly when coverage is lowest

A trajectory that dies almost immediately, near its start point, aligns
(Sim(3)/Umeyama) very well to that same small region of ground truth - producing
a *lower*, better-looking ATE than a longer, harder trajectory that actually
covers real motion and therefore accumulates real error. Never read ATE/RPE
without also reporting coverage next to it; a large ATE improvement paired
with a coverage collapse is not an improvement.

### 3. OpenCV's RANSAC state is process-global and unseeded

`cv2.findHomography`/`findFundamentalMat`/`findEssentialMat`/
`solvePnPRansac` all draw from the same global RNG, not a call-local one.
Any extra or removed RANSAC call anywhere earlier in a run (a relocalization
attempt, an extra bootstrap trial, a rotation-only-fallback attempt, even one
that's ultimately rejected) shifts every later RANSAC-based estimate for the
rest of that run away from a run that didn't make that call - a real,
measured effect on outcomes many frames later, not just a performance cost.
This means two runs of the *exact same command* can legitimately produce
different results. When a result matters (e.g. justifying a default change),
prefer a same-environment before/after pair run back-to-back over trusting a
single run's number, and treat small differences between repeated runs as
possible RNG noise rather than a real effect - but see the pitfall above
first: a *large* difference (like the desk incident) usually has a real,
findable cause, not just noise, and deserves the gap-check treatment before
either explanation is accepted.

### 4. Verify you're actually running what you think you're running

Confirm, before trusting any run: the working tree is on the commit you
think it is (`git log --oneline -1`), and the Python interpreter actually
being invoked is the project's real environment, not a coincidentally-
compatible one found earlier on `PATH` (check `python -c "import sys;
print(sys.executable)"` and compare package versions - `cv2.__version__` in
particular, since a different OpenCV major version can change RANSAC/solver
behavior, not just performance). This is not hypothetical, and has happened
**twice**: a full session's worth of evaluation runs was once executed
against an unrelated Windows Store Python install with a different OpenCV
major version (4.13 vs. the project's actual 5.0), silently, because both
happened to have the project's dependencies importable - that first time,
the numbers turned out to still hold once re-run in the real environment
(verified after the fact, not guaranteed). It recurred during
[#47](https://github.com/albinjanssonsand/slam/issues/47)'s investigation,
and that time it did NOT just hold: on `freiburg1_desk`, both environments
bootstrap identically (frame 7, model=F, R_H=0.39, same inlier/point
counts) but diverge partway through per-frame PnP tracking - OpenCV 4.13
survives 5 frames longer (to frame 45, 1 more keyframe) than 5.0's frame 40
death point, a real, reproducible, version-dependent difference in
`solvePnPRansac`'s behavior, not RNG noise (same-environment reruns of the
identical command are byte-identical). Every "current main" number in this
file predating #47's fix below should be assumed OpenCV-4.13-sourced unless
a section explicitly says otherwise.

**Concrete fix, checked in via [#47](https://github.com/albinjanssonsand/slam/issues/47):**
`requirements.txt` now pins exact versions matching this project's real
environment (a conda env named `slam`; `opencv-python==5.0.0.93`
specifically). Before running anything here:

```bash
python -c "import sys, cv2; print(sys.executable); print(cv2.__version__)"
```

and confirm the interpreter path points into that environment and
`cv2.__version__` starts with `5.0` - not just that the import succeeds.
`conda activate slam` (or invoking that env's `python.exe` directly, e.g.
`C:\Users\<you>\miniconda3\envs\slam\python.exe`) before any command in
this file's Method/Reproduction sections is required, not optional; bare
`python`/`pip` on `PATH` cannot be trusted to resolve there.
