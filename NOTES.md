# Phone-Camera SLAM Pipeline — Project Notes

Plan, design decisions, and a running log of problems encountered and fixed.

## Goal

Build a monocular visual SLAM pipeline using an Android phone as the camera,
tethered by USB cable to a laptop, with the core pipeline in Python.

Three planned versions:

- **v1** — Classical: ORB feature tracking + geometric triangulation (structure from motion).
- **v2** — Add ML-based monocular depth estimation to improve scale/robustness.
- **v3** — Fuse the phone's IMU to improve tracking through fast motion, low-texture
  scenes, and to help resolve scale drift.

## Recorded video vs. real-time video stream

**Starting with a recorded video, not live streaming.**

Reasons:

- Recorded footage is a fixed, replayable dataset — you can re-run the exact same
  frames while iterating on feature matching, pose estimation, and triangulation,
  without the added noise of variable USB latency, frame drops, or autofocus/exposure
  hunting.
- It decouples two hard problems. Getting frames off the phone reliably (USB video
  streaming, format/resolution negotiation, timestamp sync with IMU) is its own
  integration problem, separate from getting the SLAM math right. Debugging both
  simultaneously makes it unclear which one is broken when something drifts or crashes.
- Camera calibration only needs to be done once against recorded checkerboard footage,
  and can be reused for both offline and live pipelines.
- The frame-processing core (ORB → matching → pose → triangulation → map) is written
  as a pipeline stage that consumes frames from an iterator/generator. A video file
  and a live capture device can both feed that same iterator — so moving to real-time
  later is a matter of swapping the frame source, not rewriting the pipeline.

Once v1 tracks cleanly and drift is acceptable on recorded clips, switch the frame
source to live capture (still useful to keep the recorded-video mode around for
regression testing/debugging).

## Phone → Laptop video capture

Options, roughly in order of setup simplicity:

1. **USB webcam mode via scrcpy / `droidcam` / native "USB video"** — some Android
   phones support UVC (USB Video Class) output directly, or via apps
   (e.g. DroidCam, IP Webcam over USB-forwarded ADB, or `scrcpy --v4l2-sink` on Linux).
   On Windows, DroidCam or an equivalent that exposes a virtual camera / capture card
   is the simplest route — OpenCV then just opens it as a regular `cv2.VideoCapture` index.
2. **ADB + `scrcpy`** for screen mirroring/control, with a separate video capture app
   feeding a virtual camera device.
3. **IP camera over USB-tethered network** (phone shares network over USB, laptop
   pulls an MJPEG/RTSP stream from an app like IP Webcam). Adds encoding latency but
   is easy to set up cross-platform, and can also carry IMU data over the same link.

Since IMU access (v3) will be needed eventually, evaluate apps that can stream both
camera *and* IMU (accelerometer/gyro) over the same channel (e.g. IP Webcam's sensor
endpoints), so the capture path doesn't need to be redesigned when v3 starts.

## Camera calibration

Required before any triangulation will produce metric-consistent results.

- Print a checkerboard (or use a charuco board) pattern.
- Record a calibration video with the phone, extract frames, run OpenCV's
  `cv2.calibrateCamera` to get intrinsics matrix `K` and distortion coefficients.
- Store calibration as a config file (e.g. `calibration/phone_camera.yaml`) loaded
  by the pipeline. Redo if the phone's camera app/resolution changes.

---

## Version 1 — Classical ORB + Geometric SLAM

**Approach:** monocular visual odometry with keyframe-based mapping.

Pipeline stages:

1. **Frame source** — abstract iterator; recorded video first, live capture later.
2. **Preprocessing** — undistort using calibration, optionally resize/grayscale.
3. **Feature detection & description** — ORB (via OpenCV, `cv2.ORB_create`).
4. **Feature matching** — brute-force Hamming or FLANN-LSH matcher between
   consecutive frames (and against keyframes for loop closure later).
5. **Pose estimation, bootstrap-then-track**:
   - **Bootstrap (once)** — essential matrix (`cv2.findEssentialMat`) + `recoverPose`
     between the first keyframe pair only, to seed the map with an initial (arbitrary
     but fixed) scale.
   - **Track (every keyframe after)** — match against the *existing* 3D map (not
     just the previous frame) and solve pose via PnP (`cv2.solvePnPRansac`) against
     those already-scaled map points. This is the part that matters: chaining
     independent two-view essential-matrix estimates keyframe-to-keyframe compounds
     scale drift, because `recoverPose` returns a unit-length translation with no
     memory of the previous segment's scale — confirmed empirically (a static flat
     surface visibly receded further with every new keyframe under pure two-view
     chaining). Solving pose via PnP against the persistent map instead keeps every
     new keyframe's pose in the map's own established scale.
6. **Triangulation** — `cv2.triangulatePoints` for genuinely *new* map points seen
   at a keyframe (matched but not yet in the map), using the PnP-derived pose so new
   structure joins the map's existing scale rather than starting a fresh one. Reject
   points with too little parallax angle between viewing rays (numerically unstable,
   "sprinkler" artifact) and points that fail the cheirality check.
7. **Keyframe selection** — insert a new keyframe once accumulated parallax (median
   matched-point pixel displacement vs. the current reference) crosses a threshold,
   rather than attempting pose estimation on every consecutive frame — adjacent
   frames at typical frame rates rarely have enough baseline for a well-conditioned
   essential matrix / PnP solve.
8. **Local map / bundle adjustment** — a lightweight
   local bundle adjustment (e.g. via `scipy.optimize.least_squares` or `g2opy`/
   `gtsam` bindings) to jointly refine recent keyframe poses and map points. This is
   also the natural way to reduce per-point triangulation noise when only a small
   parallax angle was available (average out noise across many observations of the
   same map point instead of trusting a single two-view triangulation).
9. **Visualization** — 3D point cloud + camera trajectory (e.g. `open3d` or
   `matplotlib` 3D, or a live viewer if real-time).

Known limitations to accept in v1: no loop closure, fragile in low-texture or
fast-rotation scenes, and monocular scale is arbitrary (fixed by the bootstrap step,
not metric). This motivates v2/v3.

### v1 Components / Libraries

- `opencv-python` (or `opencv-contrib-python` for ORB + extra matchers)
- `numpy`
- `scipy` (least-squares refinement)
- `open3d` (point cloud + trajectory visualization) — optional
- A video capture bridge (DroidCam or similar) for the eventual live-source switch

---

## Version 2 — ML Monocular Depth Estimation

**Goal:** improve scale consistency and provide dense-ish depth to complement the
sparse geometric point cloud, especially in low-texture regions where ORB fails.

- Integrate a pretrained monocular depth model (e.g. MiDaS, Depth Anything, or
  ZoeDepth) via `torch`/`onnxruntime` to produce a per-frame depth map.
- Use the ML depth to:
  - Resolve/anchor the unknown monocular scale (ML depth models trained on metric
    data, or scale-aligned against the sparse geometric points via least-squares).
  - Densify the sparse map for better visualization and obstacle/surface understanding.
  - Provide fallback depth in regions with insufficient feature matches.
- Fuse ML depth with geometric triangulation depth (e.g. weighted by triangulation
  confidence/parallax) rather than replacing the geometric estimate outright.

**Why this helps, and what it changes in v1:** v1's remaining weaknesses mostly trace back to one root cause: geometric triangulation only
produces a point where two keyframes see it from a large-enough baseline angle,
and has no path to real-world scale on its own. A dense, per-pixel ML depth
estimate doesn't depend on feature matches or baseline at all, so it attacks
that root cause directly:

- Monocular scale is arbitrary, fixed once by the two-view bootstrap and never
  metric (v1's known limitations) - an ML depth model trained on real depth
  data gives an actual real-world reference to anchor against, instead of an
  arbitrary one.
- Low-texture backgrounds and the min-triangulation-angle "sprinkler" rejection
  (`pipeline/triangulation.py`) both stem from needing enough matched-feature
  baseline to triangulate at all - dense depth exists independent of that, so
  it can fill exactly the regions geometric triangulation is weakest in or must
  reject.
- The provisional/confirmed point lifecycle (`mapping.py`) exists to guard
  against trusting a single two-view triangulation that could be biased - an
  ML depth estimate for the same pixel is a second, independent source of
  truth that could corroborate a new point immediately, rather than waiting on
  `--confirm-count` future re-observations.
- The heuristic PnP-pose plausibility guards (`--max-plausible-rotation`,
  `--max-step-ratio`) exist only because there's currently no independent check
  on a solved pose besides reprojection error against the map itself - a dense
  depth map is an outside signal that could eventually replace those
  heuristics with an actual consistency residual.

None of v1's stages need to be torn out for this - v2 is additive. Likely
affected parts, roughly in the order they'd actually change:

1. **Bootstrap scale** - swap the arbitrary two-view scale for one anchored
   against ML depth.
2. **Triangulation gaps** (`pipeline/triangulation.py`'s parallax-angle
   rejection) - fill rejected/low-texture regions with ML depth instead of
   leaving them absent from the map.
3. **Point confirmation** (`Map`'s provisional/confirmed lifecycle) - could
   shorten or skip the multi-observation wait once ML depth agrees with a
   fresh triangulation.
4. **Pose plausibility checks** (`--max-plausible-rotation`/`--max-step-ratio`)
   - the most invasive, longest-term change: replace the ad hoc thresholds
   with a real depth-consistency check.

### v2 Additional Components

- `torch` + a pretrained depth model checkpoint (or `onnxruntime` for a lighter
  deployment), GPU strongly recommended for real-time-ish inference
- Depth alignment/scale-fitting utilities (`numpy`/`scipy`)

---

## Version 3 — IMU Fusion

**Goal:** use the phone's accelerometer/gyroscope to improve robustness through
fast motion and reduce reliance on visual tracking alone (visual-inertial odometry).

- Stream IMU data from the phone (many camera-streaming apps expose an IMU/sensor
  endpoint alongside video; alternatively a small companion Android app publishing
  sensor data over the same USB/network link).
- Time-synchronize IMU samples with video frames (hardware/software timestamp
  alignment — likely the trickiest part of v3).
- Implement or integrate a filter/estimator to fuse visual pose estimates with IMU
  preintegration:
  - Simplest: complementary or Kalman filter fusing visual pose + IMU-integrated
    motion for smoothing/gap-filling when visual tracking is briefly lost.
  - More robust: full visual-inertial odometry (VIO) formulation with IMU
    preintegration between keyframes, jointly optimized with visual bundle
    adjustment (e.g. via `gtsam`'s IMU preintegration factors).
- IMU also helps resolve monocular scale ambiguity more directly than ML depth alone,
  since accelerometer data has physical units.

### v3 Additional Components

- IMU data source (sensor-streaming app or companion app) with timestamps
- Time synchronization logic (video frame timestamp ↔ IMU sample timestamp)
- `gtsam` (or manual EKF/complementary filter implementation) for sensor fusion

---

## Suggested Repo Structure

```
slam/
  calibration/
    calibrate.py
    verify_undistort.py
    phone_camera.yaml
    data/                  # checkerboard calibration videos + undistort check images
  recordings/              # SLAM test/demo recordings (not calibration-specific)
  capture/
    video_source.py       # recorded-file + live-capture frame iterators (common interface)
    imu_source.py         # v3
  pipeline/
    features.py           # ORB detect/describe/match
    pose.py                # essential matrix / PnP pose estimation
    triangulation.py
    mapping.py             # keyframes, map points, PnP tracking
    bundle_adjustment.py   # local sliding-window bundle adjustment
    depth_ml.py            # v2
    fusion.py               # v3
  viz/
    plot3d.py
  main.py                  # wires frame source -> pipeline -> viz
```

## v1 Development log

- [x] Camera calibration (`calibration/calibrate.py`) + undistortion sanity check
  (`calibration/verify_undistort.py`). Problems hit along the way:
  - OpenCV 5.0's Python bindings changed array shape/type for `cornerSubPix`/
    `projectPoints`, breaking `cv2.norm`'s strict type matching in the
    reprojection-error calculation — fixed by computing it with plain NumPy instead.
  - First calibration attempt produced a wildly out-of-range k3 distortion
    coefficient (-4.15; normal range is roughly 0.01-1) from a checkerboard video
    that didn't cover the frame edges/corners well enough, which extrapolates badly
    once undistorted (visible as severe warping near the image edges). A first
    attempt to visually verify this was itself misleading — a debug grid drawn onto
    the image *before* undistorting got warped by `cv2.undistort` along with the
    photo, which looked like lens distortion but was actually an artifact of the
    verification tool. Fixed the tool (draw the grid independently on each panel,
    after undistorting) and re-verified against the checkerboard's own real straight
    edges, which confirmed the calibration itself was the problem. Recapturing with
    full-frame coverage (including corners/edges) fixed it: k3 dropped to ~1.08, a
    believable magnitude, with reprojection error of 0.2px.
- [x] `capture/video_source.py` — calibrated, undistorted frame iterator over a
  recorded video (swappable for a live source later). The standalone playback demo
  initially ran at decode speed instead of real time (`cv2.waitKey(1)` doesn't pace
  to the video's actual frame rate) — cosmetic, fixed for the demo viewer only,
  since the real pipeline should consume frames as fast as it can rather than
  throttled to real time.
- [x] `pipeline/features.py` — ORB detect/match with Lowe's ratio test.
- [x] `pipeline/pose.py` — two-view pose estimation (essential matrix +
  `recoverPose`). First attempt estimated pose between every *consecutive* video
  frame and failed roughly half the time (423 of 837 frame pairs) despite healthy
  match counts (1000+) — diagnosed as a small-baseline degeneracy: adjacent frames
  in a 30fps clip have too little real camera motion between them for a
  well-conditioned essential matrix (median rotation between "successful" pairs was
  often under 1°). This motivated keyframe selection gated on accumulated parallax
  (median matched-point pixel displacement vs. a held reference) instead of
  attempting pose estimation on every frame, which fixed it.
- [x] `pipeline/triangulation.py` — cheirality + parallax-angle filtering. Naive
  triangulation initially produced a "sprinkler" artifact: points near the
  direction of camera travel are seen from nearly the same angle in both views, so
  triangulating them is numerically unstable and throws them out to arbitrary
  distances along the viewing ray. Fixed by rejecting points below a minimum
  parallax angle. Even after that fix, an early simple validation test (near
  object + far background, sliding sideways) produced a "smeared" point cloud with
  no clean separation between the two depths. Root cause was that the *real*
  camera baseline per keyframe pair was still too small relative to the scene's
  depth for the surviving points, so ordinary feature-matching noise was enough to
  blur what should have been two tight clusters into a continuous spread. Fixed by
  requiring much more accumulated parallax before triggering a keyframe (a bigger
  real baseline), which validated cleanly: two correctly-ordered, tightly separated
  near/far depth clusters.
- [x] Confirmed empirically (not just in theory) that chaining independent two-view
  poses keyframe-to-keyframe drifts in scale which motivates the mapping stage below.
- [x] `pipeline/mapping.py` — persistent `Map` (3D points + one ORB descriptor each),
  bootstrapped once via two-view pose+triangulation, then extended via PnP
  (`cv2.solvePnPRansac`) tracking for every subsequent keyframe, per the
  bootstrap-then-track design above. Fixed the earlier scale-drift symptom (a
  static flat surface no longer recedes with every new keyframe).
- [x] **New problem found**: on longer straight-line recordings, the reconstructed
  trajectory came out clearly wrong even though this fixes scale drift. Diagnosed via
  the PnP inlier *ratio* (inliers / total map matches) per keyframe, which declined
  steadily over the sequence (98% → 97.5% → 92% → 94% → 91% → 83% → 57% → 55%) while
  the raw inlier count looked superficially healthy throughout. Root cause: every new
  keyframe's map points are triangulated using *that keyframe's own just-solved PnP
  pose*, and nothing ever revisits a point once it's added. Small pose errors
  therefore get baked permanently into the map's 3D structure instead of staying
  transient, and later keyframes must reconcile a pose against an increasingly
  internally-inconsistent map (older points vs. newer points), showing up as more
  and more "inliers" actually being rejected by PnP RANSAC. This is a different,
  arguably worse failure mode than the old pose-only scale drift, since errors are
  now baked into permanent structure rather than just the pose chain. Local bundle adjustment might correct this to correct.

- [x] `pipeline/bundle_adjustment.py` — sliding-window local bundle adjustment
  (`scipy.optimize.least_squares`), jointly refining recent keyframe poses and the
  map points they observe to minimize total reprojection error, wired into
  `mapping.py` after each accepted keyframe. Several problems surfaced and were
  fixed while building this:
  - **Froze/crawled on the very first bootstrap.** scipy's default dense
    finite-difference Jacobian re-evaluates the full residual vector once per
    parameter; with 500+ point variables that's extremely expensive. Fixed by
    supplying an explicit sparsity pattern (which parameters affect which
    residuals - each observation only touches its own keyframe's pose and its
    own point) so the sparse-aware `trf` solver can perturb many independent
    columns at once.
  - **Still visibly lagged at every keyframe.** Added `--ba-every` (skip BA on
    some keyframes) and `--ba-max-points` (cap points touched per call), and
    loosened convergence tolerances/iteration cap - this refines an
    already-good PnP/triangulation estimate, not a from-scratch solve, so it
    doesn't need to grind toward high-precision convergence.
  - **Naive point capping caused trajectories jump**
    Capping to the globally most-recent N points could strip a keyframe's
    observations down to almost nothing if it mostly re-observed old map
    points rather than contributing new ones (the normal steady-state TRACK
    case). An under-constrained pose (too few residuals for its 6 DOF) is free
    to be pushed to a wild value by the optimizer. Fixed by guaranteeing a
    minimum floor of retained observations per keyframe before applying the
    global recency cap.
  - **Trajectory would jump way off, then self-correct over later keyframes.**
    A single bad correspondence (false match) can dominate a plain
    least-squares solve since squared error lets one large residual pull the
    whole solution toward accommodating it; more data later outweighs and
    corrects it, but the bad state was still visible in the meantime. Fixed
    with a Huber robust loss, which caps any single residual's influence
    instead of letting it dominate.
- [x] **Third problem found, and the most significant**: even with working BA,
  some straight-line recordings still produced a smoothly, *consistently* wrong
  (curving, not jittery) trajectory - and critically, the PnP inlier ratio decay
  was nearly identical with or without BA, proving BA wasn't the differentiator.
  Diagnosed as a systematic bias, not noise: the fixed global ORB feature budget
  (`nfeatures=2000`) was dominated by whichever region had the strongest texture
  (a near foreground object), starving the background of features from the very
  start (confirmed by inspecting the recording: a lot of ORB keypoints in the
  foreground but not the background).
  As the foreground left the frame over the course of the slide, tracking was
  forced onto an increasingly thin set of background points - which, as
  established earlier, inherently has smaller parallax angle for the same
  baseline and is noisier to triangulate. That's a real geometric weakness BA
  faithfully reproduced rather than corrected.
- [x] Fixed via `pipeline/features.py`'s `detect_and_compute_gridded`: splits each
  frame into a grid (default 4x4) and enforces a feature quota per cell,
  guaranteeing spatial coverage regardless of texture disparity - standard
  practice in real SLAM systems for this exact reason. Its first version had its
  own bug: hard-cropping each cell truncated the descriptor patch context for
  keypoints near cell boundaries, producing corrupted descriptors - fewer
  matches, and non-horizontal "false" matches on a recording that was pure
  lateral motion (a good visual tell that a match must be wrong). Fixed by
  cropping each cell with a padding margin and only keeping keypoints that fall
  inside the true (unpadded) cell bounds.
- [x] Validated across multiple demo recordings at the original default settings
  (`--min-parallax 30`, `--ratio 0.75`) - trajectories track straight, point
  clouds separate into correct near/far depth clusters, and the PnP inlier ratio
  stays healthy across full sequences instead of decaying.

### Further improvements trajectory, map-quality, and pose-ambiguity fixes

A longer demo recording with a non-straight path (moving past several objects, turning corners) surfaced more issues worth recording.

- [x] **Rotation combined with translation still broke tracking**, even though
  pure-rotation handling (below) worked. Root cause: our keyframe/triangulation
  gating uses raw 2D pixel displacement as a proxy for "enough real camera
  translation to triangulate reliably" - but rotation alone produces large pixel
  displacement with zero real parallax (every point shifts by the same amount
  regardless of depth), so the two are indistinguishable from pixel motion alone.
  - First fix: track a per-keyframe "yield rate" (new points that pass the
    parallax-angle filter, divided by candidates attempted). If a keyframe's
    yield rate is too low, don't advance the triangulation reference for the
    *next* attempt - hold it so real translational baseline keeps accumulating
    against the same still-good reference instead of resetting onto a string of
    rotation-only frames. Confirmed working via a `[reference held...]` log tag
    that engaged and released sensibly through rotation segments.
  - Remaining problem even with the hold: points near the direction of travel
    still collapsed close to the trajectory specifically when rotation and
    translation happened together (e.g. turning to track something while still
    walking forward). The per-keyframe yield-rate check is an *aggregate*
    statistic and can't catch this - most points in such a keyframe can have
    fine parallax (so the aggregate check passes, no hold triggered), while the
    specific points near the new (rotated) viewing axis have marginal,
    just-above-threshold parallax angles that are individually unreliable, more
    so than the same marginal angle would be in a pure-translation segment
    (rotation amplifies pose-error sensitivity for near-viewing-axis points).
  - Second fix: scale the required parallax angle up with how much rotation
    occurred in that specific keyframe transition (`--rotation-angle-penalty`,
    added to `--min-triangulation-angle` per degree of relative rotation between
    consecutive keyframes, computed from the PnP-derived poses directly).
  - Both fixes were kept; a `rotation=X.Xdeg` field was added to the TRACK log
    line for visibility.

- [x] **Tried and reverted: map point culling.** On a very long recording (800+
  keyframes), PnP inlier *count* (not just ratio) declined steadily from ~345
  down to ~20-30 - close to the acceptance floor. Implemented soft point
  deactivation (per-point outlier/inlier bookkeeping, cull points repeatedly
  matched-but-PnP-rejected) plus a hard cap on the active/matchable working set
  size (`--max-active-points`, deactivating the worst-scoring excess once
  exceeded). The cap worked exactly as designed - active count held flat at the
  configured ceiling even as total (ever-created) points kept climbing past
  14,000. **But the underlying inlier-count decline persisted regardless**, and
  crucially was largely uncorrelated with the logged rotation values - ruling
  out both "unbounded map growth" and "rotation" as the dominant cause on this
  recording. This pointed instead to accumulated drift from local-only bundle
  adjustment: BA's window anchor keeps sliding forward as the trajectory grows,
  with no mechanism tying the whole map back to one consistent global frame, so
  small per-window biases have nothing to cancel them out over a long enough
  sequence. Since culling didn't address the actual bottleneck, it was reverted
  to keep the codebase simple - decided to accept this as a known v1 limitation
  (below) rather than keep layering fixes at a problem the architecture can't
  fully solve without loop closure or periodic global BA.

Following up on the rotation+translation sprinkler problem: further logs showed a
single keyframe transition (the one with the largest rotation in a stretch) could
triangulate a huge one-shot batch of brand-new points (1000+) sharing that one
pair's pose estimate. If that pair's pose had any error, every point in the batch
inherited the same bias - invisible to per-point filters (angle, depth-ratio),
since a whole batch computed from one biased pose can look internally
self-consistent while still being collectively wrong.

- [x] Implemented a provisional/confirmed point lifecycle in `mapping.Map`: new
  points start unconfirmed, are excluded from PnP pose estimation (so a bad batch
  can't influence the pose that could end up confirming its own siblings), and
  only get promoted to confirmed after accumulating `--confirm-count` (default 2)
  independent re-observations (reprojecting within `--confirm-reproj-error` under
  a pose solved from confirmed points only). Live view and final plot both show
  confirmed (black) vs. provisional (orange) points, and a point visibly flips
  color the moment it's promoted.
- [x] **Removed `min_depth_baseline_ratio`** (the "reject implausibly close
  points" filter added earlier) - confirmed redundant by testing: independent
  reconfirmation is a strictly more general check, since it also catches whole-
  batch pose bias that a single-pair geometric heuristic structurally cannot.
- [x] **Kept `min_triangulation_angle`** (rejects implausibly *far*/sprinkler
  points) - reasoned this one is NOT redundant despite looking similar: a
  wrongly-far point is, by definition, one that barely shifts in the image even
  under a real viewpoint change (that's what low parallax means), so it can
  easily *pass* reprojection-based reconfirmation from a nearby second view.
  Independent re-observation is good at catching "too close" errors (which do
  shift a lot under small pose changes) but structurally weak at catching "too
  far" ones for the same reason they occurred in the first place.
- [x] Reviewed the rest of the fix history for redundancy with this new
  mechanism: grid-based ORB extraction (feature distribution), keyframe
  parallax-gating (timing), BA's Huber loss (individual bad correspondences
  within an otherwise-good window), BA performance/point-capping, and the
  calibration/capture-layer fixes are all solving different, non-overlapping
  problems and remain necessary.

Continued testing of `--confirm-count` and further erratic-trajectory reports
surfaced several more distinct problems in sequence - each one initially looked
like it might be the same root cause as the last, but turned out not to be.

- [x] **`--confirm-count 2` starves the PnP-eligible pool.** Requiring 2
  independent re-observations before a point counts as confirmed shrinks the
  pool PnP can match against at any moment - especially right after the camera
  reveals new scene content, where most nearby points are still provisional.
  When that pool got too thin, PnP failed `--pnp-min-inliers` outright and the
  *entire frame got skipped* (no keyframe recorded at all, not just "less
  accurate"). Confirmed via testing that dropping to `--confirm-count 1` fixed
  the too-few-inliers symptom.
  - Tried: a fallback PnP path (confirmed-only first; if too thin, retry
    against the full confirmed+provisional pool to keep the trajectory
    continuous, but without letting that fallback pose extend the map).
    **Tested and found it did not help** - reverted/removed rather than
    kept as unused complexity.
- [x] **Diagnosed why BA doesn't clean up an isolated erratic keyframe**, after
  observing the trajectory "snapping back" to the correct path a keyframe or
  two after an erratic one, while the *map* stayed visibly wrong at that spot
  forever. Two compounding reasons, the second more fundamental than the
  first:
  1. Local BA only ever touches points/poses that are still inside its
     sliding `--ba-window`. Once the window advances past the keyframe that
     created a bad batch of points, nothing ever revisits them again - they
     freeze at whatever position they had when the window moved on.
  2. More fundamentally: BA is a **local, gradient-based refiner** - it
     minimizes reprojection error, it does not (and cannot) escape a pose
     that's already sitting in its own self-consistent local minimum. An
     erratic keyframe's points were triangulated *using* that keyframe's own
     (wrong) pose, so of course they reproject well under it - there's no
     residual telling BA "this is inconsistent" unless the erratic keyframe
     also shares enough observations with neighboring good keyframes to
     create a disagreeing constraint. What looks like "the trajectory finds
     its way back" is really just the *next* keyframe's independent PnP solve
     landing correctly on its own (dominated by the much larger, still-good
     confirmed map) - not BA correcting anything. This is exactly the class of
     error loop closure exists to fix in real SLAM systems (an independent
     global constraint reprojection-consistency alone can't supply); accepted
     as consistent with the already-logged v1 limitation rather than pursued
     further for now.
- [x] **Found the actual cause of new erratic jumps: PnP pose ambiguity.**
  Erratic segments consistently showed the logged relative `rotation` jumping
  to physically-implausible values (43°, 67°, 158°) despite tiny reported
  parallax (~10px) and a healthy-looking PnP inlier ratio. This is a known PnP
  failure mode: for poorly depth-distributed (near-planar/degenerate) point
  configurations, a "flipped"/mirrored alternate pose can fit the same 2D
  observations almost as well as the true one, and RANSAC can occasionally
  lock onto it while reporting a perfectly confident inlier count - it's not a
  weak fit, it's a *confidently wrong* one.
  - Fixed with `--max-plausible-rotation` (default 15°): reject a PnP result
    outright if its rotation relative to the previous keyframe exceeds this,
    computed and checked before any map-extension work happens.
  - Found a companion gap: the same kind of ambiguity doesn't always show up
    as a rotation flip - it can instead keep a plausible rotation but put the
    camera in the wrong *place*. Undetected, accepting this poisons every
    subsequent frame's rotation-vs-previous comparison (now measured against a
    corrupted baseline), which is what caused a single undetected bad pose to
    turn into a run of repeated "implausible rotation" rejections afterward -
    i.e. the "stuck" symptom was a downstream *consequence* of the first
    slip-through, not an independent problem.
  - Fixed with `--max-step-ratio` (default 6.0): reject a PnP result if the
    camera-center displacement vs. the previous keyframe exceeds this many
    multiples of a rolling median of recent accepted step sizes (scale-
    invariant, since the map's own scale is arbitrary).
- [x] **Found that BA itself could still reintroduce an implausible pose
  *after* these checks**, since the raw PnP pose was validated but then
  unconditionally overwritten by whatever `_run_local_ba` produced next - BA's
  own output was never checked. Restructured `_run_local_ba` to compute and
  return a proposed refinement without applying it, and added
  `_validate_and_apply_ba` to check it with the same rotation/step thresholds
  before committing; an implausible result is discarded entirely (falls back
  to the pre-BA, already-validated pose) rather than adopted.
  - Found a second gap one level deeper: `_apply_ba_result` rewrites *every*
    pose in the optimized window, but the validation only checked the
    *latest* one. BA (jointly optimizing the whole window) could produce a
    plausible-looking latest pose while quietly pushing an implausible
    correction into an *older* keyframe still in the window - passing the
    check, then only showing up later once that keyframe's rewritten position
    was reflected in the trajectory. Fixed by checking every pose in the
    proposed window against its own pre-BA value, rejecting the whole
    window's update (nothing touched) if any single one fails.
- [x] Cosmetic: the live display window (two video frames + the map panel,
  sized to match portrait phone footage) could be wider/taller than a typical
  screen. Now scaled down to fit within 1600x900 if needed before `imshow`,
  preserving aspect ratio, so nothing renders off-screen.

## v1 status: complete (2026-07-22)

The full stage 1-9 pipeline above (frame source → preprocessing → ORB feature
detection/matching → bootstrap-then-track pose estimation → triangulation →
keyframe selection → local bundle adjustment → visualization) is implemented and
validated end-to-end on real recorded phone footage, across multiple test
recordings with different scene compositions (near+far depth clusters, flat
foreground surfaces, straight lateral slides). Signed off as a working v1.
What's left is refinement and deferred items, not missing core functionality:

1. **Map point culling** — tried (soft deactivation + a hard active-set size
   cap, see the post-acceptance investigation above) and reverted. It worked as
   designed (bounded the active working set size) but did not fix the actual
   bottleneck on a long/rotating recording (see below), so it was removed again
   to keep the codebase simple rather than kept as unused complexity. Could be
   revisited alongside loop closure/global BA if those make it a meaningful
   contributor again.
2. **Loop closure / global consistency** — not yet implemented; confirmed to
   matter more than originally expected. On a very long recording, PnP inlier
   *count* (not just ratio) declined steadily over ~800 keyframes independent of
   map size (confirmed via the culling experiment above) and largely independent
   of rotation - consistent with plain accumulated drift that local-only BA has
   no mechanism to correct, since its window anchor slides forward forever with
   nothing tying the whole trajectory back to one consistent global frame. v1 is
   therefore expected to degrade on very long and/or rotation-heavy recordings;
   this is an accepted limitation of the local-only-BA architecture, not a bug,
   and stays the top candidate for a v1.x follow-up (periodic global BA or true
   loop closure) if longer recordings become a priority before v2/v3.
3. **Switch to the live phone feed** — `capture/video_source.py` already
   supports this (pass an integer device index instead of a file path); this is
   the natural next milestone now that the recorded-video pipeline is validated,
   per this plan's original "switch to live capture once v1 tracks cleanly"
   guidance.
4. **Ongoing tuning** — current best-known settings are `--min-parallax 30`
   `--ratio 0.75` (the defaults); `--ba-window`/`--ba-every`/`--ba-max-points`/
   `--min-triangulation-angle` may still need re-tuning per scene.


## v2 Development log

- [x] **Model: Depth Anything V2 Small**, fp32 ONNX export, downloaded from
  `onnx-community/depth-anything-v2-small` on Hugging Face to
  `pipeline/models/depth_anything_v2_vits.onnx`.
- [x] **Runtime: `onnxruntime` (CPU), not torch**, used machine has no
  dedicated GPU. Added to `requirements.txt`.
- [x] `pipeline/depth_ml.py` — `DepthEstimator` wraps the ONNX model
  (preprocessing reimplemented by hand in `numpy`/`cv2` to match the
  checkpoint's expected input, since no `transformers` dependency is pulled
  in for this).
- [x] `_demo()` in the same file (`python -m pipeline.depth_ml --video ...
  --calibration ... --model ...`) is the sanity check that the model
  actually runs: shows the source frame and colorized depth map side by
  side on a recording, confirming inference works end-to-end rather than
  just that the checkpoint loads. CPU inference is slow (~1.3s/frame), so
  `--depth-stride` reuses the last depth map for N frames instead of
  recomputing every frame.

## Ground-truth evaluation (TUM RGB-D)

Up to this point, "does it track well" was judged qualitatively (eyeball the
`--plot-output` PNG, watch the PnP inlier ratio). To get an actual number,
the pipeline can now be pointed at a public SLAM benchmark that ships
real ground-truth camera poses: [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)
(Technical University of Munich's Computer Vision Group).

- [x] **New frame source**: `capture.video_source.CalibratedImageSequenceSource`
  reads a TUM-style image-sequence folder (`rgb.txt` + PNGs) through the same
  `Frame(index, timestamp, image)` iterator interface as
  `CalibratedVideoSource`, sharing its undistort/crop logic (factored out
  into module-level `_load_calibration`/`_build_undistort_params`/
  `_undistort_and_crop` helpers rather than duplicated). A new
  `open_calibrated_source(source, calibration_path)` dispatches on
  `os.path.isdir(source)` to pick the right class; `mapping.py --video` now
  accepts a folder path as an alternative to a video file through this.
- [x] **Timestamps threaded through `keyframe_poses`**: entries are now
  `(R, t, timestamp)` instead of `(R, t)`, sourced from `frame.timestamp` at
  the point each keyframe is accepted (including the initial bootstrap
  reference frame). This is what lets `--trajectory-output` write real
  per-keyframe timestamps rather than a synthetic frame index.
- [x] **`--trajectory-output PATH`**: writes each accepted keyframe's pose as
  one TUM-format line (`timestamp tx ty tz qx qy qz qw`). The pipeline's own
  `(R, t)` is world-to-camera (`X_cam = R @ X_world + t`); TUM expects camera
  position + orientation *in the world frame* - the inverse. Position reuses
  `pose.camera_center` (`-R.T @ t`, the same inversion already used for
  plotting); orientation is `R.T` converted to a quaternion via
  `scipy.spatial.transform.Rotation.from_matrix(R.T).as_quat()`, which
  already returns `(qx, qy, qz, qw)` - TUM's column order, no reordering
  needed. Verified against the trivial case (identity pose → position
  `(0,0,0)`, quaternion `(0,0,0,1)`) and against the standard pinhole
  convention (a camera translated to world `(0,0,5)` with no rotation puts a
  world point at `(0,0,10)` at camera-frame `Z=5`, i.e. "5 in front" - not
  behind or mirrored) before trusting the ATE/RPE numbers below.

**Dataset used:** `rgbd_dataset_freiburg1_xyz` - the TUM download page's own
recommended starting sequence ("motion is relatively small and only a small
volume on an office desk is covered"), comparable in difficulty to this
project's own early demo recordings. 798 RGB frames, 640x480.

```bash
curl -LO https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
# extract to datasets/tum/rgbd_dataset_freiburg1_xyz/  (gitignored - see .gitignore's `datasets/` entry)
```

**Calibration:** `calibration/tum_freiburg1.yaml`, the published `freiburg1`
camera intrinsics from the TUM download page, in the same schema as
`calibration/phone_camera.yaml` (`image_width`, `image_height`,
`camera_matrix`, `dist_coeffs`). Kept tracked in git, unlike the dataset
itself.

**Command line run:**

```bash
python -m pipeline.mapping \
  --video datasets/tum/rgbd_dataset_freiburg1_xyz \
  --calibration calibration/tum_freiburg1.yaml \
  --trajectory-output estimate.txt \
  --no-display
```

Result: 355 keyframes accepted (18644 confirmed / 26406 total map points),
442 frames skipped - runs end-to-end exactly like a `recordings/*.mp4` file.

**Regression check:** re-ran `recordings/demo1.mp4` (existing `--video <file>`
path) before and after this change with `--no-display` - identical final
summary line (`44 keyframes accepted (3522 confirmed / 7315 total map
points), 793 frames skipped`) and byte-identical `--plot-output` PNG.

**Scoring with `evo`** (`pip install evo`):

```bash
evo_ape tum datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt estimate.txt -a -s
evo_rpe tum datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt estimate.txt -a -s
```

`-a -s` matters, not just `-a`: `-a` alone is a rigid SE(3) (no-scale)
alignment, which reports a huge, meaningless error for a monocular pipeline
whose scale is fixed once at bootstrap and isn't metric (see
`V2_INTEGRATION_PLAN.md` Section 2b) - confirmed empirically, `-a` alone gave
an ATE RMSE of 4.19m on this sequence. Adding `-s` (Sim(3) Umeyama alignment,
solving for one global scale factor too) is the correct comparison and gives:

```
APE (ATE), translation, Sim(3)-aligned:
  rmse    0.079640
  mean    0.062792
  median  0.043739
  max     0.230436
  min     0.009902

RPE (delta=1 frame), translation, Sim(3)-aligned:
  rmse    0.040234
  mean    0.028280
  median  0.018679
  max     0.175478
  min     0.002387
```

ATE RMSE ~8cm, RPE RMSE ~4cm on a sequence with meter-scale motion - a
working, reproducible measurement (per this issue's acceptance criteria, not
claimed to be a *good* score, though it's a reasonable result for a v1
monocular pipeline with no loop closure on a short, low-drift sequence).
`groundtruth.txt`'s ~3000 entries (mocap rate) vs. `rgb.txt`'s 798 (camera
rate) needed no manual handling - `evo`'s tools associate estimate/ground-truth
timestamps automatically.

## ML-vs-geometric trajectory comparison (v2 Phase 0)

`scripts/compare_trajectories.py` generalizes `plot_trajectory.py` to two
estimates at once (reusing its `load_tum`/`associate`/`umeyama_alignment`
helpers rather than duplicating them): each estimate is independently
Umeyama-aligned (Sim(3): rotation + scale + translation) onto ground truth,
both aligned trajectories are plotted overlaid on one figure, and ATE
(translation RMSE over the timestamp-matched, aligned pairs) is printed for
each. ATE is computed directly in Python - not by shelling out to `evo_ape` -
since `evo` isn't a hard dependency of this repo; verified to match
`evo_ape -a -s`'s `rmse` exactly (0.073161 on `fr1_xyz_estimate.txt`) before
trusting the number.

**Smoke test** (same file passed as both estimates, so both ATE numbers
must match exactly):

```bash
python scripts/compare_trajectories.py \
  --estimate-a results/fr1_xyz_estimate.txt \
  --estimate-b results/fr1_xyz_estimate.txt \
  --groundtruth datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --output results/compare_smoketest.png
```

```
ATE (translation RMSE), Sim(3)-aligned:
  geometric-only: 0.073161
  + ML depth: 0.073161
```

**Real baseline-vs-ML-fusion comparison (TBD once Phase 2 lands):** once
`--depth-fuse` (v2 Phase 1, see `V2_INTEGRATION_PLAN.md`) exists, run
`pipeline.mapping` once with it off and once with it on, each with its own
`--trajectory-output`, then:

```bash
python scripts/compare_trajectories.py \
  --estimate-a results/geometric_estimate.txt \
  --estimate-b results/ml_fusion_estimate.txt \
  --label-a "geometric-only" --label-b "+ ML depth" \
  --groundtruth datasets/tum/rgbd_dataset_freiburg1_xyz/groundtruth.txt \
  --output results/compare_ml_fusion.png
```