# BLINDSPOT 2.0 — Release Validation Checklist

This checklist separates automated software checks from tests that require real camera footage. A passing CI run does not prove detection accuracy, calibration quality, or safe operation.

## 1. Automated checks

Run from the repository root:

```bat
python -m pip install -r requirements.txt
python -m compileall -q app.py blindspot tests
python -m pytest -q
streamlit run app.py
```

Release gate:
- [ ] Python compilation succeeds.
- [ ] All unit tests pass.
- [ ] Docker image builds successfully.
- [ ] Streamlit dashboard opens without startup errors.

## 2. Recorded-video analysis

Use at least two short videos with different lighting and scene content.

- [ ] Upload MP4/AVI/MOV footage and confirm sampling completes.
- [ ] Confirm confidence, uncertainty, motion plots, and heatmaps render.
- [ ] Confirm frame reports include camera-motion compensation and scene-change diagnostic flags.
- [ ] Download the CSV and check timestamps and numeric columns.
- [ ] Upload a second camera clip and confirm camera comparison is shown.
- [ ] Set a known time offset for each camera and confirm synchronized sample groups are exported only within the configured skew tolerance.
- [ ] Confirm each matched sample index is used at most once per non-reference camera.
- [ ] Try an invalid or corrupt file and confirm the app reports an error without crashing.

- [ ] Run the batch multi-camera fusion CLI with two calibrated PLY clouds and verify the fused PLY and JSON summary:
  `python scripts/fuse_multicamera.py --cloud left=left.ply --cloud right=right.ply --poses camera_poses.json --output fused.ply --summary fusion_summary.json`
- [ ] Confirm that mismatched camera IDs, missing files, malformed pose JSON, and invalid rotations fail with clear messages.

## 3. Live camera

- [ ] Test the local webcam at index 0 (try 1 if multiple cameras are attached).
- [ ] Confirm capture ends at the configured duration.
- [ ] Cover and uncover the lens to check that the repeated-frame warning is only a diagnostic.
- [ ] Pan the camera across a textured static scene and check that global motion compensation reduces broad false motion where ORB matches are reliable.
- [ ] Cut between visually different scenes and confirm the motion history resets rather than showing a large cut-induced flow spike.
- [ ] Test a network stream only on a trusted network; do not commit credentials or RTSP URLs.

## 4. Pedestrian detection and tracking

Use a short clip containing several clearly visible, upright pedestrians.

- [ ] Confirm bounding boxes are drawn around visible pedestrians.
- [ ] Confirm a person moving slowly keeps the same track ID across adjacent sampled frames.
- [ ] Confirm a newly entering person receives a new ID.
- [ ] Process a second, unrelated clip and confirm track IDs start fresh rather than carrying over from the previous clip.
- [ ] Confirm the CSV contains timestamps, IDs, bounding boxes, and detector scores.
- [ ] Repeat with different distances, lighting, crowd density, and partial occlusion.
- [ ] Record false positives and missed detections. HOG is a classical upright-person detector, not a modern general-purpose detector; poor performance in difficult scenes is an expected limitation.

## 5. Single-camera calibration

- [ ] Capture 10–20 chessboard photos at varied positions, distances, and tilts.
- [ ] Keep resolution and focus fixed.
- [ ] Run calibration and save the JSON.
- [ ] Inspect overall RMS, mean per-view error, worst per-view error, and compare original/undistorted previews.
- [ ] Confirm outlier views are only removed when the requested minimum view count remains; retain diverse board poses and do not treat low reprojection error as proof of metric accuracy.
- [ ] Re-run with a different set of photos and check that results are reasonably consistent.

## 6. Stereo calibration and depth

- [ ] Calibrate both cameras individually at the same image resolution.
- [ ] Capture synchronized chessboard pairs at varied board poses.
- [ ] Keep the stereo camera rig rigid after calibration.
- [ ] Confirm stereo calibration exports rotation and translation.
- [ ] Capture a textured scene with both cameras at the same time.
- [ ] Place a flat target at a measured distance and compare estimated depth with the known distance.
- [ ] Repeat at multiple distances and document median absolute error and invalid-pixel percentage.
- [ ] Record one row per measured target with known distance, median valid depth, valid-pixel count, total-pixel count, and consistent units.
- [ ] Run `python scripts/benchmark_depth.py --input depth_samples.csv --output depth_benchmark.json` and archive the JSON report.
- [ ] Do not use the depth estimate for navigation, flight control, or safety decisions.

## 6.2 Image-quality score agreement benchmark

- [ ] Export the per-cell quality-label template from the dashboard.
- [ ] Fill `reference_score` using a documented human rubric; label cells without seeing the model score where practical to reduce bias.
- [ ] Keep the same scoring rubric across lighting, distance, blur, exposure, and scene conditions.
- [ ] Run `python scripts/benchmark_quality.py --input quality_labels.csv --output quality_benchmark.json`.
- [ ] Record sample count, MAE, RMSE, bias, correlations, and score-bin summaries for each condition.
- [ ] Do not call the score calibrated or validated until it has been evaluated on a separate held-out labelled set.

Timestamp alignment uses manual offsets and nearest-neighbour matching. It is not hardware time synchronization and does not establish geometric overlap.

## 6.3 Planar two-camera coverage preview

- [ ] Calibrate both cameras individually and run paired stereo calibration.
- [ ] Provide camera A's camera-to-world rotation and translation in the same coordinate frame and distance unit as the stereo baseline.
- [ ] Confirm world Z is vertical and the selected horizontal plane is below both cameras.
- [ ] Generate the planar footprint/overlap map and export its JSON and CSV outputs.
- [ ] Compare polygon boundaries and overlap area with a measured, unobstructed test layout.
- [ ] Repeat with changed camera spacing and orientation; record the pose source and all units.

The planar estimator intersects image-corner rays with one known plane. It does not model occlusion, arbitrary 3D geometry, or semantic visibility; it is not a full 3D coverage reconstruction.

## 7. Release evidence

Before calling the project field-validated, attach:
- CI run link and test count.
- Example input clips and anonymized output screenshots.
- Camera models, resolutions, frame rates, and lighting conditions.
- Calibration board dimensions and measured square size.
- Stereo baseline and measured target distances.
- Detection precision/recall from manually labelled footage.
- Depth error summary across measured distances.

## Completion definition

The repository can be considered **software-complete for its documented prototype scope** when automated checks pass and the dashboard workflows run. It should only be described as **field-validated** after the real-camera checks and measured benchmarks above have been completed and recorded. There is no honest universal 100% score for an open-ended computer-vision system.


## 6.1 Quantitative YOLO/ByteTrack benchmark

- [ ] Label a representative clip with stable object IDs and COCO class IDs.
- [ ] Export predictions with original frame indices and pixel-space boxes, plus the separate per-frame performance CSV (including zero-detection frames).
- [ ] Ensure ground-truth and prediction CSVs cover identical sampled frame indices.
- [ ] Run `python scripts/benchmark_tracking.py --ground-truth ground_truth.csv --predictions predictions.csv --timings blindspot_ai_frame_performance.csv --output benchmark.json`.
- [ ] Record precision, recall, F1, mean matched IoU, false positives, false negatives, and ID switches.
- [ ] Measure inference latency separately on the target machine and record the hardware, model size, image size, and frame stride.
- [ ] Repeat in daylight, low light, crowded scenes, and partial occlusion; keep each condition's result separate.

The quality-score benchmark measures agreement with supplied human labels; it does not establish probabilistic calibration or generalisation without a held-out dataset.

The depth benchmark is only as reliable as the manually measured target distances and selected target regions. It reports error metrics but does not establish general depth accuracy.

The motion compensation and scene-change heuristics are approximate; feature-poor scenes fall back to raw optical flow, and false scene-change detections remain possible. They are not visual odometry or camera-pose estimates.

The included evaluator uses greedy class-aware IoU matching and counts ID changes only on consecutive evaluated samples. It is not a replacement for a full MOT benchmark. Do not claim field accuracy without labelled footage and repeatable measurements.


## 6.4 3D point-cloud and visual-motion software validation

- [ ] Export a stereo depth map and matching camera intrinsics; convert it to PLY with `scripts/reconstruct_point_cloud.py`.
- [ ] Confirm point count, coordinate units, and colour alignment in a point-cloud viewer.
- [ ] Compare reconstructed points against measured targets at multiple distances and positions.
- [ ] Check that invalid/zero depth pixels are omitted and that max-depth filtering behaves as configured.

- [ ] Confirm the stereo depth valid mask applies left-right disparity consistency; inspect invalid pixels around occlusions and depth discontinuities.
- [ ] Compare valid-pixel coverage and measured depth error with the consistency filter enabled at its default threshold; do not maximise coverage by weakening consistency without checking error.
- [ ] Run the two-frame visual-motion estimator on textured footage with known camera movement; record inlier count and translation direction.
- [ ] Do not interpret monocular translation direction as metric displacement or absolute camera position.

## 6.5 Uncertainty calibration software validation

- [ ] Prepare independently reviewed binary outcomes and confidence values.
- [ ] Split labelled samples into calibration/training and held-out evaluation sets before fitting.
- [ ] Fit the isotonic mapping on the training split only.
- [ ] Evaluate Brier score, log loss, ECE, and reliability bins on the held-out split.
- [ ] Repeat across lighting, motion, distance, and scene categories; report sample counts.
- [ ] Do not call the uncertainty output calibrated until held-out results support that claim.


## 6.6 Stereo visual odometry and fused 3D reconstruction

- [ ] Generate synchronized stereo depth maps for consecutive frames using fixed, calibrated cameras.
- [ ] Estimate frame-to-frame motion with ORB correspondences and PnP-RANSAC using previous-frame depth.
- [ ] Record inlier counts, inlier ratios, failed-frame rate, and processing time.
- [ ] Chain relative poses into a common coordinate frame and inspect accumulated drift.
- [ ] Fuse per-frame point clouds using the estimated camera-to-world poses; compare against a measured reference object/trajectory.
- [ ] Repeat with moving objects, low texture, blur, lighting changes, and partial occlusion.
- [ ] Treat this as local stereo visual odometry only; loop closure, global optimisation, relocalisation, and drift correction remain outside the current implementation.


## 6.7 Dashboard stereo visual odometry workflow

- [ ] Upload two synchronized stereo videos recorded with the calibrated fixed camera rig.
- [ ] Upload the matching camera A/B intrinsics and stereo extrinsics JSON files.
- [ ] Confirm the videos use the calibrated resolution, camera order, and compatible frame rates.
- [ ] Run a short clip first; review reconstructed/skipped pair counts and PnP inlier ratios.
- [ ] Inspect the trajectory CSV and fused PLY in a point-cloud viewer.
- [ ] Compare trajectory and reconstruction against independently measured ground truth before making accuracy claims.


## 6.8 Held-out uncertainty calibration dashboard

- [ ] Prepare separate training and held-out CSV files with independently labelled binary outcomes.
- [ ] Confirm confidence values represent probabilities in [0, 1], not raw 0–100 image-quality scores.
- [ ] Fit the isotonic calibrator using training data only.
- [ ] Compare Brier score, log loss, ECE, and reliability bins on held-out data before/after calibration.
- [ ] Retain the held-out sample count and label-generation rubric with every report.
- [ ] Repeat on different scenes, lighting, camera distances, and motion conditions; do not generalise from one split.


## 6.9 Optional dynamic-object masking in stereo 3D fusion

- [ ] Install optional AI dependencies from `requirements-ai.txt` and verify YOLO weights are available.
- [ ] Run the same stereo clip once with masking disabled and once enabled.
- [ ] Confirm only tracked objects whose centers move beyond the configured pixel threshold are excluded from the point-cloud depth mask and previous-frame PnP feature mask.
- [ ] Compare point counts and inspect both PLY outputs; do not assume fewer points always means better reconstruction.
- [ ] Record missed detections and static objects incorrectly masked.
- [ ] Repeat with people/vehicles at different scales and partial occlusion.


## 6.10 Translation pose-graph correction

- [ ] Confirm the no-loop-closure path preserves the original trajectory.
- [ ] Add a known synthetic closure residual and verify correction is distributed while the first pose remains anchored.
- [ ] Evaluate only with independently verified revisit constraints; compare raw and corrected trajectory error against measured reference positions.
- [ ] Test multiple independently verified relative-pose constraints and confirm both rotation and translation are adjusted where justified.
- [ ] Compare translation-only and full SE(3) optimisation against the same measured reference trajectory.
- [ ] Record initial/final solver cost, iteration count, convergence flag, and raw/corrected ATE/RPE.
- [ ] Test outlier constraints and poor initial poses; preserve the raw trajectory for comparison.
- [ ] In the dashboard, verify the 6DoF panel uses the documented Rz · Ry · Rx Euler convention and exports both corrected PLY and trajectory CSV.
- [ ] Confirm the first pose remains anchored and that incorrect constraints can still deform the trajectory.


## 6.11 Multi-camera 3D fusion software validation

- [ ] Supply one point cloud and verified camera-to-world pose per camera, all in one coordinate frame and distance unit.
- [ ] Confirm camera IDs match exactly and malformed/non-rigid rotations are rejected.
- [ ] Compare a known synthetic two-camera transform with expected world coordinates.
- [ ] Compare voxel-averaged XYZRGB values against a hand-calculated result.
- [ ] Run `python scripts/benchmark_fusion.py --cameras 2 --points-per-camera 100000 --output fusion_benchmark.json` and record hardware, runtime, input/output point counts, and voxel size.
- [ ] Repeat with actual calibrated cameras and measured reference geometry; inspect overlap seams and duplicate surfaces.
- [ ] Do not interpret a successful fusion operation as proof that camera extrinsics or synchronization are accurate.


## 6.12 Dashboard loop-closure correction

- [ ] Reconstruct a short stereo sequence and export the raw trajectory/PLY.
- [ ] Confirm the loop-closure panel requires two distinct trajectory rows.
- [ ] Enter an independently measured displacement in the stereo calibration distance unit and apply the correction.
- [ ] Confirm the first trajectory position remains anchored and the end-to-start displacement approaches the supplied constraint.
- [ ] Inspect the corrected PLY and trajectory CSV; compare raw and corrected outputs against measured reference positions.
- [ ] Try an invalid or unsupported constraint and confirm the dashboard reports the error without crashing.
- [ ] Never use a guessed loop-closure displacement or describe manual constraints as automatic loop-closure detection.


## 6.13 Dashboard multi-camera PLY fusion

- [ ] Export ASCII or binary XYZRGB PLY files from at least two camera point clouds.
- [ ] Create one shared pose JSON with matching filename-stem camera IDs and verified camera-to-world transforms.
- [ ] Confirm the distance unit is consistent across all point clouds and translations.
- [ ] Run the dashboard fusion workflow and inspect the sampled 3D preview.
- [ ] Download the fused PLY and summary JSON; verify vertex count and coordinate alignment in a point-cloud viewer.
- [ ] Try duplicate filenames, missing pose entries, invalid rotations, truncated binary PLY input, and unsupported list-valued vertex properties; confirm clear validation errors.
- [ ] Compare overlapping geometry against a measured reference before claiming reconstruction accuracy.


## 6.6 Visual loop-closure candidate retrieval

- [ ] Run the stereo odometry workflow on a route with a deliberate revisit and inspect the candidate table.
- [ ] Confirm candidates are separated by the configured temporal gap and report feature matches, geometric inliers, inlier ratio, and verification model.
- [ ] Repeat on a route with no revisit and record false candidate rate; tune thresholds only against labelled sequences.
- [ ] Verify a candidate independently before entering a measured world-frame translation constraint.
- [ ] Confirm candidate retrieval never changes the trajectory or applies a pose-graph constraint by itself.
- [ ] Test low-texture, repeated-pattern, moving-object, lighting-change, and planar-scene footage.
- [ ] Do not describe a candidate as a confirmed loop closure or metric pose estimate.



## 6.14 Full SE(3) pose-graph software validation

- [ ] Run the synthetic pose-graph tests in CI and verify a non-trivial loop residual reduces robust cost over multiple iterations.
- [ ] Verify the first camera pose remains fixed and rotations as well as translations are optimised.
- [ ] Evaluate against independently measured camera poses; report ATE and RPE before and after optimisation.
- [ ] Test multiple closures, inconsistent closures, large initial drift, and longer paths; the dense solver is intended for short trajectories only.
- [ ] Never treat a visually retrieved loop candidate as a metric constraint without independent geometric/metric verification.

## 6.15 Automatic rigid point-cloud registration software validation

- [ ] Start with two clouds of a known rigid object and a measured initial transform.
- [ ] Run ICP and record correspondence fitness, RMSE, iteration count, and convergence flag.
- [ ] Compare estimated rotation/translation against the known transform, including small overlap and outlier cases.
- [ ] Repeat with different voxel sizes and correspondence thresholds; retain original poses if registration quality is poor.
- [ ] Test low overlap, symmetric/repetitive geometry, and bad initial alignment; ICP may converge to a local minimum.
- [ ] Compare automatic registration with independent camera extrinsics before using the result for multi-camera fusion.
- [ ] In the dashboard, verify the selected reference camera remains fixed and every other camera reports ICP fitness/RMSE before fused output is accepted.

## 6.16 Stereo confidence and ground-truth trajectory benchmarks

- [ ] Inspect the per-pixel stereo confidence map in the dashboard and downloaded CSV; verify invalid depth pixels receive zero confidence and all scores remain in [0, 1].
- [ ] Compare confidence bins with independently measured depth errors and coverage on held-out scenes; do not call the heuristic a probability without calibration evidence.
- [ ] Export estimated and ground-truth camera-to-world trajectories as equal-length JSON arrays of 4x4 matrices.
- [ ] Run `python scripts/benchmark_trajectory.py --estimated estimated_poses.json --ground-truth ground_truth_poses.json --output trajectory_benchmark.json`.
- [ ] Record ATE, relative translation RMSE, relative rotation RMSE, units, timestamp matching, and ground-truth source.
- [ ] Ensure ground truth is independently surveyed/measured; no scale fitting is applied by this evaluator.


## 6.17 Multi-camera rig calibration graph validation

- [ ] Calibrate each connected camera pair with synchronized chessboard pairs and matching units.
- [ ] Export the pairwise relative transforms and compose them with the dashboard camera-rig workflow.
- [ ] Confirm the selected anchor pose is identity and every camera is reachable in the pairwise graph.
- [ ] Inspect redundant-edge cycle rotation and translation residuals; inconsistent cycles indicate pairwise calibration disagreement.
- [ ] Export the shared camera-to-world pose JSON and compare it with an independently measured rig layout before fusion.
- [ ] Do not treat graph connectivity as proof of calibration accuracy or synchronization.


## 1.1 Automated regression coverage

The CI workflow compiles the application, library, scripts, and tests, then runs the complete pytest suite on Python 3.11. It also builds the runtime Docker image without publishing it.

The reconstruction regression tests cover deterministic small-transform ICP recovery, XYZRGB input handling, invalid ICP configuration, insufficient correspondences, anchored SE(3) loop-closure correction, and invalid pose-graph constraints. Tracking regression tests cover perfect matches, false positives, missed detections, identity switches, invalid benchmark inputs, latency metrics, and repeated-frame freeze diagnostics.

A green CI run is required before merging software changes. It is not a substitute for the real-camera, measured-distance, or held-out benchmark checks below.
