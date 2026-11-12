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
- [ ] Inspect reprojection RMS and compare original/undistorted previews.
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

Timestamp alignment uses manual offsets and nearest-neighbour matching. It is not hardware time synchronization and does not establish geometric overlap or 3D coverage.

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
