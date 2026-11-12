# BLINDSPOT 2.0
### Visual Uncertainty Intelligence System

BlindSpot 2.0 is an experimental computer-vision research prototype that highlights regions with weak image-quality signals and tracks motion activity over time. It supports recorded footage and bounded live camera capture.

> **Important:** confidence scores are heuristic image-quality indicators, not calibrated probabilities, semantic scene understanding, or proof that an area is visible.

## Features
- Analyse up to four recorded camera videos in one session
- Align multi-camera sample timestamps with manual per-camera offsets and a configurable skew tolerance
- Bounded live capture from a local webcam or RTSP/HTTP(S) stream
- Configurable OpenCV sampling interval and frame limit
- Per-cell image-quality confidence heatmaps
- Brightness, contrast, sharpness, and edge-density metrics
- Confidence and uncertainty trends over time
- Dense optical-flow motion maps using Farneback
- ORB/RANSAC global camera-motion compensation with conservative fallback when feature matches are weak
- Histogram-based scene-change detection that resets motion history to avoid false activity spikes
- Per-grid-cell time-since-motion activity maps
- Repeated-frame diagnostic for potentially frozen feeds
- Built-in HOG pedestrian detection with temporary centroid-based track IDs
- Optional YOLO11 object detection with persistent ByteTrack IDs for people, cars, motorcycles, buses, trucks, and other COCO classes
- Annotated sampled video frames and per-detection CSV export
- Chessboard-based camera intrinsic and lens-distortion calibration with robust outlier-view rejection and per-view reprojection diagnostics
- Paired chessboard stereo calibration for relative camera rotation and translation
- Planar two-camera ground-footprint and overlap estimation from calibrated stereo pose
- Stereo visual odometry, depth-to-3D point-cloud reconstruction, and PLY export
- Optional tracked-object masking during reconstruction to reduce moving-object contamination
- ORB/RANSAC visual loop-closure candidate retrieval plus full SE(3) pose-graph optimisation using independently verified relative-pose constraints
- Multi-camera XYZRGB point-cloud fusion using calibrated poses, plus a separate ICP rigid point-cloud registration API for overlapping clouds
- ASCII and binary little-/big-endian PLY 1.0 import, plus a batch fusion CLI and JSON summaries
- Stereo rectification, SGBM disparity estimation with bidirectional left-right consistency rejection, per-pixel relative depth confidence, and depth-map CSV export
- CSV-based stereo-depth benchmark CLI for MAE, RMSE, relative error, and valid-pixel coverage
- Ground-truth trajectory benchmark reporting ATE and relative translation/rotation errors
- Calibration JSON export and undistortion preview
- Weak-region prioritisation, multi-camera comparison, and CSV exports
- Per-cell quality-score label template and human-agreement benchmark (MAE, RMSE, bias, Pearson/Spearman, score bins)
- Unit tests and GitHub Actions CI

## Windows setup
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit, usually http://localhost:8501.

## Live camera mode
1. Select **Live webcam / RTSP** in the dashboard.
2. For a local webcam, enter camera index `0` (try `1` for another camera).
3. For a network camera, enter its RTSP URL, for example `rtsp://camera-host/stream`.
4. Select a capture duration and click **Start live analysis**.

Live capture runs on the machine hosting the Streamlit process. A remotely hosted dashboard cannot automatically access a camera attached to your personal computer. Network stream URLs should be kept private and must not be committed to Git.

## Camera calibration
1. Select **Camera calibration** in the dashboard.
2. Upload at least three photos of the same chessboard from varied positions and angles (more varied views are better).
3. Enter the number of inner corner intersections and the real-world square size.
4. Run calibration, review the reprojection RMS, and download the JSON parameters.

All images must have the same resolution. A lower reprojection error is generally preferable, but it is not by itself proof of good calibration.

For **Two-camera stereo extrinsics**, first calibrate each camera separately and upload both JSON files. Then upload synchronized paired chessboard photos in matching order: each pair must show the same rigid board pose from both cameras. The stereo workflow estimates relative rotation, translation, essential matrix, and fundamental matrix. Translation uses the same unit as the chessboard square size. Both cameras must have matching image dimensions and board geometry. The recorded-video dashboard can also align sampled timelines using manual offsets and a skew tolerance; this is timestamp matching, not hardware synchronization.

For **Planar coverage preview**, upload both intrinsics JSON files, the stereo extrinsics JSON, and a camera A-to-world pose JSON. The pose JSON contains a 3x3 rotation_matrix and 3D translation_vector, using the convention X_world = R × X_camera + t. OpenCV camera axes are X right, Y down, Z forward; define the world Z axis as vertical and keep all translations and plane heights in the same unit as the stereo baseline. The tool derives camera B's world pose from the stereo transform and projects each image's four corner rays onto a horizontal plane. It reports each footprint area, overlap, union, and a rasterized 0/1/2-camera coverage grid. All four corner rays must intersect the plane in front of each camera. This is a planar geometric estimate only: it does not infer camera pose, reconstruct arbitrary 3D surfaces, account for occluders, or prove image visibility.

For **Stereo depth preview**, upload both intrinsic JSON files, the stereo extrinsics JSON, and a synchronized image pair from the calibrated cameras. The app rectifies the pair, estimates disparity with StereoSGBM, and reprojects disparity into a depth map. Depth units follow the chessboard square-size unit. Depth quality depends on accurate calibration, fixed camera geometry, synchronization, texture, and visibility in both views; invalid pixels are expected around occlusions and low-texture regions. This is a prototype depth estimate, not a safety or measurement guarantee.

## Pedestrian detection and tracking
Select **Person detection & tracking** and upload a short video. The app uses OpenCV's built-in HOG + linear SVM pedestrian detector and a nearest-centroid tracker to associate detections across sampled frames. It displays annotated frames and exports timestamps, temporary track IDs, bounding boxes, and detector scores as CSV. This classical detector targets upright pedestrians; it is not a general object detector and may miss small, occluded, distant, or unusual poses. Track IDs are local to one video and can switch after occlusion or abrupt movement.

## Understanding motion activity age
The motion tracker estimates global frame-to-frame camera motion from ORB feature matches and a RANSAC affine transform, aligns the previous frame, then measures residual dense optical flow with Farneback. If there are too few reliable matches, it falls back to unaligned optical flow. A grayscale histogram change detector resets motion history after likely scene cuts instead of reporting a misleading large motion spike. The dashboard report records whether compensation was applied and whether a scene change was detected. These are approximate image-space diagnostics, not visual odometry, camera pose, or proof of object motion. Each grid cell stores residual flow magnitude and elapsed time since it last exceeded a configurable threshold.

**Activity age is not observation freshness.** A perfectly static scene can have a high activity age even while the camera is operating correctly. Camera movement, lighting changes, compression artefacts, and sampling interval can affect optical-flow values. Use these maps as diagnostics for human review, not as proof that an area is unseen or stale.

## Release validation
See [VALIDATION.md](VALIDATION.md) for the full automated and real-camera release checklist, including pedestrian detection, calibration, stereo depth, and benchmark evidence.

## Modern AI object detection and tracking

The **AI object detection & tracking** mode uses Ultralytics YOLO11 (nano or small model) for object detection and ByteTrack for persistent multi-object IDs across sampled video frames. Choose target classes, confidence threshold, inference image size, and frame stride. The dashboard renders annotated frames, summarizes detections by class, and exports timestamped bounding boxes, class labels, confidence scores, and track IDs to CSV.

This feature is optional to keep the base app and CI lightweight. Install its dependencies with:

```bash
pip install -r requirements-ai.txt
```

The first run downloads the selected pretrained weights and requires internet access. The nano model is the recommended starting point for CPU testing; the small model may improve accuracy at higher compute cost. YOLO/ByteTrack results are not guaranteed: evaluate on representative footage, check ID switches and missed detections, and measure FPS on the target hardware. Track IDs can change after occlusion or scene cuts. This is a research prototype, not a safety-critical perception system.

### Batch multi-camera PLY fusion

The dashboard also supports a reproducible command-line workflow for offline
point-cloud fusion. Each input PLY filename is paired with an explicit camera
ID, and the pose JSON must contain a camera-to-world rigid transform for every
input. The transforms must be independently calibrated in one shared world
frame and use the same distance unit as the point coordinates.

Example:

```bash
python scripts/fuse_multicamera.py \
  --cloud left=exports/left.ply \
  --cloud right=exports/right.ply \
  --poses camera_poses.json \
  --voxel-size 0.02 \
  --output exports/fused_scene.ply \
  --summary exports/fusion_summary.json
```

The pose JSON format is the same as the dashboard example:

```json
{
  "distance_unit": "meters",
  "cameras": {
    "left": {
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "translation_vector": [0, 0, 0]
    },
    "right": {
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "translation_vector": [0.25, 0, 0]
    }
  }
}
```

Use real calibrated transforms rather than the identity example. The command
validates camera IDs and rigid transforms, reads ASCII or binary XYZRGB PLY
files, and writes a fused ASCII PLY plus an optional JSON summary. It does not
estimate extrinsics or automatically align overlapping clouds.

## Deployment
See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker and Streamlit Community Cloud instructions. The repository includes a Dockerfile and a dark Streamlit theme.

## Run tests
```bat
python -m pytest -q
```
GitHub Actions runs the same test command on pushes and pull requests.

## How the image-quality score works
Each sampled frame is divided into a configurable grid. Every cell receives a heuristic score from exposure balance, local contrast, Laplacian sharpness, and Canny edge density. Scores range from 0 to 100; low-scoring cells are prioritised for human review. This is not a trained AI model or calibrated probability.

## Roadmap
- [x] Recorded video ingestion and frame sampling
- [x] Explainable local image-quality maps
- [x] Multi-video dashboard and comparison
- [x] Confidence/uncertainty trends and CSV exports
- [x] Live webcam / network-stream capture
- [x] Repeated-frame / possible-freeze diagnostics
- [x] Dense optical-flow maps and per-cell motion activity age
- [x] Chessboard-based single-camera intrinsic calibration and undistortion preview
- [x] Paired chessboard stereo extrinsic calibration
- [x] Stereo rectification, disparity estimation, and depth-map export
- [x] Input validation, unit tests, and CI
- [x] Baseline pedestrian detection and short-term centroid tracking
- [x] Optional YOLO11 detection across all 80 COCO classes with ByteTrack IDs
- [x] Quantitative detection/tracking evaluator and per-frame processing benchmark tooling
- [x] Scene-change resets and ORB/RANSAC global camera-motion compensation for motion diagnostics
- [x] Multi-camera timestamp alignment utility and dashboard export (manual offsets; no hardware sync guarantee)
- [x] Planar two-camera ground-footprint and overlap estimator using supplied camera pose
- [x] Stereo visual odometry and depth-based 3D point-cloud reconstruction (local trajectory; drift remains possible)
- [x] Geometric planar two-camera coverage and multi-camera point-cloud fusion using supplied camera poses
- [x] Manual loop-closure translation constraints and pose-graph drift correction
- [ ] Automatic visual localisation and loop-closure detection with full 6DoF pose-graph optimisation
- [ ] Ground-truth-validated arbitrary 3D coverage reconstruction and automatic multi-camera extrinsic calibration
- [x] Quality-score agreement evaluator and per-cell human-label export tooling
- [ ] Validated uncertainty model and representative benchmark dataset
- [x] Quantitative stereo-depth benchmark evaluator and CLI for known-distance samples

## Limitations and safety
- Image-quality and motion maps are diagnostics, not measurements of physical-world visibility or 3D coverage.
- Planar coverage uses supplied calibration and camera pose; it does not infer pose or reconstruct arbitrary 3D scene geometry.
- Texture and edge density do not guarantee useful semantic information.
- Static scenes may show high activity age; repeated frames do not alone prove camera failure.
- Results are decision-support signals only. Do not use this prototype for flight control, navigation, targeting, or safety-critical inspection.
- No claim of novelty or operational readiness is made.

## Tech stack
Python · OpenCV · NumPy · Pandas · Plotly · Streamlit · Pytest · GitHub Actions · Docker

## References
- [OpenCV dense optical flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [OpenCV Farneback optical-flow API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)

## License
MIT

## Image-quality score evaluation

The app can export a per-cell CSV template containing the heuristic `predicted_score` and an empty `reference_score` column. Review cells using a consistent human rubric (for example, 0 = unusable image detail and 100 = excellent usable detail), fill the reference scores, then run:

```bat
python scripts/benchmark_quality.py --input quality_labels.csv --output quality_benchmark.json
```

The report compares score agreement using MAE, RMSE, mean bias, Pearson and Spearman correlation, and score-bin summaries. This evaluates agreement with the labels you provide; it does not make the score a probability or prove generalisation beyond the labelled scenes.

## Stereo depth benchmark

The depth benchmark compares measured target distances with the median depth estimated from the corresponding target region. Create a CSV with one row per measured target/sample and these columns:

```csv
known_distance,measured_distance,valid_pixels,total_pixels,unit
100,104,8500,10000,cm
200,188,7600,10000,cm
```

Use the same distance unit for every row. If a target has no valid depth pixels, leave measured_distance empty and set valid_pixels to 0. Run:

```bat
python scripts/benchmark_depth.py --input depth_samples.csv --output depth_benchmark.json
```

The report contains MAE, RMSE, median absolute error, mean absolute relative error, valid-pixel percentage, and counts of measured/skipped samples. This is an evaluation utility; it does not create measured results or validate a camera rig by itself.

## Quantitative detection and tracking benchmark

The benchmark utility compares YOLO/ByteTrack predictions with manually labelled ground truth. It reports precision, recall, F1, false positives/negatives, mean matched IoU, consecutive-sampled-frame ID switches, and per-sampled-frame YOLO+ByteTrack processing throughput. ByteTrack state is reset between separate video uploads so IDs do not leak from one clip into the next. The app exports an object-level AI tracking CSV and a separate per-frame performance CSV (including frames with zero detections). Use the object-level file as predictions and pass the performance file with --timings for an unbiased per-sampled-frame latency summary. Metrics are meaningful only when labels are accurate and both CSV files use the same sampled frame indices.

Create ground-truth and prediction CSV files with this header:

```csv
frame_index,track_id,class_id,x,y,width,height
```

Each row represents one object box. Keep the same ground-truth track ID for the same real object across frames. Coordinates are pixel-space top-left X/Y and box width/height. Frame indices refer to original video frame numbers.

Run the evaluator:

```bat
python scripts/benchmark_tracking.py --ground-truth ground_truth.csv --predictions predictions.csv --timings blindspot_ai_frame_performance.csv --output benchmark.json --iou 0.5
```

Matching uses greedy class-aware IoU. ID switches count only when an object is matched to a different predicted ID on the immediately following evaluated sample (even when source frame indices skip due to sampling). This is a project diagnostic, not a full MOTChallenge/HOTA/IDF1 implementation. Do not infer FPS from video duration; measure per-frame YOLO+ByteTrack processing time directly.


## 3D point-cloud reconstruction and visual motion

The stereo depth pipeline can now be converted into a coloured 3D point cloud. Save a depth map as a 2D NumPy `.npy` array and provide the matching camera intrinsics JSON:

```bash
python scripts/reconstruct_point_cloud.py --depth depth.npy --intrinsics camera_calibration.json --output scene.ply --stride 2
```

The PLY file contains XYZ coordinates and RGB colour values and can be opened in common point-cloud viewers. Depth and translations retain the calibration distance unit. The point cloud is only as accurate as stereo calibration, synchronisation, depth estimates, and the supplied intrinsics. Invalid depth pixels are omitted.

A two-frame ORB + essential-matrix visual-motion estimator is available as `blindspot.reconstruction.estimate_visual_motion`. It estimates relative rotation and translation direction from calibrated images. Monocular translation scale is fundamentally ambiguous; this is not globally corrected visual odometry, SLAM, or absolute localisation. A metric 3D scene requires stereo depth or another scale reference and measured validation.

## Label-driven uncertainty calibration

The new `blindspot.uncertainty` module evaluates probability-like confidence values against independently labelled binary outcomes. It reports Brier score, log loss, expected calibration error (ECE), and reliability bins, and fits a monotone isotonic mapping using the pool-adjacent-violators algorithm.

Prepare a CSV with `predicted_confidence,outcome` columns, where confidence is in [0, 1] and outcome is 0 or 1:

```bash
python scripts/benchmark_uncertainty.py --train confidence_train.csv --eval confidence_heldout.csv --output uncertainty_calibration.json
```

The CLI requires separate training and held-out files: it fits the isotonic calibrator only on the training split and reports raw/calibrated metrics only on the held-out split. The existing image-quality heuristic score is not automatically a probability, and this framework does not make it calibrated without representative independent labels.


## Stereo visual odometry and multi-frame point-cloud fusion

The reconstruction module now includes a stereo visual-odometry step. It matches ORB features between adjacent frames, lifts previous-frame features into 3D using the corresponding stereo depth map, and estimates the previous-to-current camera transform with PnP-RANSAC. Unlike monocular essential-matrix motion, the translation can carry metric scale because it is anchored to the supplied depth map. Depth and translation use the same unit.

Python API: `blindspot.reconstruction.estimate_stereo_visual_motion`. Chain successive relative transforms with `compose_camera_to_world_pose`, then use `fuse_point_clouds` to transform frame point clouds into a common world coordinate system. Optional voxel averaging reduces duplicate density.

**Important limitations:** this is a frame-to-frame odometry building block, not a complete SLAM system. It does not perform loop closure, global bundle adjustment, relocalisation, dynamic-object masking, or drift correction. Pose errors accumulate over time; point-cloud fusion is only globally meaningful when supplied poses are accurate. Use calibrated stereo depth for every frame and independently measured reference trajectories before making localisation-accuracy claims.


The **Stereo visual odometry & 3D fusion** dashboard mode accepts two synchronized stereo videos plus both intrinsics files and stereo extrinsics. It samples frame pairs, computes depth, estimates local metric camera motion with PnP-RANSAC, chains camera poses, and exports a fused PLY point cloud plus a trajectory CSV. The point cloud uses the rectified camera projection matrix returned by stereo rectification. The workflow is compute-intensive; start with short clips and a coarse point-cloud stride. Failed frame pairs are skipped and counted. Pose drift accumulates because this prototype does not implement loop closure or global optimisation.


## Dashboard uncertainty calibration

Open **Camera calibration → Uncertainty calibration**. Upload two separate CSV files, each containing `predicted_confidence,outcome`: one for fitting and one for held-out evaluation. The dashboard fits isotonic regression only on the training file, then reports Brier score, log loss, ECE, and reliability bins before and after calibration on the held-out file. It can export the fitted mapping and a CSV containing calibrated held-out scores.

Do not feed the existing 0–100 image-quality heuristic directly into this probability calibrator. It must first be converted into a meaningful confidence estimate with independently labelled outcomes, and any conversion must be evaluated without data leakage. Better calibration metrics are not guaranteed; report the held-out results even when calibration makes them worse.


### Dynamic-object masking for 3D fusion

In **Camera calibration → Stereo visual odometry & 3D fusion**, enable **Mask detected moving objects before 3D fusion** to run the optional YOLO detector on each rectified left-camera frame. ByteTrack IDs are compared across samples, and only tracked boxes whose centers move beyond the configured pixel threshold (default 4 px) are excluded, with an 8-pixel margin. The same regions are excluded from the stereo depth mask used for point-cloud export and from ORB feature extraction in the previous frame used by PnP-RANSAC. Install the optional AI dependencies with `python -m pip install -r requirements-ai.txt`. This assumes a fixed camera rig and reduces contamination from tracked moving objects, but it is not semantic scene understanding: new/untracked objects are not masked, missed detections and partial occlusions remain possible, and camera movement can make static objects appear to move. The default mode remains dependency-free and unchanged.


## Translation pose-graph drift correction

The optional blindspot.pose_graph.optimize_pose_graph_translations utility
refines a short stereo-odometry trajectory when the caller has an independently
verified loop-closure displacement. It preserves consecutive odometry
measurements as weighted constraints, distributes loop-closure residual across
the trajectory with least squares, and anchors the first pose. It changes
translations only; rotations are not optimised.

A loop closure is supplied as (start_index, end_index, measured_world_delta, weight).
The measured displacement must come from a trustworthy revisit match or another
external reference. The function does not discover loop closures, validate
their correctness, estimate absolute position, or provide full SE(3) SLAM.
Incorrect constraints can make the trajectory worse. It is intended for short
trajectories and should be evaluated against measured reference paths before
being used in any operational setting.


## Multi-camera 3D point-cloud fusion

The `blindspot.multicam.fuse_multicamera_point_clouds` API registers named
camera-local XYZRGB clouds using supplied camera-to-world poses, then combines
them in one shared world frame. The pose for each camera is a rotation matrix
and translation vector; all translations and point coordinates must use the
same distance unit. Rotation matrices are checked for orthonormality and a
positive unit determinant. Optional voxel fusion averages XYZ and RGB values
using original point counts, with per-camera voxel aggregation to reduce peak
memory compared with stacking every raw point at once.

Example:

```python
from blindspot.multicam import fuse_multicamera_point_clouds

result = fuse_multicamera_point_clouds(
    {"north": north_cloud, "south": south_cloud},
    {
        "north": (north_rotation, north_translation),
        "south": (south_rotation, south_translation),
    },
    voxel_size=0.02,
)
fused_cloud = result.points
print(result.input_points, result.output_points)
```

This is a fusion stage, not an extrinsic-calibration or registration solver.
The camera poses must already refer to the same verified world coordinate
frame. It does not synchronise cameras, compensate rolling shutter, remove
occlusions, or estimate missing poses. Incorrect extrinsics can create doubled
or smeared geometry. The synthetic performance utility is
`python scripts/benchmark_fusion.py --cameras 2 --points-per-camera 100000`;
its results are hardware-dependent and are not real-scene accuracy evidence.


## Dashboard loop-closure drift correction

The **Stereo visual odometry & 3D fusion** dashboard includes an optional
loop-closure correction panel after the raw scene is reconstructed. Enter the
start/end rows and an independently measured world-frame displacement in the
calibrated distance unit. The NumPy pose-graph solver distributes the residual
across the trajectory while anchoring its first position, then the dashboard
re-fuses the original frame clouds using corrected camera translations.

This is a manually supplied constraint workflow, not automatic loop-closure
detection. Only use a displacement supported by a verified revisit or external
measurement. The optimiser currently adjusts translations only; rotations are
unchanged. Incorrect loop closures can worsen the trajectory, so preserve and
compare raw and corrected exports against measured reference positions.


## Dashboard multi-camera 3D fusion

Open **Camera calibration → Multi-camera PLY fusion**. Upload one XYZRGB
PLY per camera (the filename stem is the camera ID) and a shared pose JSON:

```json
{
  "distance_unit": "meters",
  "cameras": {
    "camera_1": {
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "translation_vector": [0, 0, 0]
    },
    "camera_2": {
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "translation_vector": [0.25, 0, 0]
    }
  }
}
```

The pose for each camera maps camera-local coordinates into the common world
frame. Camera IDs must match PLY filename stems exactly. The dashboard previews
a sampled fused point cloud and exports a fused PLY plus a JSON summary.
ASCII and binary little-/big-endian PLY 1.0 are supported. XYZRGB vertex properties may be reordered or include extra scalar fields; list-valued vertex properties and non-vertex elements before vertices are rejected. Extrinsics must already be
calibrated/verified; this workflow does not estimate camera poses or guarantee
overlap registration.


## Full SE(3) pose-graph optimisation

The optional `blindspot.pose_graph_se3.optimize_pose_graph_se3` API jointly
optimises camera rotations and translations for short trajectories. Poses are
camera-to-world transforms; loop-closure measurements map coordinates from
pose i into pose j. Consecutive odometry edges are derived from the initial
trajectory, and pose zero is fixed to remove gauge freedom. A damped dense
Gauss-Newton solver uses central-difference Jacobians and a Huber robust kernel.

This is a short-trajectory research solver, not a production large-scale SLAM
backend. Loop constraints must be independently verified and expressed in the
same distance unit as odometry. Incorrect constraints can deform the entire
trajectory. The existing translation-only API remains available for backward
compatibility. Synthetic tests establish solver behaviour, not field accuracy.


## Automatic rigid point-cloud registration (ICP)

The optional `blindspot.registration.register_point_cloud_icp` API estimates
a rigid source-to-target transform from overlapping XYZ or XYZRGB clouds. It
uses deterministic voxel sampling, bounded-memory nearest-neighbour matching,
distance gating, trimmed correspondences, and SVD rigid alignment. It returns
rotation, translation, correspondence fitness, RMSE, iteration count, and a
convergence flag. This can initialise or verify multi-camera alignment before
calling the existing fusion API.

ICP is a local optimiser and can converge to an incorrect alignment when the
initial pose is poor, overlap is small, geometry is repetitive or symmetric, or
units differ. Check fitness and RMSE and retain the original camera poses. It
does not replace synchronized capture, intrinsic/stereo calibration, or an
independent reference transform. It is not an automatic calibration guarantee.


### Relative stereo depth confidence

Each stereo depth result now includes a per-pixel `confidence_map` in [0, 1].
The relative score combines left-right disparity agreement with local image
texture; invalid depth pixels receive zero confidence. Low-texture and weakly
matched regions are therefore down-weighted. This is a heuristic quality score,
not a calibrated probability or a substitute for known-distance depth tests.


### Multi-camera extrinsic graph composition

The `blindspot.camera_rig.estimate_camera_rig_poses` API composes pairwise
stereo extrinsics into a common camera-to-world rig frame. Provide a connected
graph of calibrated pairwise transforms using `X_j = R_ji X_i + t_ji`; the
selected anchor camera becomes the world origin. The result includes cycle
rotation and translation residual diagnostics for redundant pairwise edges.

This automates coordinate-frame composition after pairwise calibration; it
does not automatically capture/detect calibration targets, synchronise
cameras, or globally refine noisy pairwise estimates. All translations must
share one unit, and disconnected or inconsistent calibration graphs must be
resolved before using the rig for fusion.


## Ground-truth trajectory benchmark

The metric trajectory evaluator compares estimated camera-to-world 4x4 pose
matrices against an equal-length independently measured reference trajectory.
It reports first-pose-aligned absolute trajectory error (ATE) and consecutive
relative pose error (translation RMSE and rotation RMSE in degrees). It does not
fit a scale factor, so both trajectories must use the same distance unit.

Save each trajectory as JSON containing an array of homogeneous 4x4 matrices,
then run:

```bash
python scripts/benchmark_trajectory.py --estimated estimated_poses.json --ground-truth ground_truth_poses.json --output trajectory_benchmark.json
```

The evaluator provides quantitative evidence only for the supplied reference
path and timestamps. Ground truth must be independently surveyed or measured;
synthetic tests do not establish real-world localisation accuracy.


## Advanced reconstruction APIs

- `blindspot.pose_graph_se3.optimize_pose_graph_se3` jointly optimises camera rotations and translations for short trajectories. The first pose is anchored; consecutive odometry edges are combined with independently verified relative-pose constraints. This is a dense research solver, not a large-scale SLAM backend.
- `blindspot.registration.register_point_cloud_icp` estimates a rigid transform between overlapping XYZ/XYZRGB clouds using voxel sampling and trimmed ICP. It requires a reasonable initial alignment and should be judged by returned fitness and RMSE; it is not intrinsic calibration or a guarantee of correct extrinsics.
- Stereo depth results expose `confidence_map`, a [0, 1] relative heuristic combining left-right consistency and local texture. It is not a calibrated probability.
- `blindspot.camera_rig.estimate_camera_rig_poses` composes pairwise stereo extrinsics into a common rig frame and reports cycle residuals. It does not automatically capture calibration boards or synchronise cameras.
- `scripts/benchmark_trajectory.py` compares estimated 4x4 camera-to-world matrices with independent ground truth and reports first-pose-aligned ATE plus consecutive relative pose errors. No scale fitting is applied.

See [VALIDATION.md](VALIDATION.md) for the synthetic CI gate and separate real-camera evidence checklist, and [RELEASE.md](RELEASE.md) for release boundaries.


The depth benchmark also accepts an optional `mean_confidence` column in the
CSV and reports the mean score plus Pearson correlation between per-target
confidence and absolute depth error. Use this diagnostic only with independent
measured targets and enough varied samples; correlation is not probability
calibration and can be unstable on small datasets.
