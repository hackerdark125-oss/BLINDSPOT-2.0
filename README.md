# BLINDSPOT 2.0
### Visual Uncertainty Intelligence System

BlindSpot 2.0 is an experimental computer-vision research prototype that highlights regions with weak image-quality signals and tracks motion activity over time. It supports recorded footage and bounded live camera capture.

> **Important:** confidence scores are heuristic image-quality indicators, not calibrated probabilities, semantic scene understanding, or proof that an area is visible.

## Features
- Analyse up to four recorded camera videos in one session
- Bounded live capture from a local webcam or RTSP/HTTP(S) stream
- Configurable OpenCV sampling interval and frame limit
- Per-cell image-quality confidence heatmaps
- Brightness, contrast, sharpness, and edge-density metrics
- Confidence and uncertainty trends over time
- Dense optical-flow motion maps using Farneback
- Per-grid-cell time-since-motion activity maps
- Repeated-frame diagnostic for potentially frozen feeds
- Built-in HOG pedestrian detection with temporary centroid-based track IDs
- Optional YOLO11 object detection with persistent ByteTrack IDs for people, cars, motorcycles, buses, trucks, and other COCO classes
- Annotated sampled video frames and per-detection CSV export
- Chessboard-based camera intrinsic and lens-distortion calibration
- Paired chessboard stereo calibration for relative camera rotation and translation
- Stereo rectification, SGBM disparity estimation, and depth-map CSV export
- Calibration JSON export and undistortion preview
- Weak-region prioritisation, multi-camera comparison, and CSV exports
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

For **Two-camera stereo extrinsics**, first calibrate each camera separately and upload both JSON files. Then upload synchronized paired chessboard photos in matching order: each pair must show the same rigid board pose from both cameras. The stereo workflow estimates relative rotation, translation, essential matrix, and fundamental matrix. Translation uses the same unit as the chessboard square size. Both cameras must have matching image dimensions and board geometry. This is a calibration result, not yet a 3D scene-coverage map.

For **Stereo depth preview**, upload both intrinsic JSON files, the stereo extrinsics JSON, and a synchronized image pair from the calibrated cameras. The app rectifies the pair, estimates disparity with StereoSGBM, and reprojects disparity into a depth map. Depth units follow the chessboard square-size unit. Depth quality depends on accurate calibration, fixed camera geometry, synchronization, texture, and visibility in both views; invalid pixels are expected around occlusions and low-texture regions. This is a prototype depth estimate, not a safety or measurement guarantee.

## Pedestrian detection and tracking
Select **Person detection & tracking** and upload a short video. The app uses OpenCV's built-in HOG + linear SVM pedestrian detector and a nearest-centroid tracker to associate detections across sampled frames. It displays annotated frames and exports timestamps, temporary track IDs, bounding boxes, and detector scores as CSV. This classical detector targets upright pedestrians; it is not a general object detector and may miss small, occluded, distant, or unusual poses. Track IDs are local to one video and can switch after occlusion or abrupt movement.

## Understanding motion activity age
The motion tracker computes dense optical flow between sampled frames with OpenCV's Farneback algorithm. Each grid cell stores its average flow magnitude and elapsed time since motion last exceeded a configurable threshold.

**Activity age is not observation freshness.** A perfectly static scene can have a high activity age even while the camera is operating correctly. Camera movement, lighting changes, compression artefacts, and sampling interval can affect optical-flow values. Use these maps as diagnostics for human review, not as proof that an area is unseen or stale.

## Release validation
See [VALIDATION.md](VALIDATION.md) for the full automated and real-camera release checklist, including pedestrian detection, calibration, stereo depth, and benchmark evidence.

## Modern AI object detection and tracking

The **AI object detection & tracking** mode uses Ultralytics YOLO11 (nano or small model) for object detection and ByteTrack for persistent multi-object IDs across sampled video frames. Choose target classes, confidence threshold, inference image size, and frame stride. The dashboard renders annotated frames, summarizes detections by class, and exports timestamped bounding boxes, class labels, confidence scores, and track IDs to CSV. The **AI live webcam / RTSP** mode applies the same detector and tracker to a bounded live capture window, with all 80 COCO classes available, per-frame timestamps, detections, and downloadable CSV reports. The live reader tolerates brief frame drops and stops after a configurable read-idle timeout. Each Streamlit session owns its detector instance so independent users do not share mutable ByteTrack state. Live processing runs on the machine hosting Streamlit; it is not browser-side camera access.

This feature is optional to keep the base app and CI lightweight. Install its dependencies with:

```bash
pip install -r requirements-ai.txt
```

The first run downloads the selected pretrained weights and requires internet access. The nano model is the recommended starting point for CPU testing; the small model may improve accuracy at higher compute cost. YOLO/ByteTrack results are not guaranteed: evaluate on representative footage, check ID switches and missed detections, and measure FPS on the target hardware. Track IDs can change after occlusion or scene cuts. This is a research prototype, not a safety-critical perception system.

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
- [x] Bounded live webcam / RTSP YOLO11 + ByteTrack tracking
- [ ] Scene-change-aware activity baselines and robust camera-motion compensation
- [ ] Multi-camera time synchronisation and validated 3D coverage reconstruction
- [ ] Geometric multi-camera coverage estimation using camera pose / visual localisation
- [ ] Validated uncertainty model and benchmark dataset
- [ ] Quantitative stereo-depth benchmark against known-distance scenes

## Limitations and safety
- Image-quality and motion maps are diagnostics, not measurements of physical-world visibility or 3D coverage.
- Camera views are analysed independently; no calibration or cross-camera geometry is inferred.
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
