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

## Understanding motion activity age
The motion tracker computes dense optical flow between sampled frames with OpenCV's Farneback algorithm. Each grid cell stores its average flow magnitude and elapsed time since motion last exceeded a configurable threshold.

**Activity age is not observation freshness.** A perfectly static scene can have a high activity age even while the camera is operating correctly. Camera movement, lighting changes, compression artefacts, and sampling interval can affect optical-flow values. Use these maps as diagnostics for human review, not as proof that an area is unseen or stale.

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
Python · OpenCV · NumPy · Pandas · Plotly · Streamlit · Pytest · GitHub Actions

## References
- [OpenCV dense optical flow tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- [OpenCV Farneback optical-flow API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)

## License
MIT
