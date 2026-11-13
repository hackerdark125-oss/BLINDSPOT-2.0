# BLINDSPOT 2.0 Architecture

## Purpose and scope

BLINDSPOT is an experimental computer-vision review dashboard. It combines local image-quality diagnostics, optical-flow activity maps, camera calibration utilities, and optional object detection/tracking. It does not infer physical visibility, guarantee obstacle detection, or control a drone.

## Application flow

1. Streamlit collects a video, local camera index, or trusted RTSP/HTTP(S) source.
2. The selected workflow validates its input and bounds processing by frame count or capture duration.
3. OpenCV decodes frames; the relevant analysis module processes sampled frames.
4. The dashboard displays annotated frames, maps, charts, and aggregate metrics.
5. Reports are exported as CSV; calibration and benchmark workflows can also export JSON.

## Modules

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit navigation, controls, charts, image rendering, and exports |
| `blindspot/engine.py` | Frame sampling, explainable image-quality scores, and motion activity integration |
| `blindspot/live.py` | Bounded webcam/RTSP capture, optical-flow activity, and repeated-frame diagnostics |
| `blindspot/motion.py` | Farneback dense optical flow and per-cell time-since-motion maps |
| `blindspot/tracking.py` | Repeated/near-identical frame diagnostics |
| `blindspot/people.py` | Classical HOG pedestrian detection and short-term centroid association |
| `blindspot/ai_detection.py` | Optional YOLO11 detection, ByteTrack association, video/live processing, and the COCO class catalogue |
| `blindspot/calibration.py` | Single-camera chessboard intrinsics and lens undistortion |
| `blindspot/stereo.py` | Paired stereo calibration, rectification, disparity, and depth estimates |
| `blindspot/evaluation.py` | Dependency-free class-aware IoU and tracking metric evaluator |
| `scripts/benchmark_tracking.py` | CSV-to-JSON benchmark command-line interface |

## Object detection and tracking

The optional AI workflow loads Ultralytics YOLO11 nano or small weights. The detector returns class-aware boxes and confidence values; ByteTrack supplies temporary identities across sampled frames. The full COCO class mapping is shared by recorded and live dashboard modes.

The app caches one detector per Streamlit session. This keeps mutable ByteTrack state isolated between independent dashboard sessions. Each new video or live capture resets the tracker's state before processing.

For recorded videos, frame indices refer to the original video. For live sources, indices count frames received after the capture starts and timestamps are relative to a monotonic capture clock. Processing latency measures the YOLO + ByteTrack call; it excludes decode, network, display, and export overhead.

The live stream reader is bounded by capture duration and maximum sampled frames. It tolerates short read failures and stops after the configured read-idle timeout. OpenCV backend behavior can still affect how quickly a blocked network read returns.

## Image-quality and motion signals

The confidence map is a heuristic combination of exposure balance, local contrast, Laplacian sharpness, and edge density. It is not a learned object-confidence score or calibrated probability.

The motion module uses dense optical flow between sampled frames. Activity age measures elapsed time since the cell's motion magnitude last crossed a threshold. A static scene can therefore have a high activity age even when the camera is functioning normally.

## Calibration and stereo depth

Single-camera calibration estimates intrinsics from multiple chessboard views. Stereo calibration estimates relative camera pose from synchronized paired chessboard images. The stereo depth preview uses rectification and Semi-Global Block Matching disparity.

Depth values depend on calibration quality, a rigid camera baseline, synchronization, texture, and valid correspondences. They are prototype estimates and must not be used for navigation or safety decisions.

## Validation layers

- **Unit tests:** input validation, metric logic, synthetic calibration, and deterministic fake-camera/model behavior.
- **GitHub Actions:** Python compilation, pytest, and Docker image build.
- **Local application checks:** Streamlit startup and user-interface workflows.
- **Field evidence:** real camera footage, manually labelled detections, measured stereo targets, and repeatable hardware-specific performance results.

Passing CI verifies software checks only. It does not establish field accuracy or operational readiness.

## Data and privacy

Uploaded video is processed by the Python process hosting the app. Live camera access is also performed by that host; a remote dashboard cannot directly access a visitor's local webcam. Camera URLs may contain credentials and must not be committed, logged, or shared. Use trusted network sources and keep camera streams private.
