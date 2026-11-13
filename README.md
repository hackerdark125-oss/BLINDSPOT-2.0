# BLINDSPOT AERIAL

### Drone-only AI vision, object tracking and 3D mapping

BLINDSPOT AERIAL analyses recorded drone footage and supported live drone video streams. It combines YOLO object detection, ByteTrack multi-object tracking, flight telemetry visualisation and a georeferenced point-cloud export workflow.

**Product scope:** drone imagery and drone-associated telemetry only. This application is not a CCTV/VMS product and does not control, navigate, or autonomously pilot an aircraft.

## Core capabilities

- Recorded aerial video analysis (MP4, MOV, AVI, MKV, M4V)
- Live drone video input over RTSP/RTSPS/HTTP/HTTPS
- YOLO11 object detection with configurable confidence and image size
- ByteTrack persistent track IDs within a processed video/stream
- Aerial target classes including people, cars, motorcycles, buses, trucks, boats, birds and common animals
- Annotated frame review and timestamped detection CSV export
- Flight telemetry CSV ingestion, validation, interpolation and GPS track visualisation
- Local East/North/Up (ENU) conversion from GPS coordinates
- Stereo depth generation from a synchronized, calibrated drone-camera image pair using the existing OpenCV SGBM pipeline
- Georeferenced XYZRGB PLY export from stereo depth or a supplied depth map, camera intrinsics, telemetry and calibrated camera-to-ENU rotation
- Reuse of the existing stereo depth, visual odometry, point-cloud, trajectory evaluation and reconstruction modules

## Quick start (Windows)

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ai.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually http://localhost:8501.

The first YOLO run downloads model weights and requires internet access. Start with `yolo11n.pt` for lower compute requirements. GPU acceleration depends on a compatible PyTorch/CUDA installation.

## Live drone video

1. Configure the drone or ground station to expose a supported video stream.
2. Select **Live drone stream (RTSP/HTTP)**.
3. Enter the stream URL and choose the detection classes and processing rate.
4. Run analysis and export the timestamped detections.

The stream is read by the machine running BLINDSPOT AERIAL. Network reachability, codec support and stream credentials must be configured by the operator. Keep credentials out of source control and redact them from logs.

## Flight telemetry CSV

Required columns:

```csv
timestamp_seconds,latitude_deg,longitude_deg,altitude_m
0.0,30.000000,78.000000,120.0
1.0,30.000010,78.000020,120.8
```

Optional columns: `roll_deg`, `pitch_deg`, `yaw_deg`.

Video-relative timestamps and telemetry timestamps must share the same time origin. Telemetry must be ordered with strictly increasing timestamps. Latitude/longitude are WGS84 degrees. Altitude values must use one consistent vertical datum.

## 3D mapping workflow

The current mapping workflow accepts either a precomputed NumPy `.npy` depth map, or a synchronized left/right image pair plus the two camera intrinsic JSON files and stereo extrinsics JSON. In stereo mode, the app calls the existing `rectify_and_compute_depth` SGBM pipeline. It also requires:
- Camera intrinsics JSON containing the camera calibration parameters
- Flight telemetry CSV
- A calibrated camera-to-local-ENU rotation matrix
- A depth-unit conversion factor to metres (use 1 for metres or 0.001 for millimetres)

The first telemetry GPS position is used as the local ENU origin. The output PLY coordinates are metres relative to that origin. Camera mounting/gimbal extrinsics must be measured or calibrated and supplied; the UI's identity rotation is only a placeholder, not a valid default for a downward-facing drone camera.

The repository already includes stereo calibration/depth, stereo visual odometry, point-cloud reconstruction, pose-graph optimisation, point-cloud fusion and trajectory evaluation utilities. These can be composed into a fuller multi-frame aerial mapping pipeline after the first hardware validation.

## Important limitations

- YOLO/ByteTrack performance must be measured on representative aerial footage. Small, distant, blurred, occluded or low-contrast objects may be missed.
- Track IDs are temporary tracker identities, not verified real-world identities.
- GPS interpolation and ENU conversion are local approximations; large-area mapping should use a suitable geodetic/geospatial reference workflow.
- Georeferenced point-cloud accuracy depends on stereo calibration, depth quality, timestamp alignment, camera-to-gimbal extrinsics, GPS quality and altitude datum.
- A single monocular video does not provide metric depth by itself. Metric 3D reconstruction requires calibrated stereo, known scale, or another valid depth source.
- BLINDSPOT AERIAL is not flight-control, collision-avoidance, navigation, targeting or safety-critical software. A human operator remains responsible for aircraft operation and applicable aviation rules.

## Tests

```bash
python -m pytest -q
```

The CI workflow runs the unit suite on supported Python and operating-system combinations. Hardware-in-the-loop testing with the intended drone, camera, gimbal and telemetry source remains a separate release gate.

## Licence

MIT
