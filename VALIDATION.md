# BLINDSPOT AERIAL — Release validation checklist

This checklist separates automated software checks from drone hardware/field validation. Passing CI does not prove aerial detection accuracy, mapping accuracy, stream reliability or safe aircraft operation.

## 1. Automated checks

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-ai.txt
python -m compileall -q app.py drone_app.py blindspot tests
python -m pytest -q
streamlit run app.py
```

- [ ] Python compilation succeeds.
- [ ] All unit tests pass on supported Python/OS combinations.
- [ ] Docker runtime image builds and starts.
- [ ] BLINDSPOT AERIAL opens with drone-only navigation and no CCTV workflows.
- [ ] Optional YOLO weights download and load successfully.

## 2. Recorded aerial video detection

Use representative drone footage at multiple heights, speeds, camera angles and lighting conditions.

- [ ] Upload MP4/MOV/AVI/MKV/M4V footage and confirm processing completes.
- [ ] Confirm people, vehicles and selected animal classes are detected where visible.
- [ ] Confirm bounding boxes, class labels, confidence values and track IDs render.
- [ ] Confirm a continuously visible object retains its track ID across sampled frames.
- [ ] Record ID switches, false positives, missed detections and inference latency.
- [ ] Download the detection CSV and verify timestamps, frame indices and pixel-space boxes.
- [ ] Repeat for small/distant objects, motion blur, partial occlusion, shadows and low light.
- [ ] Confirm a second video starts a fresh tracker session.

## 3. Live drone video stream

- [ ] Connect a supported RTSP/RTSPS/HTTP/HTTPS stream from the host running BLINDSPOT.
- [ ] Record stream codec, resolution, source FPS, network latency and reconnect behaviour.
- [ ] Confirm processing is bounded by configured duration/sample count.
- [ ] Disconnect the stream and record error/recovery behaviour.
- [ ] Keep stream credentials out of screenshots, logs, Git history and reports.
- [ ] Repeat on the actual drone/ground-station video link; a synthetic or desktop stream is not hardware validation.

## 4. Flight telemetry

- [ ] Export telemetry with timestamp_seconds, latitude_deg, longitude_deg and altitude_m.
- [ ] Confirm video and telemetry use the same timestamp origin.
- [ ] Confirm timestamps are strictly increasing and GPS/altitude ranges are valid.
- [ ] Compare the displayed GPS track with the flight-controller log.
- [ ] Verify the altitude vertical datum (relative, ellipsoid, or mean-sea-level) and document it.
- [ ] Compare interpolated telemetry with known samples at selected timestamps.
- [ ] Confirm the app rejects extrapolation beyond the telemetry time range.

## 5. Stereo depth and 3D mapping

- [ ] Calibrate the actual stereo camera pair at its operating resolution and focus.
- [ ] Keep the stereo baseline rigid and synchronise both camera feeds.
- [ ] Generate a depth map using the existing stereo pipeline; confirm its distance unit is metres for the drone georeferencing workflow.
- [ ] Compare depth against independently measured targets at multiple distances.
- [ ] Confirm camera intrinsics match the depth map dimensions.
- [ ] Measure camera-to-gimbal and gimbal-to-body extrinsics; derive the camera-to-ENU rotation for the recorded gimbal attitude.
- [ ] Confirm the first telemetry GPS fix is an appropriate local ENU origin.
- [ ] Generate the PLY and inspect it in a point-cloud viewer.
- [ ] Compare the point cloud against surveyed reference geometry and record horizontal/vertical error.
- [ ] Repeat with different altitude, pitch, roll, yaw, gimbal angles and flight directions.
- [ ] Document GPS quality, telemetry delay, timestamp offset, stereo depth error and camera mounting offsets.

The current dashboard exports a georeferenced point cloud from a supplied depth map. It does not yet perform complete multi-frame aerial photogrammetry, dense image-based mapping, automatic gimbal calibration, SLAM relocalisation or global map optimisation.

## 6. Performance and field evidence

For each supported hardware configuration, archive:
- Drone/camera/gimbal model and firmware.
- Video resolution, source FPS, codec and stream transport.
- Computer/edge device, CPU/GPU, RAM and power configuration.
- YOLO model/weights, image size, confidence threshold and processed-frame stride.
- Per-class precision/recall on manually labelled aerial footage.
- Mean/p95 inference latency, processed FPS, stream drop rate and thermal behaviour.
- Stereo calibration, measured baseline, depth error and valid-pixel coverage.
- GPS/altitude source, datum, telemetry timing error and mapping residuals.
- Sample input footage, anonymised screenshots and exported CSV/PLY reports.

## Completion definition

Software-complete means the documented drone workflows run and automated checks pass. Field-validated means the actual drone, camera, gimbal, video link and telemetry have been tested against independent measurements. Do not claim operational accuracy or safety-critical capability without that evidence.

BLINDSPOT AERIAL is not flight-control, collision-avoidance, navigation, targeting or autonomous-piloting software.
