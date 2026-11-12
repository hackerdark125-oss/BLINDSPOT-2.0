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
- [x] Input validation, unit tests, and CI
- [ ] Scene-change-aware activity baselines and robust camera-motion compensation
- [ ] Camera calibration and time synchronisation
- [ ] Geometric multi-camera coverage estimation using camera pose / visual localisation
- [ ] Validated uncertainty model and benchmark dataset

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
