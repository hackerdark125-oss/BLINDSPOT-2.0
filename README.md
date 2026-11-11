# BLINDSPOT

### Drone Visual-Awareness Research Prototype

BLINDSPOT analyses recorded video and optional live webcam/RTSP streams to highlight **weak image-quality regions** that may deserve operator review.

It is a research prototype, not a drone autopilot, obstacle-avoidance system, calibrated visibility estimator, or certified safety tool.

## Current features

- Upload and analyse up to four video files (MP4, AVI, MOV, MKV)
- Sample video frames at a configurable interval
- Per-grid-cell heuristic image-quality scores
- Brightness, contrast, Laplacian sharpness, and edge-detail measurements
- Confidence/uncertainty trend charts and a heatmap
- Ranked low-scoring image cells for manual review
- Optional bounded live capture from the host computer webcam
- Optional RTSP/HTTP stream capture from a reachable camera
- Unit tests for image analysis, prioritisation, and temporal frame-change utility

## Run on Windows

Python 3.10 or 3.11 is recommended for broad binary-wheel compatibility.

```bat
git clone https://github.com/hackerdark125-oss/BLINDSPOT-2.0.git
cd BLINDSPOT-2.0
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If Python 3.11 is not installed, install it from python.org and retry. Open the local URL printed by Streamlit, usually http://localhost:8501.

## Run tests

```bat
python -m pytest -q
```

## Use recorded drone footage

1. Record a short video from a drone camera, or use phone footage as a test input.
2. Transfer the MP4/MOV file to the computer running BLINDSPOT.
3. Open the **Recorded video** tab and upload it.
4. Review the trend chart, latest heatmap, and lowest-scoring grid cells.

Phone footage simulates camera input; it is not a live connection to a drone.

## Use a live camera

Open **Live camera / RTSP**:
- **Host computer webcam** reads a webcam connected to the computer running Streamlit.
- **RTSP / HTTP stream** accepts a camera stream URL reachable from that computer.

For a phone camera, an IP-camera app may expose an RTSP/HTTP stream on the same trusted Wi-Fi network. Configure the app and network yourself; stream URL formats vary by app. Do not expose an unauthenticated camera stream publicly. Live capture is bounded by the selected duration.

## How the score works

Each sampled frame is divided into a configurable grid. Each cell receives a heuristic score based on exposure, local contrast, Laplacian sharpness, and edge density. Scores are scaled to 0–100. The dashboard reports the average and highlights the lowest-scoring cells.

**Confidence is a label for this image-quality heuristic only.** It is not a calibrated probability that an obstacle is visible, detected, or safely avoidable.

## What is not implemented

- Object detection for trees, wires, buildings, people, or other obstacles
- Physical blind-spot inference or 3D scene/coverage reconstruction
- Camera calibration, pose estimation, or multi-camera geometry
- Drone telemetry, flight-controller integration, or autonomous control
- Validated collision-risk prediction or safety guarantees

## Roadmap

- [x] Recorded video ingestion and frame sampling
- [x] Explainable local image-quality heatmap
- [x] Per-frame trend dashboard and weak-cell prioritisation
- [x] Bounded webcam / RTSP sampling
- [ ] Validated motion-aware visual tracking
- [ ] Camera calibration and pose-aware 3D coverage
- [ ] Evaluated object-detection module with documented model and dataset
- [ ] Multi-camera synchronisation and geometric fusion
- [ ] Benchmark dataset, metrics, and field validation

## Tech stack

Python · OpenCV · NumPy · Pandas · Plotly · Streamlit · Pytest

## Safety

Use only as an experimental visual-review aid. Do not rely on it for flight control, navigation, obstacle avoidance, targeting, or safety-critical decisions. Test in a controlled environment and keep a human operator responsible for all flight decisions.

## License

MIT
