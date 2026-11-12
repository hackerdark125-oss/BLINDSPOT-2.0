# BLINDSPOT 2.0
### Visual Uncertainty Intelligence System

BlindSpot 2.0 is an experimental computer-vision research prototype that highlights image regions with weak visual-quality signals. It analyses recorded footage and bounded live camera captures, then produces explainable review indicators.

> **Important:** scores are heuristic image-quality indicators, not calibrated probabilities, semantic scene understanding, or proof that an area is visible.

## Features
- Analyse up to four recorded camera videos in one session
- Bounded live capture from a local webcam or RTSP/HTTP(S) stream
- Configurable OpenCV sampling interval and frame limit
- Per-cell confidence heatmaps with configurable grid resolution
- Brightness, contrast, sharpness, and edge-density metrics
- Confidence and uncertainty trends over time
- Low-confidence region prioritisation
- Multi-camera comparison and downloadable CSV reports
- Repeated-frame diagnostic that can flag a potentially frozen feed
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

The repeated-frame warning is only a diagnostic: a genuinely static scene can look like a frozen feed. Confirm suspected failures independently.

## Run tests
```bat
python -m pytest -q
```
GitHub Actions runs the same test command on pushes and pull requests.

## How the score works
Each sampled frame is divided into a configurable grid. Every cell receives a heuristic score from exposure balance, local contrast, Laplacian sharpness, and Canny edge density. Scores range from 0 to 100; low-scoring cells are prioritised for human review. This is not a trained AI model or calibrated probability.

## Roadmap
- [x] Recorded video ingestion and frame sampling
- [x] Explainable local image-quality maps
- [x] Multi-video dashboard and comparison
- [x] Confidence/uncertainty trends and CSV exports
- [x] Live webcam / network-stream capture
- [x] Repeated-frame / possible-freeze diagnostics
- [x] Input validation, unit tests, and CI
- [ ] Motion-aware stale-observation tracking with scene-change handling
- [ ] Camera calibration and time synchronisation
- [ ] Geometric coverage mapping using camera pose / visual localisation
- [ ] Validated uncertainty model and benchmark dataset

## Limitations and safety
- The current map measures image quality, not physical-world visibility or 3D coverage.
- Camera views are analysed independently; no calibration or cross-camera geometry is inferred.
- Texture and edge density do not guarantee useful semantic information.
- Static scenes may trigger the repeated-frame diagnostic; it is not proof of camera failure.
- Results are decision-support signals only. Do not use this prototype for flight control, navigation, targeting, or safety-critical inspection.
- No claim of novelty or operational readiness is made.

## Tech stack
Python · OpenCV · NumPy · Pandas · Plotly · Streamlit · Pytest · GitHub Actions

## License
MIT
