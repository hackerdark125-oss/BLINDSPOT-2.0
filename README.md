# BLINDSPOT 2.0
### Visual Uncertainty Intelligence System

BlindSpot 2.0 is an experimental computer-vision system that helps identify image regions where visual evidence may be weak, missing, or unreliable. It analyses recorded camera footage and turns local image-quality measurements into explainable review signals.

> **Important:** confidence scores are heuristic image-quality indicators, not calibrated probabilities, semantic understanding, or proof that an area is visible.

## Features
- Analyse up to four recorded camera videos in one session
- OpenCV frame sampling with configurable interval and frame limit
- Per-cell confidence maps with configurable grid resolution
- Explainable brightness, contrast, sharpness, and edge-density metrics
- Confidence and uncertainty trends over video time
- Weak-region prioritisation for the latest sampled frame
- Multi-camera comparison table
- Downloadable per-camera and comparison CSV reports
- Unit tests and automated GitHub Actions test workflow

## Windows setup
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit, usually http://localhost:8501.

## Run tests
```bat
python -m pytest -q
```

## How the score works
Each sampled frame is divided into a configurable grid. Every cell receives a heuristic score based on:
- Exposure balance (brightness relative to mid-range)
- Local contrast
- Laplacian sharpness
- Canny edge density

The weighted score is clipped to 0–100. Lower scores are prioritised for human review. This is an explainable baseline and must not be interpreted as a trained AI model or calibrated probability.

## Current scope and roadmap
- [x] Real video ingestion and frame sampling
- [x] Explainable local visual-quality map
- [x] Multi-video dashboard
- [x] Confidence/uncertainty trend charts
- [x] CSV exports and camera comparison
- [x] Input validation and unit-test coverage
- [x] Automated tests on GitHub Actions
- [ ] Live webcam / RTSP capture in the published repository
- [ ] Camera calibration and time synchronisation
- [ ] Motion-aware stale-observation tracking
- [ ] Geometric coverage mapping using camera pose / visual localisation
- [ ] Validated uncertainty model and benchmark dataset

## Limitations and safety
- The current map measures image quality, not physical-world visibility or 3D coverage.
- Camera views are analysed independently; no calibration or cross-camera geometry is inferred.
- Blur, low contrast, darkness, and texture can affect scores; high texture does not necessarily mean useful scene information.
- Results are decision-support signals only. Do not use this prototype for flight control, navigation, targeting, or safety-critical inspection.
- No claim of novelty or operational readiness is made.

## Tech stack
Python · OpenCV · NumPy · Pandas · Plotly · Streamlit · Pytest · GitHub Actions

## License
MIT
