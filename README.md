# BLINDSPOT 2.0
### Visual Uncertainty Intelligence System

BlindSpot 2.0 is an experimental computer-vision system that measures the quality and coverage of visual observations. Its central question is: **where is the visual evidence weak, missing, or unreliable?**

The first milestone analyses real video files from up to four camera viewpoints. It samples frames, estimates local image quality, and builds per-view confidence maps. Image quality is only a proxy for observation confidence, not proof that a scene area has been understood.

## Current features
- Analyse up to four real camera videos
- OpenCV frame sampling
- Local confidence maps
- Blur/sharpness, brightness, contrast, and edge-detail measurements
- Per-camera uncertainty summaries
- Low-confidence regions prioritised for review
- Interactive Streamlit dashboard
- Unit tests for the analysis engine

## Run on Windows
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## Run tests
```bat
python -m pytest -q
```

## How the confidence map works
Each sampled frame is divided into a configurable grid. Every cell receives a heuristic score based on luminance, contrast, Laplacian sharpness, and edge density. Scores range from 0 to 100; low-scoring cells are highlighted for review. This is an explainable baseline, not a trained AI model or calibrated probability.

## Roadmap
- [x] Real video ingestion and frame sampling
- [x] Explainable local visual-quality map
- [ ] Live webcam / RTSP capture
- [ ] Camera calibration and time synchronisation
- [ ] Motion-aware stale-observation tracking
- [ ] Geometric coverage mapping using camera pose / visual localisation
- [ ] Validated uncertainty model and benchmark dataset

## Limitations and safety
- The current map measures image quality, not physical-world visibility or 3D coverage.
- Camera views are analysed independently; no camera calibration or cross-camera geometry is inferred.
- Results are decision-support signals only. Do not use this prototype for flight control, navigation, targeting, or safety-critical inspection.
- No claim of novelty or operational readiness is made.

## Tech stack
Python · OpenCV · NumPy · Pandas · Plotly · Streamlit · Pytest

## License
MIT
