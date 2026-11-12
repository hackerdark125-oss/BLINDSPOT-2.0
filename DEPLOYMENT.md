# Deployment guide

## Run locally
Follow the Windows setup in the main README, then run:

```bat
streamlit run app.py
```

## Deploy with Docker

Build the image from the repository root:

```bash
docker build -t blindspot-2 .
```

Run the dashboard (bind to localhost by default so it is not exposed to the LAN):

```bash
docker run --rm --name blindspot-2 -p 127.0.0.1:8501:8501 blindspot-2
```

For a server deployment, choose an explicit network binding and firewall rule rather than exposing the dashboard unintentionally. Use a reverse proxy and access control for remote users. The container health check only verifies that Streamlit responds; it does not verify camera connectivity or reconstruction accuracy.

Open http://localhost:8501.

For a network camera, the RTSP/HTTP stream must be reachable from inside the container. Local webcam device passthrough requires platform-specific device access and is not enabled by the default container command.

## Deploy on Streamlit Community Cloud
1. Push the repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select `hackerdark125-oss/BLINDSPOT-2.0`, branch `main`, and file `app.py`.
4. Deploy using the repository's `requirements.txt`.

A hosted Streamlit app cannot automatically access a webcam attached to a visitor's personal computer. Live capture runs where the Streamlit Python process runs. Use a network camera reachable by the deployed server, or run the app locally.

## Camera stream security
- Never commit RTSP credentials, API keys, or private stream URLs.
- Keep secrets in the hosting platform's secret manager or environment variables.
- Restrict access to private camera networks; do not expose camera ports to the public internet.
- The application currently accepts a stream URL in the UI; use only trusted streams.

## Resource sizing
Video analysis, optical flow, and stereo depth estimation are CPU- and memory-intensive. Start with short clips, a 1-second sample interval, a 4x4 or 8x8 grid, and a modest frame limit. Increase settings only after observing resource use.


## Benchmark and reconstruction utilities

The dashboard remains the primary application entry point. Advanced reconstruction APIs are also available to Python callers:

- `blindspot.pose_graph_se3.optimize_pose_graph_se3`: short-trajectory full SE(3) refinement with independently verified relative-pose constraints.
- `blindspot.registration.register_point_cloud_icp`: local rigid alignment for overlapping point clouds; supply a reasonable initial transform and inspect fitness/RMSE.
- `scripts/benchmark_trajectory.py`: evaluate exported estimated poses against an independent ground-truth JSON trajectory.

These tools run inside the same Python environment/container. For large point clouds or long trajectories, size CPU and memory accordingly. Do not expose camera streams publicly; use trusted network paths and keep stream credentials outside source control.


The dashboard's advanced reconstruction workflows are available under
**Camera calibration**. Build the Docker image after updating the repository;
the image health check validates Streamlit startup, not connected camera access
or reconstruction accuracy. Camera rig JSON, PLY inputs, and calibration files
should be retained with the output reports for reproducibility.


## Reproducible deployment record

For each deployment, record:

- Git commit SHA and Docker image tag/digest.
- CI workflow run URL and result for that exact commit.
- Runtime host, operating system, CPU/RAM limits, and Python/container image.
- Camera stream type and resolution/FPS (never the credential-bearing URL).
- Calibration file identifiers and distance units.
- Benchmark input identifiers and output reports.

Do not include RTSP passwords, access tokens, private camera URLs, or personal footage in public logs or release artifacts.
