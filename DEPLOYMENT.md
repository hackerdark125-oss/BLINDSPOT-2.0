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

Run the dashboard:

```bash
docker run --rm -p 8501:8501 blindspot-2
```

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
