# BLINDSPOT AERIAL deployment guide

## Local installation

Follow the Windows setup in the main README, then install the optional AI stack:

```bat
python -m pip install -r requirements-ai.txt
streamlit run app.py
```

The first YOLO run downloads the selected weights. GPU acceleration depends on a compatible PyTorch/CUDA installation.

## Docker deployment

Build the base runtime image:

```bash
docker build -t blindspot-aerial .
docker run --rm --name blindspot-aerial -p 127.0.0.1:8501:8501 blindspot-aerial
```

The default Dockerfile installs the lightweight runtime dependencies. For YOLO inference, build a derived image that installs `requirements-ai.txt` using a PyTorch build appropriate for the target CPU/GPU. AI dependencies and model weights are intentionally separate from the lightweight CI image.

Open http://localhost:8501.

A live drone stream must be reachable from inside the container. USB/CSI camera passthrough and GPU device access require platform-specific configuration and are not enabled by the default container command.

## Streamlit hosting

The Streamlit process must be able to reach the drone or ground-station video stream. A hosted app cannot automatically access a drone video feed that is only available on a private local network. Use a secure VPN/private network or run BLINDSPOT AERIAL on the local ground-station computer.

## Security and privacy

- Never commit RTSP credentials, private stream URLs, API keys or flight logs containing sensitive locations.
- Restrict network access to trusted drone/ground-station devices.
- Protect uploaded footage, telemetry and exported point clouds as potentially sensitive operational data.
- Add authentication, access controls and retention policies before multi-user or customer deployment.
- Do not expose the Streamlit port publicly without a secured reverse proxy and firewall policy.

## Resource sizing

YOLO inference and stereo depth are compute-intensive. Start with YOLO11n, 640-pixel inference, a conservative frame stride and short test clips. Record CPU/GPU, memory, latency and thermal behaviour before increasing throughput.

## Reproducible deployment record

For each deployment, record the Git commit SHA, CI run, image digest, host hardware, operating system, Python/PyTorch/Ultralytics versions, model-weight identifier, stream codec/resolution and benchmark reports. Never include credential-bearing URLs or private flight footage in public artifacts.

The Docker health check verifies that Streamlit responds. It does not verify drone-stream connectivity, detection accuracy, telemetry alignment or mapping quality.
