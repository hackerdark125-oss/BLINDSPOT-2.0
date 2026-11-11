from __future__ import annotations
import tempfile
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from blindspot.engine import sample_video, prioritise_cells
from blindspot.live import sample_stream

st.set_page_config(page_title="BLINDSPOT 2.1", page_icon="◉", layout="wide")
st.title("◉ BLINDSPOT")
st.caption("DRONE VISUAL-AWARENESS RESEARCH PROTOTYPE · VIDEO + LIVE STREAM")
st.info("This prototype measures image-quality signals (brightness, contrast, sharpness and edge detail). These are not physical visibility, obstacle detection, 3D coverage, or a flight-safety guarantee.")

with st.sidebar:
    st.header("Analysis settings")
    interval = st.slider("Sample interval (seconds)", 0.25, 5.0, 1.0, 0.25)
    max_frames = st.slider("Maximum frames", 5, 200, 60, 5)
    grid = st.selectbox("Confidence map resolution", [4, 6, 8, 12], index=2)
    st.caption("Higher grid resolution takes longer to process.")

def show_results(name, results):
    st.subheader(name)
    summary = pd.DataFrame([{
        "Time (s)": r.timestamp_seconds,
        "Confidence": r.metrics.confidence,
        "Uncertainty": r.metrics.uncertainty,
        "Brightness": r.metrics.mean_brightness,
        "Contrast": r.metrics.contrast,
        "Sharpness": r.metrics.sharpness,
        "Edge detail": r.metrics.edge_density,
    } for r in results])
    latest = results[-1]
    a, b, c = st.columns(3)
    a.metric("Mean image confidence", f"{summary['Confidence'].mean():.1f}/100")
    b.metric("Mean uncertainty", f"{summary['Uncertainty'].mean():.1f}/100")
    c.metric("Frames analysed", len(results))
    trend = px.line(summary, x="Time (s)", y=["Confidence", "Uncertainty"],
                    title="Visual-quality trend", markers=True)
    st.plotly_chart(trend, use_container_width=True)
    heat = px.imshow(latest.confidence_map, zmin=0, zmax=100,
                     color_continuous_scale="RdYlGn", origin="upper",
                     labels={"x":"Grid column", "y":"Grid row", "color":"Image score"})
    heat.update_layout(height=420, title="Latest frame · local image-quality map")
    st.plotly_chart(heat, use_container_width=True)
    st.write("**Frame-by-frame measurements**")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.write("**Regions to review · lowest image score first**")
    st.dataframe(pd.DataFrame(prioritise_cells(latest.confidence_map, 8)),
                 use_container_width=True, hide_index=True)

tab_upload, tab_live = st.tabs(["Recorded video", "Live camera / RTSP"])

with tab_upload:
    st.write("Upload up to four recordings. Phone footage is fine for a first test.")
    uploads = st.file_uploader("Add camera videos", type=["mp4", "avi", "mov", "mkv"],
                               accept_multiple_files=True, key="video_upload")
    if uploads:
        if len(uploads) > 4:
            st.warning("Only the first four videos will be analysed.")
        for i, upload in enumerate(uploads[:4]):
            suffix = Path(upload.name).suffix or ".mp4"
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(upload.getvalue())
                    temp_path = tmp.name
                with st.spinner(f"Analysing {upload.name}..."):
                    results = sample_video(temp_path, interval, max_frames, grid)
                show_results(f"Camera {i+1} · {upload.name}", results)
            except Exception as exc:
                st.error(f"Could not analyse {upload.name}: {exc}")
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)

with tab_live:
    st.warning("Live capture runs on the computer hosting this Streamlit app, not automatically on your phone. For a phone camera, use a trusted IP-camera app that provides an RTSP/HTTP stream reachable on the same network.")
    source_kind = st.radio("Source type", ["Host computer webcam", "RTSP / HTTP stream"], horizontal=True)
    if source_kind == "Host computer webcam":
        source = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
    else:
        source = st.text_input("Stream URL", placeholder="rtsp://camera-address:554/stream")
        st.caption("Avoid sharing stream URLs containing passwords. Do not expose a camera stream to the public internet.")
    duration = st.slider("Capture duration (seconds)", 5, 60, 15, 5)
    if st.button("Start live analysis", type="primary", disabled=(source_kind == "RTSP / HTTP stream" and not source.strip())):
        try:
            with st.spinner("Connecting and sampling live frames..."):
                live_results = sample_stream(source if source_kind == "RTSP / HTTP stream" else int(source),
                                              duration, interval, max_frames, grid)
            show_results("Live source analysis", live_results)
        except Exception as exc:
            st.error(f"Live capture failed: {exc}")

st.divider()
st.caption("Research prototype only · No drone control · No calibrated probability · No obstacle or collision detection.")
