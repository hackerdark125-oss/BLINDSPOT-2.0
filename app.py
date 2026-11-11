from __future__ import annotations
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from blindspot.engine import sample_video, prioritise_cells

st.set_page_config(page_title="BlindSpot 2.0", page_icon="◉", layout="wide")
st.title("◉ BLINDSPOT 2.0")
st.caption("VISUAL UNCERTAINTY INTELLIGENCE SYSTEM · REAL VIDEO ANALYSIS")
st.info("Upload up to four camera recordings. The current engine scores local image quality as a proxy for visual confidence; it does not infer physical visibility or 3D coverage.")
with st.sidebar:
    st.header("Analysis settings")
    interval = st.slider("Sample interval (seconds)", 0.25, 5.0, 1.0, 0.25)
    max_frames = st.slider("Maximum frames per camera", 5, 200, 60, 5)
    grid = st.selectbox("Confidence map resolution", [4, 6, 8, 12], index=2)
    st.caption("Higher resolution increases processing time.")
uploads = st.file_uploader("Add camera videos", type=["mp4","avi","mov","mkv"], accept_multiple_files=True)
if uploads:
    if len(uploads) > 4:
        st.warning("Only the first four videos will be analysed.")
    all_results = []
    for i, upload in enumerate(uploads[:4]):
        st.subheader(f"Camera {i+1} · {upload.name}")
        suffix = Path(upload.name).suffix or ".mp4"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getvalue()); temp_path = tmp.name
            try:
                with st.spinner(f"Analysing {upload.name}..."):
                    results = sample_video(temp_path, interval, max_frames, grid)
            finally:
                Path(temp_path).unlink(missing_ok=True)
            all_results.append((upload.name, results))
            summary = pd.DataFrame([{"Time (s)":r.timestamp_seconds,
                "Confidence":r.metrics.confidence,"Uncertainty":r.metrics.uncertainty,
                "Brightness":r.metrics.mean_brightness,"Contrast":r.metrics.contrast,
                "Sharpness":r.metrics.sharpness} for r in results])
            latest = results[-1]
            a,b,c = st.columns(3)
            a.metric("Mean confidence", f"{summary['Confidence'].mean():.1f}/100")
            b.metric("Mean uncertainty", f"{summary['Uncertainty'].mean():.1f}/100")
            c.metric("Frames sampled", len(results))
            fig = px.imshow(latest.confidence_map, zmin=0, zmax=100,
                color_continuous_scale="RdYlGn", origin="upper",
                labels={"x":"Grid column","y":"Grid row","color":"Confidence"})
            fig.update_layout(height=360, title="Latest frame · local confidence map")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(summary, use_container_width=True, hide_index=True)
            weak = prioritise_cells(latest.confidence_map, 5)
            st.write("**Regions to review (lowest confidence first)**")
            st.dataframe(pd.DataFrame(weak), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Could not analyse {upload.name}: {exc}")
    if all_results:
        st.success(f"Completed analysis for {len(all_results)} camera feed(s).")
else:
    st.markdown("### How to begin")
    st.write("Choose one or more short video files above. You can start with a phone recording or a sample video.")
    st.write("The dashboard will show a per-frame confidence trend, a local heatmap, and the weakest image regions.")
st.divider()
st.caption("Research prototype only · No drone control · No calibrated probability or safety guarantee.")
