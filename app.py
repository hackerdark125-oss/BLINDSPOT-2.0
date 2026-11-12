from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from blindspot.engine import sample_video, prioritise_cells

st.set_page_config(page_title="BlindSpot 2.0", page_icon="◉", layout="wide")
st.title("◉ BLINDSPOT 2.0")
st.caption("VISUAL UNCERTAINTY INTELLIGENCE SYSTEM · REAL VIDEO ANALYSIS")
st.info(
    "Upload up to four camera recordings. Scores describe local image quality as a "
    "proxy for review priority; they are not calibrated probabilities or proof of visibility."
)

with st.sidebar:
    st.header("Analysis settings")
    interval = st.slider("Sample interval (seconds)", 0.25, 5.0, 1.0, 0.25)
    max_frames = st.slider("Maximum frames per camera", 5, 200, 60, 5)
    grid = st.selectbox("Confidence map resolution", [4, 6, 8, 12], index=2)
    review_count = st.slider("Weak regions to show", 3, 20, 5)
    st.caption("Higher map resolution and frame limits increase processing time.")

uploads = st.file_uploader(
    "Add camera videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True
)

if uploads:
    if len(uploads) > 4:
        st.warning("Only the first four videos will be analysed.")

    camera_summaries = []
    for i, upload in enumerate(uploads[:4]):
        st.subheader(f"Camera {i+1} · {upload.name}")
        suffix = Path(upload.name).suffix or ".mp4"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getvalue())
                temp_path = tmp.name
            try:
                with st.spinner(f"Analysing {upload.name}..."):
                    results = sample_video(temp_path, interval, max_frames, grid)
            finally:
                Path(temp_path).unlink(missing_ok=True)

            summary = pd.DataFrame([{
                "Time (s)": r.timestamp_seconds,
                "Confidence": r.metrics.confidence,
                "Uncertainty": r.metrics.uncertainty,
                "Brightness": r.metrics.mean_brightness,
                "Contrast": r.metrics.contrast,
                "Sharpness": r.metrics.sharpness,
                "Edge density": r.metrics.edge_density,
            } for r in results])
            camera_summaries.append({
                "Camera": upload.name,
                "Mean confidence": summary["Confidence"].mean(),
                "Mean uncertainty": summary["Uncertainty"].mean(),
                "Frames sampled": len(results),
            })

            latest = results[-1]
            a, b, c = st.columns(3)
            a.metric("Mean confidence", f"{summary['Confidence'].mean():.1f}/100")
            b.metric("Mean uncertainty", f"{summary['Uncertainty'].mean():.1f}/100")
            c.metric("Frames sampled", len(results))

            trend = px.line(
                summary, x="Time (s)", y=["Confidence", "Uncertainty"],
                title="Confidence and uncertainty over time",
                markers=True,
            )
            trend.update_layout(height=330, legend_title_text="Metric")
            st.plotly_chart(trend, use_container_width=True)

            heatmap = px.imshow(
                latest.confidence_map, zmin=0, zmax=100,
                color_continuous_scale="RdYlGn", origin="upper",
                labels={"x": "Grid column", "y": "Grid row", "color": "Confidence"},
                title="Latest frame · local confidence map",
            )
            heatmap.update_layout(height=360)
            st.plotly_chart(heatmap, use_container_width=True)

            with st.expander("Frame-by-frame measurements", expanded=False):
                st.dataframe(summary, use_container_width=True, hide_index=True)
            st.download_button(
                "Download camera report (CSV)",
                data=summary.to_csv(index=False).encode("utf-8"),
                file_name=f"{Path(upload.name).stem}_blindspot_report.csv",
                mime="text/csv",
                key=f"download_{i}",
            )

            weak = prioritise_cells(latest.confidence_map, review_count)
            st.write("**Regions to review · lowest confidence first**")
            st.dataframe(pd.DataFrame(weak), use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Could not analyse {upload.name}: {exc}")

    if camera_summaries:
        st.divider()
        st.subheader("Camera comparison")
        comparison = pd.DataFrame(camera_summaries)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.download_button(
            "Download camera comparison (CSV)",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="blindspot_camera_comparison.csv",
            mime="text/csv",
            key="download_comparison",
        )
        st.success(f"Analysis completed for {len(camera_summaries)} camera feed(s).")
else:
    st.markdown("### How to begin")
    st.write("Choose one or more short video files above. Start with a phone recording or sample video.")
    st.write("The dashboard provides confidence trends, a local heatmap, weak-region priorities, and downloadable CSV reports.")

st.divider()
st.caption(
    "Research prototype only · No drone control · No calibrated probability or safety guarantee."
)
