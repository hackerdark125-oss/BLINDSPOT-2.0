from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from blindspot.engine import sample_video, prioritise_cells
from blindspot.live import capture_live_source

st.set_page_config(page_title="BlindSpot 2.0", page_icon="◉", layout="wide")
st.title("◉ BLINDSPOT 2.0")
st.caption("VISUAL UNCERTAINTY INTELLIGENCE SYSTEM · VIDEO & LIVE CAMERA ANALYSIS")
st.info(
    "Image-quality scores are heuristic review signals, not calibrated probabilities. "
    "Motion activity age measures time since measurable pixel motion, not whether an area "
    "was observed. A static scene can naturally have a high activity age."
)

with st.sidebar:
    st.header("Analysis settings")
    interval = st.slider("Sample interval (seconds)", 0.25, 5.0, 1.0, 0.25)
    max_frames = st.slider("Maximum frames", 5, 200, 60, 5)
    grid = st.selectbox("Confidence map resolution", [4, 6, 8, 12], index=2)
    review_count = st.slider("Weak regions to show", 3, 20, 5)

mode = st.radio("Analysis source", ["Recorded videos", "Live webcam / RTSP"], horizontal=True)


def render_results(name: str, results, live: bool = False) -> pd.DataFrame:
    analyses = [r.analysis if live else r for r in results]
    summary = pd.DataFrame([{
        "Time (s)": a.timestamp_seconds,
        "Confidence": a.metrics.confidence,
        "Uncertainty": a.metrics.uncertainty,
        "Brightness": a.metrics.mean_brightness,
        "Contrast": a.metrics.contrast,
        "Sharpness": a.metrics.sharpness,
        "Edge density": a.metrics.edge_density,
        "Mean motion": a.mean_motion,
        "Maximum activity age (s)": (
            float(a.motion_age_seconds.max()) if a.motion_age_seconds is not None else None
        ),
    } for a in analyses])
    if live:
        summary["Frame change"] = [r.frame_change_score for r in results]
        summary["Possible frozen feed"] = [r.possible_frozen_feed for r in results]

    a, b, c, d = st.columns(4)
    a.metric("Mean confidence", f"{summary['Confidence'].mean():.1f}/100")
    b.metric("Mean uncertainty", f"{summary['Uncertainty'].mean():.1f}/100")
    c.metric("Frames sampled", len(results))
    d.metric("Mean motion", f"{summary['Mean motion'].mean():.2f}")

    trend = px.line(summary, x="Time (s)", y=["Confidence", "Uncertainty"],
                    title="Confidence and uncertainty over time", markers=True)
    trend.update_layout(height=330, legend_title_text="Metric")
    st.plotly_chart(trend, use_container_width=True)

    latest = analyses[-1]
    heatmap = px.imshow(
        latest.confidence_map, zmin=0, zmax=100,
        color_continuous_scale="RdYlGn", origin="upper",
        labels={"x": "Grid column", "y": "Grid row", "color": "Confidence"},
        title="Latest sampled frame · local confidence map",
    )
    heatmap.update_layout(height=360)
    st.plotly_chart(heatmap, use_container_width=True)

    if latest.motion_magnitude_map is not None and latest.motion_age_seconds is not None:
        motion_col, age_col = st.columns(2)
        with motion_col:
            motion_fig = px.imshow(
                latest.motion_magnitude_map, origin="upper",
                color_continuous_scale="Viridis",
                labels={"x": "Grid column", "y": "Grid row", "color": "Flow magnitude"},
                title="Dense optical-flow activity",
            )
            motion_fig.update_layout(height=330)
            st.plotly_chart(motion_fig, use_container_width=True)
        with age_col:
            age_fig = px.imshow(
                latest.motion_age_seconds, origin="upper",
                color_continuous_scale="YlOrRd",
                labels={"x": "Grid column", "y": "Grid row", "color": "Seconds"},
                title="Time since measurable motion · not observation freshness",
            )
            age_fig.update_layout(height=330)
            st.plotly_chart(age_fig, use_container_width=True)

    st.write("**Regions to review · lowest confidence first**")
    st.dataframe(pd.DataFrame(prioritise_cells(latest.confidence_map, review_count)),
                 use_container_width=True, hide_index=True)
    with st.expander("Frame-by-frame measurements"):
        st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV report", summary.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(name).stem}_blindspot_report.csv",
        mime="text/csv", key=f"download_{name}_{'live' if live else 'video'}",
    )
    return summary


if mode == "Recorded videos":
    uploads = st.file_uploader(
        "Add up to four camera videos", type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
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
                summary = render_results(upload.name, results)
                camera_summaries.append({
                    "Camera": upload.name,
                    "Mean confidence": summary["Confidence"].mean(),
                    "Mean uncertainty": summary["Uncertainty"].mean(),
                    "Mean motion": summary["Mean motion"].mean(),
                    "Max activity age (s)": summary["Maximum activity age (s)"].max(),
                    "Frames sampled": len(results),
                })
            except Exception as exc:
                st.error(f"Could not analyse {upload.name}: {exc}")
        if camera_summaries:
            st.divider()
            st.subheader("Camera comparison")
            comparison = pd.DataFrame(camera_summaries)
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            st.download_button(
                "Download camera comparison (CSV)",
                comparison.to_csv(index=False).encode("utf-8"),
                file_name="blindspot_camera_comparison.csv",
                mime="text/csv", key="download_comparison",
            )
            st.success(f"Analysis completed for {len(camera_summaries)} camera feed(s).")
    else:
        st.write("Upload one or more short recordings to begin.")

else:
    st.subheader("Live camera capture")
    st.warning(
        "This captures from the machine running Streamlit. For a remote deployment, "
        "the camera must be reachable by that server. A static scene can resemble a frozen feed."
    )
    source_text = st.text_input("Camera index or RTSP/HTTP(S) stream URL", value="0")
    duration = st.slider("Capture duration (seconds)", 3, 30, 10)
    if st.button("Start live analysis", type="primary"):
        try:
            with st.spinner("Connecting and sampling live frames..."):
                results = capture_live_source(
                    source_text, duration_seconds=float(duration),
                    sample_every_seconds=interval, max_samples=max_frames,
                    grid_size=grid,
                )
            st.success(f"Captured {len(results)} samples.")
            if any(sample.possible_frozen_feed for sample in results):
                st.warning(
                    "Possible repeated/frozen frames detected. This can also happen "
                    "when the camera is pointed at a completely static scene."
                )
            render_results("live_camera", results, live=True)
            st.metric("Samples with possible frozen feed", sum(s.possible_frozen_feed for s in results))
        except Exception as exc:
            st.error(f"Live capture failed: {exc}")

st.divider()
st.caption(
    "Research prototype only · No drone control · No calibrated probability or safety guarantee."
)
