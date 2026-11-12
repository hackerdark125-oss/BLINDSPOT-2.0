from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from blindspot.engine import sample_video, prioritise_cells
from blindspot.live import capture_live_source
from blindspot.calibration import calibrate_camera, load_calibration_json, undistort_frame
from blindspot.stereo import (
    calibrate_stereo_cameras, load_stereo_calibration_json,
    rectify_and_compute_depth,
)

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

mode = st.radio("Analysis source", ["Recorded videos", "Live webcam / RTSP", "Camera calibration"], horizontal=True)


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

elif mode == "Live webcam / RTSP":
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



else:
    st.subheader("Camera calibration")
    calibration_mode = st.radio(
        "Calibration workflow",
        ["Single-camera intrinsics", "Two-camera stereo extrinsics", "Stereo depth preview"],
        horizontal=True,
    )
    if calibration_mode == "Single-camera intrinsics":
        st.write(
            "Upload multiple photos of the same printed chessboard taken from different "
            "positions and angles. Calibration estimates camera intrinsics and lens distortion."
        )
        st.warning(
            "Use the number of INNER chessboard corners, not the number of squares. "
            "All calibration images must have the same resolution."
        )
        calibration_files = st.file_uploader(
            "Chessboard calibration photos", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True, key="calibration_photos",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            board_columns = st.number_input("Inner corners · columns", min_value=2, max_value=20, value=9)
        with c2:
            board_rows = st.number_input("Inner corners · rows", min_value=2, max_value=20, value=6)
        with c3:
            square_size = st.number_input("Square size (consistent units)", min_value=0.001, value=1.0, step=0.1)
        if st.button("Calibrate camera", type="primary", disabled=not calibration_files):
            decoded = []
            for uploaded in calibration_files:
                raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
                image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if image is not None:
                    decoded.append(image)
            try:
                with st.spinner("Detecting chessboard corners and estimating camera parameters..."):
                    calibration = calibrate_camera(
                        decoded,
                        board_size=(int(board_columns), int(board_rows)),
                        square_size=float(square_size),
                    )
                st.success(
                    f"Calibration complete · {calibration.accepted_views} usable views · "
                    f"{calibration.rejected_views} rejected"
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Reprojection RMS", f"{calibration.rms_reprojection_error:.4f}")
                m2.metric("Image width", calibration.image_width)
                m3.metric("Image height", calibration.image_height)
                st.json({
                    "camera_matrix": calibration.camera_matrix,
                    "distortion_coefficients": calibration.distortion_coefficients,
                })
                st.download_button(
                    "Download calibration JSON",
                    calibration.to_json().encode("utf-8"),
                    file_name="blindspot_camera_calibration.json",
                    mime="application/json",
                )
                preview_file = st.selectbox(
                    "Choose an uploaded photo to preview undistortion",
                    options=list(range(len(calibration_files))),
                    format_func=lambda index: calibration_files[index].name,
                )
                raw = np.frombuffer(calibration_files[preview_file].getvalue(), dtype=np.uint8)
                original = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if original is not None and (original.shape[1], original.shape[0]) == (
                    calibration.image_width, calibration.image_height
                ):
                    corrected = undistort_frame(original, calibration)
                    left, right = st.columns(2)
                    with left:
                        st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original")
                    with right:
                        st.image(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB), caption="Undistorted")
                else:
                    st.info("The selected preview must match the calibration image resolution.")
            except Exception as exc:
                st.error(f"Calibration failed: {exc}")
    elif calibration_mode == "Two-camera stereo extrinsics":
        st.write(
            "Upload synchronized chessboard image pairs from two fixed cameras. "
            "Each A image and B image at the same index must show the same board pose."
        )
        st.warning(
            "First calibrate each camera separately and upload its JSON. Both cameras "
            "must use the same resolution, board geometry, and square-size unit."
        )
        left, right = st.columns(2)
        with left:
            intrinsics_a_file = st.file_uploader(
                "Camera A intrinsics JSON", type=["json"], key="intrinsics_a"
            )
            images_a_files = st.file_uploader(
                "Camera A synchronized chessboard photos", type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True, key="stereo_images_a",
            )
        with right:
            intrinsics_b_file = st.file_uploader(
                "Camera B intrinsics JSON", type=["json"], key="intrinsics_b"
            )
            images_b_files = st.file_uploader(
                "Camera B synchronized chessboard photos", type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True, key="stereo_images_b",
            )
        stereo_columns, stereo_rows, stereo_square = st.columns(3)
        with stereo_columns:
            stereo_board_columns = st.number_input("Stereo inner corners · columns", min_value=2, max_value=20, value=9)
        with stereo_rows:
            stereo_board_rows = st.number_input("Stereo inner corners · rows", min_value=2, max_value=20, value=6)
        with stereo_square:
            stereo_square_size = st.number_input("Stereo square size", min_value=0.001, value=1.0, step=0.1)
        translation_unit = st.text_input("Translation unit", value="same as square size")
        ready = bool(intrinsics_a_file and intrinsics_b_file and images_a_files and images_b_files)
        if st.button("Calibrate camera pair", type="primary", disabled=not ready):
            if len(images_a_files) != len(images_b_files):
                st.error("Upload the same number of synchronized images for both cameras.")
            else:
                try:
                    calibration_a = load_calibration_json(intrinsics_a_file.getvalue().decode("utf-8"))
                    calibration_b = load_calibration_json(intrinsics_b_file.getvalue().decode("utf-8"))
                    def decode_images(files):
                        decoded = []
                        for uploaded in files:
                            raw = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
                            decoded.append(cv2.imdecode(raw, cv2.IMREAD_COLOR))
                        return decoded
                    decoded_a = decode_images(images_a_files)
                    decoded_b = decode_images(images_b_files)
                    with st.spinner("Estimating relative camera pose from paired chessboards..."):
                        stereo = calibrate_stereo_cameras(
                            decoded_a, decoded_b, calibration_a, calibration_b,
                            board_size=(int(stereo_board_columns), int(stereo_board_rows)),
                            square_size=float(stereo_square_size),
                            translation_unit=translation_unit,
                        )
                    st.success(
                        f"Stereo calibration complete · {stereo.accepted_pairs} usable pairs · "
                        f"{stereo.rejected_pairs} rejected"
                    )
                    x, y = st.columns(2)
                    x.metric("Stereo reprojection RMS", f"{stereo.rms_reprojection_error:.4f}")
                    y.metric("Translation unit", stereo.translation_unit)
                    st.json({
                        "rotation_matrix": stereo.rotation_matrix,
                        "translation_vector": stereo.translation_vector,
                        "essential_matrix": stereo.essential_matrix,
                        "fundamental_matrix": stereo.fundamental_matrix,
                    })
                    st.download_button(
                        "Download stereo extrinsics JSON",
                        stereo.to_json().encode("utf-8"),
                        file_name="blindspot_stereo_extrinsics.json",
                        mime="application/json",
                    )
                except Exception as exc:
                    st.error(f"Stereo calibration failed: {exc}")


    else:
        st.write(
            "Upload calibrated camera intrinsics, stereo extrinsics, and a synchronized "
            "image pair. The app rectifies the pair and estimates disparity/depth with SGBM."
        )
        st.warning(
            "Use the same fixed camera arrangement and resolution used during calibration. "
            "Depth units follow the square-size unit used for calibration. Textureless or "
            "occluded regions may have invalid depth."
        )
        c1, c2 = st.columns(2)
        with c1:
            depth_intrinsics_a = st.file_uploader(
                "Camera A intrinsics JSON", type=["json"], key="depth_intrinsics_a"
            )
            depth_image_a = st.file_uploader(
                "Camera A image", type=["jpg", "jpeg", "png", "bmp"], key="depth_image_a"
            )
        with c2:
            depth_intrinsics_b = st.file_uploader(
                "Camera B intrinsics JSON", type=["json"], key="depth_intrinsics_b"
            )
            depth_image_b = st.file_uploader(
                "Camera B image", type=["jpg", "jpeg", "png", "bmp"], key="depth_image_b"
            )
        stereo_json = st.file_uploader(
            "Stereo extrinsics JSON", type=["json"], key="depth_stereo_json"
        )
        d1, d2 = st.columns(2)
        with d1:
            num_disparities = st.select_slider(
                "Disparity search range", options=list(range(16, 257, 16)), value=128
            )
        with d2:
            block_size = st.select_slider(
                "Matching block size", options=[3, 5, 7, 9, 11, 13, 15], value=5
            )
        ready_depth = bool(
            depth_intrinsics_a and depth_intrinsics_b and depth_image_a
            and depth_image_b and stereo_json
        )
        if st.button("Generate stereo depth map", type="primary", disabled=not ready_depth):
            try:
                calibration_a = load_calibration_json(depth_intrinsics_a.getvalue().decode("utf-8"))
                calibration_b = load_calibration_json(depth_intrinsics_b.getvalue().decode("utf-8"))
                stereo = load_stereo_calibration_json(stereo_json.getvalue().decode("utf-8"))
                raw_a = np.frombuffer(depth_image_a.getvalue(), dtype=np.uint8)
                raw_b = np.frombuffer(depth_image_b.getvalue(), dtype=np.uint8)
                image_a = cv2.imdecode(raw_a, cv2.IMREAD_COLOR)
                image_b = cv2.imdecode(raw_b, cv2.IMREAD_COLOR)
                with st.spinner("Rectifying camera pair and estimating depth..."):
                    result = rectify_and_compute_depth(
                        image_a, image_b, calibration_a, calibration_b, stereo,
                        num_disparities=int(num_disparities), block_size=int(block_size),
                    )
                valid_count = int(result.valid_mask.sum())
                total_count = int(result.valid_mask.size)
                valid_depth = result.depth_map[result.valid_mask]
                if valid_count == 0:
                    st.warning(
                        "No valid depth pixels were estimated. Check camera order, "
                        "calibration, image synchronization, and scene texture."
                    )
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Valid depth pixels", f"{100*valid_count/total_count:.1f}%")
                    m2.metric("Median depth", f"{np.median(valid_depth):.2f} {result.translation_unit}")
                    m3.metric("Depth samples", f"{valid_count:,}")
                    left, right = st.columns(2)
                    with left:
                        st.image(cv2.cvtColor(result.rectified_a, cv2.COLOR_BGR2RGB), caption="Rectified camera A")
                    with right:
                        st.image(cv2.cvtColor(result.rectified_b, cv2.COLOR_BGR2RGB), caption="Rectified camera B")
                    disparity_figure = px.imshow(
                        np.where(result.valid_mask, result.disparity_pixels, np.nan),
                        color_continuous_scale="Turbo",
                        labels={"x": "Pixel", "y": "Pixel row", "color": "Disparity (px)"},
                        title="Stereo disparity map",
                    )
                    disparity_figure.update_layout(height=420)
                    st.plotly_chart(disparity_figure, use_container_width=True)
                    upper_depth = float(np.nanpercentile(valid_depth, 95))
                    depth_figure = px.imshow(
                        np.where(result.valid_mask, result.depth_map, np.nan),
                        zmin=float(np.nanpercentile(valid_depth, 5)),
                        zmax=max(upper_depth, float(np.nanpercentile(valid_depth, 5)) + 1e-6),
                        color_continuous_scale="Viridis",
                        labels={"x": "Pixel", "y": "Pixel row", "color": f"Depth ({result.translation_unit})"},
                        title="Estimated depth map",
                    )
                    depth_figure.update_layout(height=420)
                    st.plotly_chart(depth_figure, use_container_width=True)
                    st.download_button(
                        "Download depth map (CSV)",
                        pd.DataFrame(result.depth_map).to_csv(index=False).encode("utf-8"),
                        file_name="blindspot_stereo_depth.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                st.error(f"Stereo depth estimation failed: {exc}")



st.divider()
st.caption(
    "Research prototype only · No drone control · No calibrated probability or safety guarantee."
)
