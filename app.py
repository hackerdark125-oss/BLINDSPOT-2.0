from __future__ import annotations

import json
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
from blindspot.people import process_video as process_people_video
from blindspot.ai_detection import YOLODetector, process_video as process_ai_video
from blindspot.synchronization import synchronize_timestamps
from blindspot.stereo import (
    calibrate_stereo_cameras, load_stereo_calibration_json,
    rectify_and_compute_depth,
)
from blindspot.coverage import load_camera_pose_json, estimate_stereo_planar_coverage
from blindspot.reconstruction import (
    depth_to_point_cloud, write_ply, estimate_stereo_visual_motion,
    compose_camera_to_world_pose, fuse_point_clouds, exclude_detection_regions,
    select_moving_detection_regions,
)
from blindspot.uncertainty import (
    evaluate_calibration, fit_isotonic_calibrator, apply_isotonic_calibrator,
)
from blindspot.pose_graph import optimize_pose_graph_translations
from blindspot.loop_closure import detect_loop_closure_candidates
from blindspot.multicam import fuse_multicamera_point_clouds
from blindspot.ply import read_ascii_ply


@st.cache_resource(show_spinner=False)
def get_yolo_detector(weights: str):
    """Cache loaded YOLO models between Streamlit reruns."""
    return YOLODetector(weights)

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

mode = st.radio("Analysis source", ["Recorded videos", "Live webcam / RTSP", "Person detection & tracking", "AI object detection & tracking", "Camera calibration"], horizontal=True)


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
        "Camera motion compensated": a.camera_motion_compensated,
        "Scene change detected": a.scene_changed,
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
    if summary["Scene change detected"].any():
        st.warning(f"Scene change reset applied to {int(summary['Scene change detected'].sum())} sampled frame(s).")

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
    cell_rows = [
        {
            "sample_index": sample_index,
            "time_seconds": analysis.timestamp_seconds,
            "grid_row": row,
            "grid_column": column,
            "predicted_score": float(analysis.confidence_map[row, column]),
            "reference_score": "",
        }
        for sample_index, analysis in enumerate(analyses)
        for row in range(analysis.confidence_map.shape[0])
        for column in range(analysis.confidence_map.shape[1])
    ]
    st.download_button(
        "Download per-cell quality labels template (CSV)",
        pd.DataFrame(cell_rows).to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(name).stem}_quality_labels_template.csv",
        mime="text/csv", key=f"quality_labels_{name}_{'live' if live else 'video'}",
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
        camera_streams = {}
        camera_offsets = {}
        for i, upload in enumerate(uploads[:4]):
            camera_label = f"Camera {i+1} · {upload.name}"
            st.subheader(camera_label)
            camera_offsets[camera_label] = st.number_input(
                "Timeline offset (seconds; added to sample timestamps)",
                min_value=-3600.0, max_value=3600.0, value=0.0, step=0.1,
                key=f"camera_time_offset_{i}",
            )
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
                camera_streams[camera_label] = [item.timestamp_seconds for item in results]
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
            if len(camera_streams) >= 2:
                st.divider()
                st.subheader("Multi-camera timestamp alignment")
                st.caption(
                    "Set each camera's timeline offset if recording start times differ. "
                    "This aligns sampled timestamps only; it does not prove hardware synchronization "
                    "or that frames depict the exact same instant."
                )
                tolerance = st.slider(
                    "Maximum allowed inter-camera skew (seconds)",
                    min_value=0.01, max_value=1.0, value=0.25, step=0.01,
                    key="camera_sync_tolerance",
                )
                aligned = synchronize_timestamps(
                    camera_streams, camera_offsets, tolerance_seconds=float(tolerance)
                )
                if aligned:
                    aligned_rows = []
                    for match in aligned:
                        row = {"Reference time (s)": match["reference_time_seconds"]}
                        for camera_name, sample in match["samples"].items():
                            row[f"{camera_name} · sample"] = sample["sample_index"]
                            row[f"{camera_name} · source time (s)"] = sample["source_time_seconds"]
                            row[f"{camera_name} · skew (s)"] = sample["skew_seconds"]
                        aligned_rows.append(row)
                    aligned_df = pd.DataFrame(aligned_rows)
                    st.metric("Aligned multi-camera sample groups", len(aligned_df))
                    st.dataframe(aligned_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download synchronized sample index (CSV)",
                        aligned_df.to_csv(index=False).encode("utf-8"),
                        file_name="blindspot_synchronized_camera_samples.csv",
                        mime="text/csv", key="download_sync_index",
                    )
                else:
                    st.warning(
                        "No sample groups fell within the selected skew tolerance. "
                        "Adjust the per-camera offsets or increase the tolerance."
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



elif mode == "Person detection & tracking":
    st.subheader("Pedestrian detection and tracking")
    st.info(
        "Uses OpenCV's built-in HOG pedestrian detector. It detects upright people, "
        "not arbitrary object classes. Track IDs are temporary and can change after "
        "long occlusions or abrupt movement."
    )
    tracking_upload = st.file_uploader(
        "Upload a video for pedestrian tracking",
        type=["mp4", "avi", "mov", "mkv"], key="people_tracking_video",
    )
    track_col1, track_col2 = st.columns(2)
    with track_col1:
        frame_stride = st.slider("Process every Nth frame", 1, 30, 5)
    with track_col2:
        detector_scale = st.select_slider(
            "Detection speed / sensitivity", options=[1.01, 1.03, 1.05, 1.08, 1.1],
            value=1.05, format_func=lambda value: f"{value:.2f}",
        )
    if tracking_upload and st.button("Detect and track people", type="primary"):
        suffix = Path(tracking_upload.name).suffix or ".mp4"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(tracking_upload.getvalue())
                tracking_path = tmp.name
            try:
                with st.spinner("Detecting pedestrians and associating track IDs..."):
                    tracking_results = process_people_video(
                        tracking_path, sample_every_frames=frame_stride,
                        max_samples=max_frames, detector_scale=float(detector_scale),
                    )
            finally:
                Path(tracking_path).unlink(missing_ok=True)
            records, visible_ids = [], set()
            for result in tracking_results:
                st.image(
                    cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                    caption=f"Time {result.timestamp_seconds:.2f}s · {len(result.detections)} person(s)",
                    use_container_width=True,
                )
                for item in result.detections:
                    visible_ids.add(item.track_id)
                    box = item.detection
                    records.append({
                        "Time (s)": result.timestamp_seconds, "Track ID": item.track_id,
                        "X": box.x, "Y": box.y, "Width": box.width, "Height": box.height,
                        "Detector score": box.score,
                    })
            tracking_df = pd.DataFrame(records, columns=[
                "Time (s)", "Track ID", "X", "Y", "Width", "Height", "Detector score"
            ])
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Frames processed", len(tracking_results))
            metric_b.metric("Unique track IDs", len(visible_ids))
            metric_c.metric("Person observations", len(records))
            st.dataframe(tracking_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download tracking results (CSV)",
                tracking_df.to_csv(index=False).encode("utf-8"),
                file_name="blindspot_people_tracking.csv", mime="text/csv",
            )
            if not records:
                st.warning("No pedestrians were detected in the sampled frames.")
        except Exception as exc:
            st.error(f"Person detection failed: {exc}")

elif mode == "AI object detection & tracking":
    st.subheader("AI object detection and multi-object tracking")
    st.info(
        "Uses YOLO11 for object detection and ByteTrack for persistent IDs. "
        "The optional model package and pretrained weights are downloaded separately. "
        "Tracking IDs can switch after long occlusions or scene cuts."
    )
    ai_upload = st.file_uploader(
        "Upload a video for AI object tracking",
        type=["mp4", "avi", "mov", "mkv"], key="ai_tracking_video",
    )
    ai_col1, ai_col2, ai_col3 = st.columns(3)
    with ai_col1:
        ai_weights = st.selectbox("Model size", ["yolo11n.pt", "yolo11s.pt"], index=0)
        ai_stride = st.slider("Process every Nth frame", 1, 20, 3, key="ai_stride")
    with ai_col2:
        ai_confidence = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
        ai_image_size = st.select_slider("Inference image size", options=[320, 416, 512, 640, 768, 960], value=640)
    with ai_col3:
        class_options = {
            "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
            "airplane": 4, "bus": 5, "train": 6, "truck": 7, "boat": 8,
            "traffic light": 9, "fire hydrant": 10, "stop sign": 11,
            "parking meter": 12, "bench": 13, "bird": 14, "cat": 15,
            "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
            "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23,
            "backpack": 24, "umbrella": 25, "handbag": 26, "tie": 27,
            "suitcase": 28, "frisbee": 29, "skis": 30, "snowboard": 31,
            "sports ball": 32, "kite": 33, "baseball bat": 34,
            "baseball glove": 35, "skateboard": 36, "surfboard": 37,
            "tennis racket": 38, "bottle": 39, "wine glass": 40, "cup": 41,
            "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
            "banana": 46, "apple": 47, "sandwich": 48, "orange": 49,
            "broccoli": 50, "carrot": 51, "hot dog": 52, "pizza": 53,
            "donut": 54, "cake": 55, "chair": 56, "couch": 57,
            "potted plant": 58, "bed": 59, "dining table": 60, "toilet": 61,
            "tv": 62, "laptop": 63, "mouse": 64, "remote": 65,
            "keyboard": 66, "cell phone": 67, "microwave": 68, "oven": 69,
            "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
            "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77,
            "hair drier": 78, "toothbrush": 79,
        }
        selected_labels = st.multiselect(
            "Object classes",
            list(class_options),
            default=["person", "car", "motorcycle", "bus", "truck"],
        )
    if ai_upload and st.button("Run AI detection and tracking", type="primary"):
        if not selected_labels:
            st.warning("Select at least one object class.")
        else:
            suffix = Path(ai_upload.name).suffix or ".mp4"
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(ai_upload.getvalue())
                    ai_video_path = tmp.name
                try:
                    with st.spinner("Loading YOLO weights and tracking objects with ByteTrack..."):
                        detector = get_yolo_detector(ai_weights)
                        ai_results = process_ai_video(
                            ai_video_path,
                            sample_every_frames=ai_stride,
                            max_samples=max_frames,
                            weights=ai_weights,
                            classes=[class_options[label] for label in selected_labels],
                            confidence=ai_confidence,
                            image_size=ai_image_size,
                            detector=detector,
                        )
                finally:
                    Path(ai_video_path).unlink(missing_ok=True)

                observations = []
                track_ids = set()
                class_counts = {}
                for result in ai_results:
                    st.image(
                        cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Time {result.timestamp_seconds:.2f}s · "
                                f"{len(result.detections)} object(s)",
                        use_container_width=True,
                    )
                    for item in result.detections:
                        if item.track_id is not None:
                            track_ids.add(item.track_id)
                        class_counts[item.class_name] = class_counts.get(item.class_name, 0) + 1
                        observations.append({
                            "Frame index": result.frame_index,
                            "Time (s)": result.timestamp_seconds,
                            "YOLO + ByteTrack (ms)": result.inference_time_ms,
                            "Track ID": item.track_id,
                            "Class": item.class_name,
                            "Class ID": item.class_id,
                            "Confidence": item.confidence,
                            "X": item.x, "Y": item.y,
                            "Width": item.width, "Height": item.height,
                        })
                ai_df = pd.DataFrame(observations, columns=[
                    "Frame index", "Time (s)", "YOLO + ByteTrack (ms)", "Track ID", "Class", "Class ID", "Confidence",
                    "X", "Y", "Width", "Height",
                ])
                mean_inference_ms = float(np.mean([r.inference_time_ms for r in ai_results]))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Frames processed", len(ai_results))
                m2.metric("Unique tracked IDs", len(track_ids))
                m3.metric("Object observations", len(observations))
                m4.metric("Mean YOLO + tracker", f"{mean_inference_ms:.1f} ms")
                st.caption(
                    f"YOLO + ByteTrack processing: {1000.0 / mean_inference_ms:.1f} FPS · "
                    "excludes video decoding, rendering, and CSV export"
                )
                if class_counts:
                    st.write("**Detections by class**")
                    st.bar_chart(pd.Series(class_counts, name="Observations"))
                st.dataframe(ai_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download AI tracking results (CSV)",
                    ai_df.to_csv(index=False).encode("utf-8"),
                    file_name="blindspot_ai_object_tracking.csv",
                    mime="text/csv",
                )
                frame_metrics_df = pd.DataFrame([{
                    "Frame index": result.frame_index,
                    "Time (s)": result.timestamp_seconds,
                    "YOLO + ByteTrack (ms)": result.inference_time_ms,
                    "Detections": len(result.detections),
                } for result in ai_results])
                st.download_button(
                    "Download per-frame performance (CSV)",
                    frame_metrics_df.to_csv(index=False).encode("utf-8"),
                    file_name="blindspot_ai_frame_performance.csv",
                    mime="text/csv",
                )
                if not observations:
                    st.warning("No selected objects were detected in the sampled frames.")
            except ImportError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"AI object tracking failed: {exc}")

else:
    st.subheader("Camera calibration")
    calibration_mode = st.radio(
        "Calibration workflow",
        ["Single-camera intrinsics", "Two-camera stereo extrinsics", "Stereo depth preview", "Stereo visual odometry & 3D fusion", "Multi-camera PLY fusion", "Planar coverage preview", "Uncertainty calibration"],
        horizontal=True,
    )
    if calibration_mode == "Uncertainty calibration":
        st.write(
            "Fit a monotone isotonic confidence mapping on one labelled training CSV, "
            "then evaluate raw and calibrated scores on a separate held-out CSV."
        )
        st.warning(
            "Do not use the held-out evaluation file to fit the calibrator. Confidence "
            "values must be probabilities between 0 and 1, and outcomes must be independently "
            "labelled as 0 or 1. The existing image-quality score (0–100) is not a probability."
        )
        train_file = st.file_uploader(
            "Calibration training labels CSV", type=["csv"], key="uncertainty_train_csv"
        )
        eval_file = st.file_uploader(
            "Held-out evaluation labels CSV", type=["csv"], key="uncertainty_eval_csv"
        )
        bin_count = st.slider("Reliability bins", min_value=5, max_value=20, value=10, key="uncertainty_bins")
        if st.button("Fit calibrator and evaluate held-out data", type="primary", disabled=not (train_file and eval_file)):
            try:
                train_frame = pd.read_csv(train_file)
                eval_frame = pd.read_csv(eval_file)
                required_columns = {"predicted_confidence", "outcome"}
                if not required_columns.issubset(train_frame.columns) or not required_columns.issubset(eval_frame.columns):
                    raise ValueError("both CSV files must contain predicted_confidence and outcome columns")
                train_samples = train_frame[["predicted_confidence", "outcome"]].to_dict(orient="records")
                eval_samples = eval_frame[["predicted_confidence", "outcome"]].to_dict(orient="records")
                calibrator = fit_isotonic_calibrator(train_samples)
                raw_metrics = evaluate_calibration(eval_samples, bins=int(bin_count))
                calibrated_scores = apply_isotonic_calibrator(
                    [sample["predicted_confidence"] for sample in eval_samples], calibrator
                )
                calibrated_samples = [
                    {"predicted_confidence": score, "outcome": sample["outcome"]}
                    for score, sample in zip(calibrated_scores, eval_samples)
                ]
                calibrated_metrics = evaluate_calibration(calibrated_samples, bins=int(bin_count))
                st.success(
                    f"Calibrator fitted on {len(train_samples):,} training labels and evaluated "
                    f"on {len(eval_samples):,} separate held-out labels."
                )
                metric_rows = [
                    {"Metric": "Brier score", "Before": raw_metrics["brier_score"], "After": calibrated_metrics["brier_score"]},
                    {"Metric": "Log loss", "Before": raw_metrics["log_loss"], "After": calibrated_metrics["log_loss"]},
                    {"Metric": "Expected calibration error", "Before": raw_metrics["expected_calibration_error"], "After": calibrated_metrics["expected_calibration_error"]},
                ]
                st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
                raw_bins = pd.DataFrame(raw_metrics["reliability_bins"])
                calibrated_bins = pd.DataFrame(calibrated_metrics["reliability_bins"])
                if not raw_bins.empty and not calibrated_bins.empty:
                    raw_bins["Series"] = "Before calibration"
                    calibrated_bins["Series"] = "After calibration"
                    reliability = pd.concat([raw_bins, calibrated_bins], ignore_index=True)
                    figure = px.line(
                        reliability, x="mean_confidence", y="observed_frequency",
                        color="Series", markers=True,
                        title="Held-out reliability diagram",
                        labels={"mean_confidence": "Mean predicted confidence", "observed_frequency": "Observed positive frequency"},
                    )
                    figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash"})
                    figure.update_layout(height=420, xaxis_range=[0, 1], yaxis_range=[0, 1])
                    st.plotly_chart(figure, use_container_width=True)
                st.write("**Held-out reliability bins · before calibration**")
                st.dataframe(raw_bins.drop(columns=["Series"], errors="ignore"), use_container_width=True, hide_index=True)
                st.write("**Held-out reliability bins · after calibration**")
                st.dataframe(calibrated_bins.drop(columns=["Series"], errors="ignore"), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download fitted isotonic calibrator (JSON)",
                    json.dumps(calibrator, indent=2).encode("utf-8"),
                    file_name="blindspot_isotonic_calibrator.json", mime="application/json",
                )
                output = eval_frame.copy()
                output["calibrated_confidence"] = calibrated_scores
                st.download_button(
                    "Download held-out calibrated scores (CSV)",
                    output.to_csv(index=False).encode("utf-8"),
                    file_name="blindspot_heldout_calibrated_scores.csv", mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Uncertainty calibration failed: {exc}")
    elif calibration_mode == "Single-camera intrinsics":
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


    elif calibration_mode == "Stereo depth preview":
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
                    cloud_stride = st.select_slider(
                        "3D point-cloud density (pixel stride)",
                        options=[1, 2, 3, 4, 6, 8],
                        value=2,
                        key="point_cloud_stride",
                    )
                    try:
                        point_cloud = depth_to_point_cloud(
                            result.depth_map,
                            result.rectified_camera_matrix if result.rectified_camera_matrix is not None else calibration_a.camera_matrix,
                            color_image=result.rectified_a,
                            valid_mask=result.valid_mask,
                            stride=int(cloud_stride),
                        )
                        with tempfile.TemporaryDirectory() as temp_dir:
                            ply_path = write_ply(Path(temp_dir) / "blindspot_scene.ply", point_cloud)
                            ply_bytes = ply_path.read_bytes()
                        st.metric("3D points generated", f"{len(point_cloud):,}")
                        st.download_button(
                            "Download reconstructed 3D point cloud (PLY)",
                            ply_bytes,
                            file_name="blindspot_scene.ply",
                            mime="application/octet-stream",
                        )
                        st.caption(
                            "Point coordinates use the stereo calibration distance unit. "
                            "This is a stereo depth point cloud, not globally aligned SLAM."
                        )
                    except Exception as cloud_exc:
                        st.warning(f"3D point-cloud export unavailable: {cloud_exc}")
            except Exception as exc:
                st.error(f"Stereo depth estimation failed: {exc}")

    elif calibration_mode == "Stereo visual odometry & 3D fusion":
        st.write(
            "Upload synchronized stereo videos and matching camera calibration files. "
            "The pipeline estimates stereo depth, tracks 3D features with PnP-RANSAC, "
            "chains camera poses, and fuses sampled point clouds."
        )
        st.warning(
            "Use fixed, synchronized cameras with the same resolution and calibration. "
            "This is local stereo visual odometry and drift can accumulate. An optional "
            "manual translation-only correction accepts an independently verified loop-closure "
            "constraint; automatic loop-closure detection and full 6DoF SLAM are not implemented."
        )
        left, right = st.columns(2)
        with left:
            vo_video_a = st.file_uploader("Camera A video", type=["mp4", "avi", "mov", "mkv"], key="vo_video_a")
            vo_intrinsics_a = st.file_uploader("Camera A intrinsics JSON", type=["json"], key="vo_intrinsics_a")
        with right:
            vo_video_b = st.file_uploader("Camera B video", type=["mp4", "avi", "mov", "mkv"], key="vo_video_b")
            vo_intrinsics_b = st.file_uploader("Camera B intrinsics JSON", type=["json"], key="vo_intrinsics_b")
        vo_stereo_json = st.file_uploader("Stereo extrinsics JSON", type=["json"], key="vo_stereo_json")
        v1, v2, v3 = st.columns(3)
        with v1:
            vo_interval = st.slider("VO sampling interval (seconds)", 0.25, 3.0, 1.0, 0.25, key="vo_interval")
        with v2:
            vo_limit = st.slider("Maximum stereo frame pairs", 2, 30, 10, 1, key="vo_limit")
        with v3:
            vo_stride = st.select_slider("Point-cloud pixel stride", options=[2, 3, 4, 6, 8], value=4, key="vo_stride")
        vo_voxel = st.number_input("Voxel averaging size (stereo distance units)", min_value=0.001, value=1.0, step=0.1, key="vo_voxel")
        vo_remove_dynamic = st.checkbox(
            "Filter tracked moving objects before 3D fusion (optional YOLO + ByteTrack)",
            value=False, key="vo_remove_dynamic",
            help="Requires requirements-ai.txt. Only objects with persistent ByteTrack IDs that move beyond the pixel threshold are masked; new/untracked objects are not classified as moving."
        )
        vo_ai_weights = st.selectbox(
            "Dynamic-object detector weights", ["yolo11n.pt", "yolo11s.pt"],
            disabled=not vo_remove_dynamic, key="vo_dynamic_weights",
        )
        vo_motion_threshold = st.slider(
            "Minimum track movement to mask (pixels)", 1.0, 20.0, 4.0, 1.0,
            disabled=not vo_remove_dynamic, key="vo_motion_threshold",
        )
        ready_vo = bool(vo_video_a and vo_video_b and vo_intrinsics_a and vo_intrinsics_b and vo_stereo_json)
        if st.button("Run stereo visual odometry & fuse 3D scene", type="primary", disabled=not ready_vo):
            capture_a = capture_b = None
            temp_paths = []
            try:
                calibration_a = load_calibration_json(vo_intrinsics_a.getvalue().decode("utf-8"))
                calibration_b = load_calibration_json(vo_intrinsics_b.getvalue().decode("utf-8"))
                stereo = load_stereo_calibration_json(vo_stereo_json.getvalue().decode("utf-8"))
                dynamic_detector = get_yolo_detector(vo_ai_weights) if vo_remove_dynamic else None
                if dynamic_detector is not None:
                    dynamic_detector.reset_tracker()
                for upload in (vo_video_a, vo_video_b):
                    suffix = Path(upload.name).suffix or ".mp4"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(upload.getvalue())
                        temp_paths.append(temp_file.name)
                capture_a = cv2.VideoCapture(temp_paths[0])
                capture_b = cv2.VideoCapture(temp_paths[1])
                if not capture_a.isOpened() or not capture_b.isOpened():
                    raise ValueError("could not open both uploaded stereo videos")
                fps_a = capture_a.get(cv2.CAP_PROP_FPS)
                fps_b = capture_b.get(cv2.CAP_PROP_FPS)
                if not np.isfinite(fps_a) or not np.isfinite(fps_b) or fps_a <= 0 or fps_b <= 0:
                    raise ValueError("both videos must report a valid frame rate")
                duration_a = capture_a.get(cv2.CAP_PROP_FRAME_COUNT) / fps_a
                duration_b = capture_b.get(cv2.CAP_PROP_FRAME_COUNT) / fps_b
                if abs(duration_a - duration_b) > max(vo_interval, 0.5):
                    st.warning("Video durations differ. Sampling will stop when either stream ends.")
                local_clouds, world_poses, trajectory, loop_keyframes = [], [], [], []
                previous_frame = previous_depth = previous_valid_mask = None
                world_rotation, world_translation = np.eye(3), np.zeros(3)
                processed, skipped, masked_detections = 0, 0, 0
                previous_track_centers = {}
                progress = st.progress(0.0)
                status = st.empty()
                for sample_index in range(int(vo_limit)):
                    timestamp = sample_index * float(vo_interval)
                    capture_a.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    capture_b.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
                    ok_a, frame_a = capture_a.read()
                    ok_b, frame_b = capture_b.read()
                    if not ok_a or not ok_b:
                        break
                    status.write(f"Processing stereo pair {sample_index + 1}/{vo_limit} · {timestamp:.2f}s")
                    try:
                        depth_result = rectify_and_compute_depth(
                            frame_a, frame_b, calibration_a, calibration_b, stereo,
                            num_disparities=128, block_size=5,
                        )
                        if int(depth_result.valid_mask.sum()) < 100:
                            skipped += 1
                            continue
                        frame_valid_mask = depth_result.valid_mask.copy()
                        if dynamic_detector is not None:
                            tracked_boxes = dynamic_detector.track_frame(
                                depth_result.rectified_a, confidence=0.25, image_size=640
                            )
                            moving_boxes, previous_track_centers = select_moving_detection_regions(
                                tracked_boxes, previous_track_centers,
                                min_displacement_pixels=float(vo_motion_threshold),
                            )
                            masked_detections += len(moving_boxes)
                            frame_valid_mask = exclude_detection_regions(
                                frame_valid_mask, moving_boxes, padding=8,
                            )
                        if int(frame_valid_mask.sum()) < 100:
                            skipped += 1
                            continue
                        rectified_k = (
                            depth_result.rectified_camera_matrix
                            if depth_result.rectified_camera_matrix is not None
                            else calibration_a.camera_matrix
                        )
                        if previous_frame is None:
                            local_cloud = depth_to_point_cloud(
                                depth_result.depth_map, rectified_k,
                                color_image=depth_result.rectified_a,
                                valid_mask=frame_valid_mask, stride=int(vo_stride),
                            )
                            if len(local_cloud) == 0:
                                skipped += 1
                                continue
                            local_clouds.append(local_cloud)
                            loop_keyframes.append(depth_result.rectified_a.copy())
                            world_poses.append((world_rotation.copy(), world_translation.copy()))
                            trajectory.append({
                                "sample_index": sample_index, "time_seconds": timestamp,
                                "x": 0.0, "y": 0.0, "z": 0.0,
                                "pnp_inliers": 0, "pnp_inlier_ratio": 0.0,
                            })
                            previous_frame = depth_result.rectified_a
                            previous_depth = depth_result.depth_map
                            previous_valid_mask = frame_valid_mask
                            processed += 1
                        else:
                            motion = estimate_stereo_visual_motion(
                                previous_frame, depth_result.rectified_a, previous_depth,
                                rectified_k, min_inliers=8,
                                previous_valid_mask=previous_valid_mask,
                            )
                            candidate_rotation, candidate_translation = compose_camera_to_world_pose(
                                world_rotation, world_translation,
                                motion.rotation_matrix, motion.translation_vector,
                            )
                            local_cloud = depth_to_point_cloud(
                                depth_result.depth_map, rectified_k,
                                color_image=depth_result.rectified_a,
                                valid_mask=frame_valid_mask, stride=int(vo_stride),
                            )
                            if len(local_cloud) == 0:
                                skipped += 1
                                continue
                            local_clouds.append(local_cloud)
                            loop_keyframes.append(depth_result.rectified_a.copy())
                            world_poses.append((candidate_rotation.copy(), candidate_translation.copy()))
                            world_rotation, world_translation = candidate_rotation, candidate_translation
                            trajectory.append({
                                "sample_index": sample_index, "time_seconds": timestamp,
                                "x": float(world_translation[0]), "y": float(world_translation[1]),
                                "z": float(world_translation[2]),
                                "pnp_inliers": motion.inlier_count,
                                "pnp_inlier_ratio": motion.inlier_ratio,
                            })
                            previous_frame = depth_result.rectified_a
                            previous_depth = depth_result.depth_map
                            previous_valid_mask = frame_valid_mask
                            processed += 1
                    except Exception as frame_exc:
                        skipped += 1
                        st.warning(f"Pair {sample_index + 1} skipped: {frame_exc}")
                    progress.progress((sample_index + 1) / int(vo_limit))
                status.empty()
                progress.empty()
                if not local_clouds:
                    st.error("No valid stereo frames could be reconstructed. Check synchronisation, texture, and calibration.")
                else:
                    fused = fuse_point_clouds(local_clouds, world_poses, voxel_size=float(vo_voxel))
                    with tempfile.TemporaryDirectory() as output_dir:
                        ply_path = write_ply(Path(output_dir) / "blindspot_fused_scene.ply", fused)
                        ply_bytes = ply_path.read_bytes()
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Frames reconstructed", processed)
                    m2.metric("Frames skipped", skipped)
                    m3.metric("Fused 3D points", f"{len(fused):,}")
                    if dynamic_detector is not None:
                        st.metric("Detected object regions masked", masked_detections)
                        st.caption("Only tracked objects moving beyond the selected image-space threshold are masked. This assumes a fixed camera rig; missed detections and camera movement can still cause errors.")
                    st.dataframe(pd.DataFrame(trajectory), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download fused multi-frame 3D scene (PLY)", ply_bytes,
                        file_name="blindspot_fused_scene.ply", mime="application/octet-stream",
                    )
                    st.download_button(
                        "Download estimated camera trajectory (CSV)",
                        pd.DataFrame(trajectory).to_csv(index=False).encode("utf-8"),
                        file_name="blindspot_camera_trajectory.csv", mime="text/csv",
                    )
                    if len(loop_keyframes) >= 3:
                        with st.expander("Loop-closure candidate suggestions (visual retrieval)"):
                            st.caption(
                                "BLINDSPOT compares non-adjacent sampled views using ORB feature "
                                "matching and RANSAC geometric verification. These are candidate "
                                "revisits only—not verified metric poses. Review them independently "
                                "before entering any translation constraint below."
                            )
                            try:
                                candidates = detect_loop_closure_candidates(
                                    loop_keyframes,
                                    min_frame_gap=max(2, min(5, len(loop_keyframes) // 3)),
                                    min_matches=10,
                                    min_inliers=8,
                                    min_inlier_ratio=0.35,
                                )
                                if candidates:
                                    candidate_rows = [{
                                        "Earlier trajectory row": item.earlier_index,
                                        "Later trajectory row": item.later_index,
                                        "Feature matches": item.match_count,
                                        "Geometric inliers": item.inlier_count,
                                        "Inlier ratio": round(item.inlier_ratio, 3),
                                        "Verification": item.verification_model,
                                        "Retrieval score": round(item.score, 3),
                                    } for item in candidates]
                                    st.dataframe(
                                        pd.DataFrame(candidate_rows),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                else:
                                    st.info(
                                        "No strong revisit candidates found in these sampled frames. "
                                        "This does not prove that the route contains no loop closure."
                                    )
                            except Exception as loop_exc:
                                st.warning(f"Loop-closure candidate search unavailable: {loop_exc}")
                    if len(trajectory) >= 2:
                        with st.expander("Optional · correct drift with a verified loop closure"):
                            st.warning(
                                "Only enter a loop closure measured independently (for example, a verified "
                                "revisit displacement). The software does not discover loop closures. A wrong "
                                "constraint can make the reconstruction less accurate."
                            )
                            st.caption(
                                "Indices refer to rows in the reconstructed trajectory table, not original video frame numbers. "
                                f"Displacements use {stereo.translation_unit} in the shared world coordinate frame."
                            )
                            closure_start, closure_end = st.columns(2)
                            with closure_start:
                                start_index = st.selectbox(
                                    "Trajectory start row", options=list(range(len(trajectory))),
                                    index=0, key="pose_graph_start_index",
                                )
                            with closure_end:
                                end_index = st.selectbox(
                                    "Trajectory end row", options=list(range(len(trajectory))),
                                    index=len(trajectory) - 1, key="pose_graph_end_index",
                                )
                            delta_cols = st.columns(4)
                            with delta_cols[0]:
                                measured_dx = st.number_input("Measured ΔX", value=0.0, key="pose_graph_dx")
                            with delta_cols[1]:
                                measured_dy = st.number_input("Measured ΔY", value=0.0, key="pose_graph_dy")
                            with delta_cols[2]:
                                measured_dz = st.number_input("Measured ΔZ", value=0.0, key="pose_graph_dz")
                            with delta_cols[3]:
                                closure_weight = st.number_input(
                                    "Constraint weight", min_value=0.01, value=100.0, step=1.0,
                                    key="pose_graph_weight",
                                )
                            if st.button(
                                "Apply verified loop-closure correction",
                                key="apply_pose_graph_correction",
                                disabled=start_index >= end_index,
                            ):
                                try:
                                    raw_positions = [[row["x"], row["y"], row["z"]] for row in trajectory]
                                    corrected_positions = optimize_pose_graph_translations(
                                        raw_positions,
                                        [(int(start_index), int(end_index),
                                          [float(measured_dx), float(measured_dy), float(measured_dz)],
                                          float(closure_weight))],
                                    )
                                    corrected_poses = [
                                        (rotation, np.asarray(corrected_positions[index], dtype=float))
                                        for index, (rotation, _) in enumerate(world_poses)
                                    ]
                                    corrected_fused = fuse_point_clouds(
                                        local_clouds, corrected_poses, voxel_size=float(vo_voxel)
                                    )
                                    corrected_trajectory = pd.DataFrame(trajectory).copy()
                                    corrected_array = np.asarray(corrected_positions, dtype=float)
                                    corrected_trajectory[["x", "y", "z"]] = corrected_array
                                    with tempfile.TemporaryDirectory() as corrected_dir:
                                        corrected_path = write_ply(
                                            Path(corrected_dir) / "blindspot_drift_corrected_scene.ply",
                                            corrected_fused,
                                        )
                                        corrected_ply_bytes = corrected_path.read_bytes()
                                    st.success(
                                        "Applied the supplied loop-closure constraint. Compare this result "
                                        "with the uncorrected trajectory and measured reference before relying on it."
                                    )
                                    st.dataframe(corrected_trajectory, use_container_width=True, hide_index=True)
                                    st.download_button(
                                        "Download drift-corrected 3D scene (PLY)",
                                        corrected_ply_bytes,
                                        file_name="blindspot_drift_corrected_scene.ply",
                                        mime="application/octet-stream",
                                        key="download_drift_corrected_scene",
                                    )
                                    st.download_button(
                                        "Download drift-corrected trajectory (CSV)",
                                        corrected_trajectory.to_csv(index=False).encode("utf-8"),
                                        file_name="blindspot_drift_corrected_trajectory.csv",
                                        mime="text/csv",
                                        key="download_drift_corrected_trajectory",
                                    )
                                except Exception as correction_exc:
                                    st.error(f"Loop-closure correction failed: {correction_exc}")
                    st.caption(
                        f"All translations and 3D coordinates use {stereo.translation_unit}. "
                        "The raw trajectory is locally chained and may drift. Optional correction "
                        "requires an independently verified loop-closure measurement."
                    )
            except Exception as exc:
                st.error(f"Stereo visual odometry failed: {exc}")
            finally:
                if capture_a is not None:
                    capture_a.release()
                if capture_b is not None:
                    capture_b.release()
                for temp_path in temp_paths:
                    Path(temp_path).unlink(missing_ok=True)

    elif calibration_mode == "Multi-camera PLY fusion":
        st.write(
            "Fuse exported camera-local XYZRGB point clouds into one shared 3D world frame "
            "using a verified camera-to-world pose for each camera."
        )
        st.warning(
            "Upload XYZRGB PLY files (ASCII or binary) and a pose JSON with matching camera IDs. "
            "All poses must use one verified world coordinate frame and the same distance unit. "
            "This workflow does not estimate camera extrinsics or synchronize recordings."
        )
        example_pose = {
            "distance_unit": "meters",
            "cameras": {
                "camera_1": {
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation_vector": [0, 0, 0],
                },
                "camera_2": {
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation_vector": [0.25, 0, 0],
                },
            },
        }
        st.download_button(
            "Download example multi-camera pose JSON",
            json.dumps(example_pose, indent=2).encode("utf-8"),
            file_name="blindspot_multicamera_poses_example.json",
            mime="application/json",
        )
        multi_cloud_files = st.file_uploader(
            "Camera point clouds (name files camera_1.ply, camera_2.ply, etc.)",
            type=["ply"], accept_multiple_files=True, key="multicam_ply_uploads",
        )
        multi_pose_file = st.file_uploader(
            "Shared camera-to-world poses JSON", type=["json"], key="multicam_pose_upload"
        )
        multi_voxel_size = st.number_input(
            "World-frame voxel size (same distance unit as poses)",
            min_value=0.0001, value=0.02, step=0.01, format="%.4f",
            key="multicam_voxel_size",
        )
        if st.button(
            "Fuse multi-camera 3D point clouds",
            type="primary",
            disabled=not (multi_cloud_files and multi_pose_file),
            key="run_multicam_fusion",
        ):
            try:
                pose_document = json.loads(multi_pose_file.getvalue().decode("utf-8"))
                camera_pose_document = pose_document.get("cameras")
                if not isinstance(camera_pose_document, dict) or not camera_pose_document:
                    raise ValueError("pose JSON must contain a non-empty cameras object")
                distance_unit = pose_document.get("distance_unit", "unspecified")
                if not isinstance(distance_unit, str) or not distance_unit.strip():
                    raise ValueError("distance_unit must be a non-empty string")
                camera_ids = [Path(upload.name).stem for upload in multi_cloud_files]
                if len(camera_ids) != len(set(camera_ids)):
                    raise ValueError("uploaded PLY filenames must have unique camera IDs")
                if set(camera_ids) != set(camera_pose_document):
                    raise ValueError(
                        "PLY filename stems must match the camera IDs in the pose JSON exactly"
                    )
                camera_clouds = {}
                camera_poses = {}
                with st.spinner("Reading camera point clouds and registering them in the shared world frame..."):
                    for camera_id, uploaded in zip(camera_ids, multi_cloud_files):
                        camera_clouds[camera_id] = read_ascii_ply(uploaded.getvalue())
                        camera_pose = load_camera_pose_json(
                            json.dumps(camera_pose_document[camera_id])
                        )
                        camera_poses[camera_id] = (
                            camera_pose.rotation_matrix, camera_pose.translation_vector
                        )
                    fusion_result = fuse_multicamera_point_clouds(
                        camera_clouds, camera_poses, voxel_size=float(multi_voxel_size)
                    )
                if fusion_result.output_points == 0:
                    st.warning("No vertices were present in the uploaded camera point clouds.")
                else:
                    st.success(
                        f"Fused {fusion_result.input_points:,} input points into "
                        f"{fusion_result.output_points:,} world-frame points."
                    )
                    metric_cols = st.columns(3)
                    metric_cols[0].metric("Cameras fused", len(fusion_result.cameras))
                    metric_cols[1].metric("Input points", f"{fusion_result.input_points:,}")
                    metric_cols[2].metric("Output points", f"{fusion_result.output_points:,}")
                    camera_summary = pd.DataFrame([
                        {"Camera": item.camera_id, "Input points": item.input_points,
                         "Valid points": item.valid_points}
                        for item in fusion_result.cameras
                    ])
                    st.dataframe(camera_summary, use_container_width=True, hide_index=True)
                    preview_step = max(1, len(fusion_result.points) // 10000)
                    preview = pd.DataFrame(
                        fusion_result.points[::preview_step, :3], columns=["X", "Y", "Z"]
                    )
                    scene_figure = px.scatter_3d(
                        preview, x="X", y="Y", z="Z", color="Z",
                        title="Fused multi-camera 3D point cloud · sampled preview",
                    )
                    scene_figure.update_layout(height=520)
                    st.plotly_chart(scene_figure, use_container_width=True)
                    with tempfile.TemporaryDirectory() as output_dir:
                        fused_path = write_ply(
                            Path(output_dir) / "blindspot_multicamera_fused.ply",
                            fusion_result.points,
                        )
                        fused_bytes = fused_path.read_bytes()
                    st.download_button(
                        "Download fused multi-camera scene (PLY)", fused_bytes,
                        file_name="blindspot_multicamera_fused.ply",
                        mime="application/octet-stream",
                    )
                    fusion_summary = {
                        "distance_unit": distance_unit,
                        "camera_count": len(fusion_result.cameras),
                        "input_points": fusion_result.input_points,
                        "output_points": fusion_result.output_points,
                        "voxel_size": fusion_result.voxel_size,
                        "cameras": [
                            {"camera_id": item.camera_id, "input_points": item.input_points,
                             "valid_points": item.valid_points}
                            for item in fusion_result.cameras
                        ],
                    }
                    st.download_button(
                        "Download fusion summary (JSON)",
                        json.dumps(fusion_summary, indent=2).encode("utf-8"),
                        file_name="blindspot_multicamera_fusion_summary.json",
                        mime="application/json",
                    )
            except Exception as exc:
                st.error(f"Multi-camera 3D fusion failed: {exc}")

    else:
        st.write(
            "Estimate where two calibrated cameras project onto a known horizontal plane. "
            "This requires stereo calibration plus camera A's camera-to-world pose."
        )
        st.warning(
            "This is a planar geometric footprint only, not 3D reconstruction or proof of "
            "visibility. Camera pose, plane height, and stereo translation must use the same "
            "world coordinate frame and distance unit."
        )
        st.download_button(
            "Download example downward-facing camera pose JSON",
            json.dumps({
                "rotation_matrix": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                "translation_vector": [0, 0, 2],
            }, indent=2).encode("utf-8"),
            file_name="example_camera_a_world_pose.json",
            mime="application/json",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            coverage_intrinsics_a = st.file_uploader(
                "Camera A intrinsics JSON", type=["json"], key="coverage_intrinsics_a"
            )
            coverage_pose_a = st.file_uploader(
                "Camera A to-world pose JSON", type=["json"], key="coverage_pose_a"
            )
        with col_b:
            coverage_intrinsics_b = st.file_uploader(
                "Camera B intrinsics JSON", type=["json"], key="coverage_intrinsics_b"
            )
            coverage_stereo = st.file_uploader(
                "Stereo extrinsics JSON", type=["json"], key="coverage_stereo"
            )
        plane_z = st.number_input("World ground-plane Z", value=0.0, step=0.1)
        coverage_grid_size = st.slider(
            "Coverage map resolution", min_value=32, max_value=256, value=128, step=16
        )
        ready_coverage = bool(
            coverage_intrinsics_a and coverage_intrinsics_b and coverage_pose_a and coverage_stereo
        )
        if st.button("Estimate planar camera coverage", type="primary", disabled=not ready_coverage):
            try:
                calibration_a = load_calibration_json(coverage_intrinsics_a.getvalue().decode("utf-8"))
                calibration_b = load_calibration_json(coverage_intrinsics_b.getvalue().decode("utf-8"))
                stereo = load_stereo_calibration_json(coverage_stereo.getvalue().decode("utf-8"))
                pose_a = load_camera_pose_json(coverage_pose_a.getvalue().decode("utf-8"))
                with st.spinner("Projecting image footprints onto the selected world plane..."):
                    coverage_result = estimate_stereo_planar_coverage(
                        calibration_a, calibration_b, stereo, pose_a,
                        plane_z=float(plane_z), grid_size=int(coverage_grid_size),
                    )
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Camera A footprint", f"{coverage_result.camera_a_area:.2f} {stereo.translation_unit}²")
                m2.metric("Camera B footprint", f"{coverage_result.camera_b_area:.2f} {stereo.translation_unit}²")
                m3.metric("Overlap area", f"{coverage_result.overlap_area:.2f} {stereo.translation_unit}²")
                m4.metric("Combined area", f"{coverage_result.union_area:.2f} {stereo.translation_unit}²")
                coverage_figure = px.imshow(
                    coverage_result.coverage_count_map,
                    x=np.linspace(coverage_result.x_min, coverage_result.x_max, coverage_result.coverage_count_map.shape[1]),
                    y=np.linspace(coverage_result.y_max, coverage_result.y_min, coverage_result.coverage_count_map.shape[0]),
                    origin="upper",
                    labels={"x": f"World X ({stereo.translation_unit})", "y": f"World Y ({stereo.translation_unit})", "color": "Camera count"},
                    title="Planar camera footprint overlap (0, 1, or 2 cameras)",
                )
                coverage_figure.update_layout(height=500, yaxis_scaleanchor="x")
                st.plotly_chart(coverage_figure, use_container_width=True)
                st.download_button(
                    "Download planar coverage grid (CSV)",
                    pd.DataFrame(coverage_result.coverage_count_map).to_csv(index=False).encode("utf-8"),
                    file_name="blindspot_planar_coverage_grid.csv", mime="text/csv",
                )
                coverage_summary = {
                    "distance_unit": stereo.translation_unit,
                    "plane_z": coverage_result.plane_z,
                    "camera_a_area": coverage_result.camera_a_area,
                    "camera_b_area": coverage_result.camera_b_area,
                    "overlap_area": coverage_result.overlap_area,
                    "union_area": coverage_result.union_area,
                    "world_x_min": coverage_result.x_min,
                    "world_x_max": coverage_result.x_max,
                    "world_y_min": coverage_result.y_min,
                    "world_y_max": coverage_result.y_max,
                    "camera_a_polygon_xy": coverage_result.polygon_a.tolist(),
                    "camera_b_polygon_xy": coverage_result.polygon_b.tolist(),
                    "grid_resolution": int(coverage_grid_size),
                }
                st.download_button(
                    "Download planar coverage summary (JSON)",
                    json.dumps(coverage_summary, indent=2).encode("utf-8"),
                    file_name="blindspot_planar_coverage_summary.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"Planar coverage estimation failed: {exc}")



st.divider()
st.caption(
    "Research prototype only · No drone control · No calibrated probability or safety guarantee."
)
