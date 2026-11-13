"""Streamlit interface for the drone-only BLINDSPOT AERIAL product."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from blindspot.ai_detection import COCO_CLASSES, YOLODetector, process_stream, process_video
from blindspot.drone import (
    georeference_point_cloud, interpolate_telemetry, parse_telemetry_csv,
)
from blindspot.reconstruction import depth_to_point_cloud, write_ply
from blindspot.calibration import load_calibration_json
from blindspot.stereo import load_stereo_calibration_json, rectify_and_compute_depth


DRONE_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "boat",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
]


def _tracking_table(results) -> pd.DataFrame:
    rows = []
    for sample in results:
        for detection in sample.detections:
            rows.append({
                "timestamp_seconds": sample.timestamp_seconds,
                "frame_index": sample.frame_index,
                "track_id": detection.track_id,
                "class": detection.class_name,
                "confidence": detection.confidence,
                "x": detection.x,
                "y": detection.y,
                "width": detection.width,
                "height": detection.height,
                "inference_time_ms": sample.inference_time_ms,
            })
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="BLINDSPOT AERIAL", page_icon="🛩️", layout="wide")
    st.title("🛩️ BLINDSPOT AERIAL")
    st.caption("DRONE-ONLY · LIVE AERIAL AI · OBJECT TRACKING · 3D MAPPING")
    st.info(
        "BLINDSPOT AERIAL analyses drone imagery and optional flight telemetry. "
        "It does not control, navigate, or autonomously pilot the aircraft."
    )

    with st.sidebar:
        st.header("Aerial AI settings")
        weights = st.selectbox("Detection model", ["yolo11n.pt", "yolo11s.pt"])
        selected_names = st.multiselect(
            "Object classes",
            DRONE_CLASSES,
            default=["person", "car", "truck", "bird", "dog", "cow"],
        )
        confidence = st.slider("Detection confidence", 0.05, 0.95, 0.25, 0.05)
        image_size = st.selectbox("Inference image size", [416, 512, 640, 768, 960], index=2)
        sample_stride = st.slider("Process every Nth frame", 1, 12, 3)
        max_samples = st.slider("Maximum processed frames", 10, 500, 120, 10)

    detect_tab, telemetry_tab, mapping_tab = st.tabs(
        ["Live / recorded drone AI", "Flight telemetry", "3D mapping"]
    )

    with detect_tab:
        source_mode = st.radio(
            "Drone video source",
            ["Recorded drone video", "Live drone stream (RTSP/HTTP)"],
            horizontal=True,
        )
        results = st.session_state.get("aerial_tracking_results")
        if source_mode == "Recorded drone video":
            upload = st.file_uploader(
                "Upload drone footage", type=["mp4", "mov", "avi", "mkv", "m4v"]
            )
            run = st.button(
                "Run aerial detection and tracking", type="primary",
                disabled=upload is None or not selected_names,
            )
            if run and upload is not None:
                suffix = Path(upload.name).suffix or ".mp4"
                with tempfile.TemporaryDirectory() as folder:
                    video_path = Path(folder) / f"drone_upload{suffix}"
                    video_path.write_bytes(upload.getvalue())
                    try:
                        with st.spinner("Running YOLO detection and ByteTrack..."):
                            detector = YOLODetector(weights)
                            results = process_video(
                                str(video_path), sample_every_frames=sample_stride,
                                max_samples=max_samples, detector=detector,
                                classes=[COCO_CLASSES[name] for name in selected_names],
                                confidence=confidence, image_size=image_size,
                            )
                        st.session_state["aerial_tracking_results"] = results
                    except Exception as exc:
                        st.error(f"Aerial video analysis failed: {exc}")
        else:
            stream_url = st.text_input(
                "Drone video stream URL", placeholder="rtsp://drone-or-ground-station/stream"
            )
            run = st.button(
                "Connect and analyse drone stream", type="primary",
                disabled=not stream_url.strip() or not selected_names,
            )
            if run:
                if not stream_url.startswith(("rtsp://", "rtsps://", "http://", "https://")):
                    st.error("Enter an RTSP, RTSPS, HTTP or HTTPS drone stream URL.")
                else:
                    try:
                        with st.spinner("Connecting to drone video stream..."):
                            detector = YOLODetector(weights)
                            results = process_stream(
                                stream_url, duration_seconds=60,
                                sample_every_frames=sample_stride,
                                max_samples=max_samples, detector=detector,
                                classes=[COCO_CLASSES[name] for name in selected_names],
                                confidence=confidence, image_size=image_size,
                            )
                        st.session_state["aerial_tracking_results"] = results
                    except Exception as exc:
                        st.error(f"Live drone analysis failed: {exc}")

        if results:
            metrics = _tracking_table(results)
            frame_col, stats_col = st.columns([2, 1])
            with frame_col:
                st.subheader("Annotated aerial frames")
                display_indices = np.linspace(
                    0, len(results) - 1, min(6, len(results)), dtype=int
                )
                for index in display_indices:
                    sample = results[int(index)]
                    st.image(
                        cv2.cvtColor(sample.annotated_frame, cv2.COLOR_BGR2RGB),
                        caption=(
                            f"Frame {sample.frame_index} · {sample.timestamp_seconds:.2f}s · "
                            f"{len(sample.detections)} objects · {sample.inference_time_ms:.1f} ms"
                        ),
                        use_container_width=True,
                    )
            with stats_col:
                st.subheader("Flight video analytics")
                st.metric("Frames analysed", len(results))
                st.metric("Objects detected", len(metrics))
                st.metric("Unique track IDs", metrics["track_id"].nunique()
                          if not metrics.empty else 0)
                st.metric("Mean inference", f"{np.mean([r.inference_time_ms for r in results]):.1f} ms")
                if not metrics.empty:
                    st.dataframe(
                        metrics.groupby("class").size().rename("detections").reset_index(),
                        use_container_width=True, hide_index=True,
                    )
            st.download_button(
                "Download aerial detection CSV",
                metrics.to_csv(index=False).encode("utf-8"),
                file_name="blindspot_aerial_detections.csv",
                mime="text/csv",
            )
            st.caption(
                "Track IDs are tracker identities within this processed stream, not real-world "
                "identity. They may change after occlusion, missed detections or reconnects."
            )

    telemetry_content = None
    telemetry_records = None
    with telemetry_tab:
        telemetry_upload = st.file_uploader(
            "Upload flight telemetry CSV",
            type=["csv"],
            help=(
                "Required columns: timestamp_seconds, latitude_deg, longitude_deg, altitude_m. "
                "Optional: roll_deg, pitch_deg, yaw_deg. Video and telemetry timestamps must "
                "share the same start-time reference."
            ),
            key="telemetry_csv",
        )
        if telemetry_upload is not None:
            try:
                telemetry_content = telemetry_upload.getvalue().decode("utf-8-sig")
                telemetry_records = parse_telemetry_csv(telemetry_content)
                telemetry_df = pd.DataFrame([item.__dict__ for item in telemetry_records])
                st.success(f"Loaded {len(telemetry_records)} telemetry samples.")
                st.dataframe(telemetry_df, use_container_width=True, hide_index=True)
                st.map(telemetry_df.rename(columns={
                    "latitude_deg": "latitude", "longitude_deg": "longitude"
                })[["latitude", "longitude"]])
                st.line_chart(telemetry_df.set_index("timestamp_seconds")[
                    ["altitude_m", "roll_deg", "pitch_deg", "yaw_deg"]
                ])
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"Telemetry file could not be loaded: {exc}")

    with mapping_tab:
        st.subheader("Drone stereo depth and georeferenced 3D point-cloud export")
        mapping_mode = st.radio(
            "Depth source",
            ["Precomputed depth map (.npy)", "Generate depth from calibrated stereo pair"],
            horizontal=True,
        )
        depth_upload = None
        intrinsics_upload = None
        left_upload = None
        right_upload = None
        intrinsics_b_upload = None
        stereo_upload = None
        if mapping_mode == "Precomputed depth map (.npy)":
            depth_upload = st.file_uploader("Depth map (.npy)", type=["npy"], key="depth_npy")
            intrinsics_upload = st.file_uploader(
                "Camera A intrinsics JSON", type=["json"], key="mapping_intrinsics"
            )
        else:
            left_upload = st.file_uploader(
                "Synchronized left-camera image", type=["png", "jpg", "jpeg"], key="stereo_left"
            )
            right_upload = st.file_uploader(
                "Synchronized right-camera image", type=["png", "jpg", "jpeg"], key="stereo_right"
            )
            intrinsics_upload = st.file_uploader(
                "Left-camera intrinsics JSON", type=["json"], key="mapping_intrinsics_stereo_a"
            )
            intrinsics_b_upload = st.file_uploader(
                "Right-camera intrinsics JSON", type=["json"], key="mapping_intrinsics_stereo_b"
            )
            stereo_upload = st.file_uploader(
                "Stereo extrinsics JSON", type=["json"], key="mapping_stereo_extrinsics"
            )
        mapping_telemetry_upload = st.file_uploader(
            "Flight telemetry CSV", type=["csv"], key="mapping_telemetry"
        )
        rotation_text = st.text_area(
            "Calibrated camera-to-ENU rotation matrix (3×3 JSON)",
            value="[[1,0,0],[0,1,0],[0,0,1]]",
            help="Replace the example with the measured camera/gimbal-to-ENU rotation.",
        )
        depth_scale_to_m = st.number_input(
            "Depth-unit scale to metres", min_value=0.000001, max_value=1000000.0,
            value=1.0, format="%.6f",
            help="Use 1 for metres, 0.001 for millimetres, or the appropriate conversion factor.",
        )
        stride = st.slider("Point-cloud sampling stride", 1, 12, 3)
        mapping_timestamp = None
        if mapping_telemetry_upload is not None:
            try:
                mapping_records_preview = parse_telemetry_csv(
                    mapping_telemetry_upload.getvalue().decode("utf-8-sig")
                )
                mapping_timestamp = st.number_input(
                    "Telemetry timestamp for this depth map (seconds)",
                    min_value=float(mapping_records_preview[0].timestamp_seconds),
                    max_value=float(mapping_records_preview[-1].timestamp_seconds),
                    value=float(mapping_records_preview[0].timestamp_seconds),
                )
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"Telemetry file could not be loaded: {exc}")

        depth_inputs_ready = (
            depth_upload is not None and intrinsics_upload is not None
            if mapping_mode == "Precomputed depth map (.npy)"
            else all([left_upload, right_upload, intrinsics_upload, intrinsics_b_upload, stereo_upload])
        )
        if st.button(
            "Generate georeferenced PLY",
            type="primary",
            disabled=not depth_inputs_ready or mapping_telemetry_upload is None
                     or mapping_timestamp is None,
        ):
            try:
                intrinsics_a = load_calibration_json(
                    intrinsics_upload.getvalue().decode("utf-8")
                )
                if mapping_mode == "Precomputed depth map (.npy)":
                    depth_map = np.load(depth_upload, allow_pickle=False)
                    matrix = np.asarray(intrinsics_a.camera_matrix, dtype=float)
                    color_image = None
                    valid_mask = None
                else:
                    intrinsics_b = load_calibration_json(
                        intrinsics_b_upload.getvalue().decode("utf-8")
                    )
                    stereo = load_stereo_calibration_json(
                        stereo_upload.getvalue().decode("utf-8")
                    )
                    image_a = cv2.imdecode(
                        np.frombuffer(left_upload.getvalue(), dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    image_b = cv2.imdecode(
                        np.frombuffer(right_upload.getvalue(), dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if image_a is None or image_b is None:
                        raise ValueError("could not decode one or both stereo images")
                    depth_result = rectify_and_compute_depth(
                        image_a, image_b, intrinsics_a, intrinsics_b, stereo
                    )
                    depth_map = depth_result.depth_map
                    matrix = np.asarray(depth_result.rectified_camera_matrix, dtype=float)
                    color_image = depth_result.rectified_a
                    valid_mask = depth_result.valid_mask
                    st.caption(
                        f"Stereo depth generated · calibration unit: {stereo.translation_unit}. "
                        "Set the depth-unit scale correctly to export metres."
                    )

                depth_metres = np.asarray(depth_map, dtype=np.float32) * float(depth_scale_to_m)
                telemetry = parse_telemetry_csv(
                    mapping_telemetry_upload.getvalue().decode("utf-8-sig")
                )
                pose = interpolate_telemetry(telemetry, float(mapping_timestamp))
                rotation = np.asarray(json.loads(rotation_text), dtype=float)
                camera_cloud = depth_to_point_cloud(
                    depth_metres, matrix, color_image=color_image,
                    valid_mask=valid_mask, stride=stride,
                )
                origin = (
                    telemetry[0].latitude_deg,
                    telemetry[0].longitude_deg,
                    telemetry[0].altitude_m,
                )
                world_cloud = georeference_point_cloud(
                    camera_cloud, pose, origin, rotation
                )
                with tempfile.TemporaryDirectory() as folder:
                    path = write_ply(Path(folder) / "blindspot_aerial_map.ply", world_cloud)
                    ply_bytes = path.read_bytes()
                st.success(f"Generated {len(world_cloud):,} georeferenced points.")
                st.download_button(
                    "Download 3D point cloud (PLY)", ply_bytes,
                    file_name="blindspot_aerial_map.ply", mime="application/octet-stream",
                )
                st.warning(
                    "This produces a georeferenced point cloud from one supplied depth map "
                    "or one calibrated stereo image pair. It is not yet a complete multi-frame "
                    "photogrammetry/SLAM map. Accuracy depends on stereo depth, camera/gimbal "
                    "extrinsics, telemetry timing and altitude datum."
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                st.error(f"3D mapping failed: {exc}")

    st.divider()
    st.caption(
        "BLINDSPOT AERIAL is an analysis and mapping aid. It must not be used for autonomous "
        "flight control, navigation, targeting, or safety-critical decisions."
    )


if __name__ == "__main__":
    main()
