"""Modern YOLO object detection with Ultralytics ByteTrack video tracking.

Ultralytics is an optional dependency. Install requirements-ai.txt to enable this
module. Model weights are downloaded by Ultralytics on first use and cached locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import time
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectDetection:
    track_id: int | None
    class_id: int
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class ObjectTrackingFrame:
    frame_index: int
    timestamp_seconds: float
    inference_time_ms: float
    annotated_frame: np.ndarray
    detections: tuple[ObjectDetection, ...]


COCO_CLASSES: dict[str, int] = {
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


class YOLODetector:
    """YOLO object detector whose persistent track IDs are assigned by ByteTrack."""

    def __init__(self, weights: str = "yolo11n.pt", model: Any | None = None) -> None:
        if not isinstance(weights, str) or not weights.strip():
            raise ValueError("weights must be a non-empty model name or path")
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "Modern AI detection requires the optional dependencies. "
                    "Install them with: pip install -r requirements-ai.txt"
                ) from exc
            model = YOLO(weights)
        self.weights = weights
        self.model = model

    def reset_tracker(self) -> None:
        """Reset ByteTrack state before processing a separate video source."""
        predictor = getattr(self.model, "predictor", None)
        if predictor is None:
            return
        for tracker in getattr(predictor, "trackers", []) or []:
            reset = getattr(tracker, "reset", None)
            if callable(reset):
                reset()

    def track_frame(
        self,
        frame: np.ndarray,
        classes: Sequence[int] | None = None,
        confidence: float = 0.25,
        image_size: int = 640,
    ) -> list[ObjectDetection]:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy image")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a uint8 three-channel BGR image")
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 32:
            raise ValueError("image_size must be an integer of at least 32")
        if classes is not None:
            if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in classes):
                raise ValueError("class IDs must be non-negative integers")

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(classes) if classes else None,
            conf=float(confidence),
            imgsz=image_size,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        scores = boxes.conf.cpu().numpy()
        raw_ids = getattr(boxes, "id", None)
        track_ids = raw_ids.cpu().numpy().astype(int) if raw_ids is not None else None
        names = getattr(result, "names", {})
        height, width = frame.shape[:2]
        detections: list[ObjectDetection] = []
        for index, coordinates in enumerate(xyxy):
            x1, y1, x2, y2 = map(int, coordinates)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            class_id = int(class_ids[index])
            if isinstance(names, dict):
                class_name = str(names.get(class_id, f"class_{class_id}"))
            elif isinstance(names, (list, tuple)) and class_id < len(names):
                class_name = str(names[class_id])
            else:
                class_name = f"class_{class_id}"
            detections.append(ObjectDetection(
                track_id=int(track_ids[index]) if track_ids is not None else None,
                class_id=class_id,
                class_name=class_name,
                confidence=float(scores[index]),
                x=x1, y=y1, width=x2 - x1, height=y2 - y1,
            ))
        return detections


def process_video(
    path: str,
    sample_every_frames: int = 3,
    max_samples: int = 120,
    weights: str = "yolo11n.pt",
    classes: Sequence[int] | None = None,
    confidence: float = 0.25,
    image_size: int = 640,
    detector: YOLODetector | None = None,
) -> list[ObjectTrackingFrame]:
    """Run YOLO + persistent ByteTrack on sampled frames of a video."""
    if not str(path).strip():
        raise ValueError("video path must not be empty")
    if isinstance(sample_every_frames, bool) or not isinstance(sample_every_frames, int) or sample_every_frames < 1:
        raise ValueError("sample_every_frames must be a positive integer")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size < 32:
        raise ValueError("image_size must be an integer of at least 32")

    video_path = Path(path)
    if not video_path.exists():
        raise ValueError(f"Video does not exist: {path}")
    detector = detector or YOLODetector(weights)
    detector.reset_tracker()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    output: list[ObjectTrackingFrame] = []
    frame_index = 0
    try:
        while len(output) < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every_frames == 0:
                inference_started = perf_counter()
                detections = detector.track_frame(
                    frame, classes=classes, confidence=confidence, image_size=image_size
                )
                inference_time_ms = (perf_counter() - inference_started) * 1000.0
                annotated = frame.copy()
                for item in detections:
                    x1, y1 = item.x, item.y
                    x2, y2 = x1 + item.width, y1 + item.height
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 80), 2)
                    identity = f"ID {item.track_id}" if item.track_id is not None else "ID pending"
                    label = f"{item.class_name} {identity} {item.confidence:.2f}"
                    cv2.putText(
                        annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 2, cv2.LINE_AA,
                    )
                output.append(ObjectTrackingFrame(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    inference_time_ms=inference_time_ms,
                    annotated_frame=annotated,
                    detections=tuple(detections),
                ))
            frame_index += 1
    finally:
        capture.release()
    if not output:
        raise ValueError("No readable frames were found in this video")
    return output


def process_stream(
    source: str | int,
    duration_seconds: float = 15.0,
    sample_every_frames: int = 3,
    max_samples: int = 120,
    weights: str = "yolo11n.pt",
    classes: Sequence[int] | None = None,
    confidence: float = 0.25,
    image_size: int = 640,
    detector: YOLODetector | None = None,
    read_timeout_seconds: float = 5.0,
) -> list[ObjectTrackingFrame]:
    """Run YOLO + ByteTrack on a bounded local webcam or network stream.

    Frame indices count frames received from the stream during this session;
    timestamps use a monotonic clock and are relative to capture start.
    """
    if isinstance(duration_seconds, bool) or not np.isfinite(duration_seconds) or not 1 <= duration_seconds <= 60:
        raise ValueError("duration_seconds must be between 1 and 60")
    if isinstance(sample_every_frames, bool) or not isinstance(sample_every_frames, int) or sample_every_frames < 1:
        raise ValueError("sample_every_frames must be a positive integer")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size < 32:
        raise ValueError("image_size must be an integer of at least 32")
    if isinstance(read_timeout_seconds, bool) or not np.isfinite(read_timeout_seconds) or read_timeout_seconds <= 0:
        raise ValueError("read_timeout_seconds must be a positive finite number")

    if isinstance(source, bool):
        raise ValueError("source must be a camera index or stream URL")
    if isinstance(source, int):
        if source < 0:
            raise ValueError("camera index must be zero or greater")
        camera_source: str | int = source
    else:
        source_text = str(source).strip()
        if not source_text:
            raise ValueError("source must not be empty")
        if source_text.isdigit():
            camera_source = int(source_text)
        elif source_text.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            camera_source = source_text
        else:
            raise ValueError("source must be a camera index or RTSP/HTTP(S) stream URL")

    detector = detector or YOLODetector(weights)
    detector.reset_tracker()
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        capture.release()
        raise ValueError(
            f"Could not open camera source: {camera_source if isinstance(camera_source, int) else 'stream URL'}"
        )

    output: list[ObjectTrackingFrame] = []
    started = perf_counter()
    last_frame_received = started
    frame_index = 0
    try:
        while perf_counter() - started < duration_seconds and len(output) < max_samples:
            ok, frame = capture.read()
            if not ok:
                # Tolerate short network hiccups but do not wait forever on a dead stream.
                if perf_counter() - last_frame_received >= read_timeout_seconds:
                    break
                time.sleep(0.02)
                continue
            last_frame_received = perf_counter()
            if frame_index % sample_every_frames == 0:
                timestamp = perf_counter() - started
                inference_started = perf_counter()
                detections = detector.track_frame(
                    frame, classes=classes, confidence=confidence, image_size=image_size
                )
                processing_time_ms = (perf_counter() - inference_started) * 1000.0
                annotated = frame.copy()
                for item in detections:
                    x1, y1 = item.x, item.y
                    x2, y2 = x1 + item.width, y1 + item.height
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 80), 2)
                    identity = f"ID {item.track_id}" if item.track_id is not None else "ID pending"
                    label = f"{item.class_name} {identity} {item.confidence:.2f}"
                    cv2.putText(
                        annotated, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 2, cv2.LINE_AA,
                    )
                output.append(ObjectTrackingFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    inference_time_ms=processing_time_ms,
                    annotated_frame=annotated,
                    detections=tuple(detections),
                ))
            frame_index += 1
    finally:
        capture.release()

    if not output:
        raise ValueError("No frames received. Check camera permissions, camera index, or stream URL.")
    return output
