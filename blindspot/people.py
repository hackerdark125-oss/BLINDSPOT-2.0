"""OpenCV HOG pedestrian detection and lightweight centroid tracking."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    score: float
    @property
    def centroid(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

@dataclass(frozen=True)
class TrackedDetection:
    track_id: int
    detection: Detection

class CentroidTracker:
    """Greedy nearest-centroid association with a bounded missed-frame lifetime."""
    def __init__(self, max_distance: float = 80.0, max_disappeared: int = 8) -> None:
        if not np.isfinite(max_distance) or max_distance <= 0:
            raise ValueError("max_distance must be a positive finite number")
        if isinstance(max_disappeared, bool) or not isinstance(max_disappeared, int) or max_disappeared < 0:
            raise ValueError("max_disappeared must be a non-negative integer")
        self.max_distance = float(max_distance)
        self.max_disappeared = max_disappeared
        self._next_id = 1
        self._centroids: dict[int, tuple[float, float]] = {}
        self._missed: dict[int, int] = {}

    def _register(self, point: tuple[float, float]) -> int:
        track_id = self._next_id
        self._next_id += 1
        self._centroids[track_id] = point
        self._missed[track_id] = 0
        return track_id

    def _age_unmatched(self, matched: set[int]) -> None:
        for track_id in list(self._centroids):
            if track_id not in matched:
                self._missed[track_id] += 1
                if self._missed[track_id] > self.max_disappeared:
                    del self._centroids[track_id]
                    del self._missed[track_id]

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        if not isinstance(detections, list) or any(not isinstance(d, Detection) for d in detections):
            raise ValueError("detections must be a list of Detection objects")
        if not self._centroids:
            return [TrackedDetection(self._register(d.centroid), d) for d in detections]
        if not detections:
            self._age_unmatched(set())
            return []
        track_ids = list(self._centroids)
        points = [d.centroid for d in detections]
        distances = [
            (float(np.linalg.norm(np.asarray(self._centroids[t]) - np.asarray(p))), t, i)
            for t in track_ids for i, p in enumerate(points)
        ]
        matched_tracks: set[int] = set()
        matched_indices: set[int] = set()
        assignments: dict[int, int] = {}
        for distance, track_id, index in sorted(distances):
            if distance > self.max_distance:
                break
            if track_id in matched_tracks or index in matched_indices:
                continue
            matched_tracks.add(track_id)
            matched_indices.add(index)
            assignments[index] = track_id
            self._centroids[track_id] = points[index]
            self._missed[track_id] = 0
        self._age_unmatched(matched_tracks)
        for index, detection in enumerate(detections):
            if index not in matched_indices:
                assignments[index] = self._register(detection.centroid)
        return [TrackedDetection(assignments[i], d) for i, d in enumerate(detections)]

class HOGPeopleDetector:
    """OpenCV's built-in HOG + linear SVM detector for upright pedestrians."""
    def __init__(self, hit_threshold: float = 0.0, scale: float = 1.05) -> None:
        if not np.isfinite(hit_threshold):
            raise ValueError("hit_threshold must be finite")
        if not np.isfinite(scale) or scale <= 1.0:
            raise ValueError("scale must be greater than 1")
        self.hit_threshold = float(hit_threshold)
        self.scale = float(scale)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy image")
        if frame.dtype != np.uint8:
            raise ValueError("frame must use uint8 pixel values")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a three-channel BGR image")
        height, width = frame.shape[:2]
        if width < 64 or height < 128:
            return []
        boxes, weights = self._hog.detectMultiScale(
            frame, hitThreshold=self.hit_threshold,
            winStride=(8, 8), padding=(8, 8), scale=self.scale,
        )
        if len(boxes) == 0:
            return []
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), [float(v) for v in weights],
            score_threshold=self.hit_threshold, nms_threshold=0.45,
        )
        output = []
        for index in np.asarray(indices).reshape(-1):
            x, y, w, h = map(int, boxes[int(index)])
            x0, y0 = max(0, x), max(0, y)
            output.append(Detection(x0, y0, min(w, width-x0), min(h, height-y0), float(weights[int(index)])))
        return output

@dataclass(frozen=True)
class TrackingFrame:
    timestamp_seconds: float
    annotated_frame: np.ndarray
    detections: tuple[TrackedDetection, ...]

def process_video(path: str, sample_every_frames: int = 5, max_samples: int = 120,
                  detector_scale: float = 1.05) -> list[TrackingFrame]:
    """Detect and track pedestrians in sampled video frames."""
    if not str(path).strip():
        raise ValueError("video path must not be empty")
    if isinstance(sample_every_frames, bool) or not isinstance(sample_every_frames, int) or sample_every_frames < 1:
        raise ValueError("sample_every_frames must be a positive integer")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    detector, tracker = HOGPeopleDetector(scale=detector_scale), CentroidTracker()
    results, frame_index = [], 0
    try:
        while len(results) < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_every_frames == 0:
                tracked = tracker.update(detector.detect(frame))
                annotated = frame.copy()
                for item in tracked:
                    box = item.detection
                    cv2.rectangle(annotated, (box.x, box.y), (box.x+box.width, box.y+box.height), (40, 220, 80), 2)
                    cv2.putText(annotated, f"Person #{item.track_id}", (box.x, max(20, box.y-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 80), 2, cv2.LINE_AA)
                results.append(TrackingFrame(frame_index/fps, annotated, tuple(tracked)))
            frame_index += 1
    finally:
        capture.release()
    if not results:
        raise ValueError("No readable frames were found in this video")
    return results
