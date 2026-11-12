"""Explainable visual-quality analysis for sampled video frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from blindspot.motion import MotionFreshnessTracker


@dataclass(frozen=True)
class FrameMetrics:
    mean_brightness: float
    contrast: float
    sharpness: float
    edge_density: float
    confidence: float
    uncertainty: float


@dataclass(frozen=True)
class FrameAnalysis:
    timestamp_seconds: float
    metrics: FrameMetrics
    confidence_map: np.ndarray
    motion_magnitude_map: np.ndarray | None = None
    motion_age_seconds: np.ndarray | None = None
    mean_motion: float | None = None
    camera_motion_compensated: bool = False
    scene_changed: bool = False


def _validate_grid_size(grid_size: int) -> None:
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer")


def _normalise(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def analyse_frame(frame: np.ndarray, grid_size: int = 8) -> FrameAnalysis:
    """Return an explainable local image-quality map with scores from 0 to 100."""
    _validate_grid_size(grid_size)
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty NumPy image")
    if frame.dtype != np.uint8:
        raise ValueError("frame must use uint8 pixel values")
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("frame must be grayscale, BGR, or BGRA")

    gray = cv2.resize(gray, (max(grid_size * 24, 96), max(grid_size * 24, 96)))
    h, w = gray.shape
    scores, brightnesses, contrasts, sharpnesses, edges_all = [], [], [], [], []
    for row in range(grid_size):
        for col in range(grid_size):
            patch = gray[row*h//grid_size:(row+1)*h//grid_size,
                         col*w//grid_size:(col+1)*w//grid_size]
            brightness = float(np.mean(patch))
            contrast = float(np.std(patch))
            sharpness = float(cv2.Laplacian(patch, cv2.CV_64F).var())
            edge_density = float(np.mean(cv2.Canny(patch, 80, 160) > 0))
            exposure = 1.0 - min(abs(brightness - 127.5) / 127.5, 1.0)
            detail = 0.55*_normalise(contrast, 5, 55) + 0.45*_normalise(sharpness, 8, 500)
            edge_score = _normalise(edge_density, 0.01, 0.25)
            scores.append(float(np.clip(100*(0.35*exposure + 0.45*detail + 0.20*edge_score), 0, 100)))
            brightnesses.append(brightness)
            contrasts.append(contrast)
            sharpnesses.append(sharpness)
            edges_all.append(edge_density)

    confidence_map = np.asarray(scores, dtype=np.float32).reshape(grid_size, grid_size)
    confidence = float(np.mean(confidence_map))
    metrics = FrameMetrics(
        float(np.mean(brightnesses)), float(np.mean(contrasts)),
        float(np.mean(sharpnesses)), float(np.mean(edges_all)),
        confidence, 100-confidence,
    )
    return FrameAnalysis(0.0, metrics, confidence_map)


def sample_video(path: str, sample_every_seconds: float = 1.0,
                 max_samples: int = 120, grid_size: int = 8) -> list[FrameAnalysis]:
    """Sample a video and calculate per-cell optical-flow activity diagnostics."""
    _validate_grid_size(grid_size)
    if not np.isfinite(sample_every_seconds) or sample_every_seconds <= 0:
        raise ValueError("sample_every_seconds must be a positive finite number")
    if isinstance(max_samples, bool) or not isinstance(max_samples, (int, np.integer)) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")
    if not str(path).strip():
        raise ValueError("video path must not be empty")

    capture = cv2.VideoCapture(str(Path(path)))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    step = max(1, int(round(fps * sample_every_seconds)))
    results, index = [], 0
    motion_tracker = MotionFreshnessTracker(grid_size=grid_size)
    try:
        while len(results) < max_samples:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                timestamp = index / fps
                result = analyse_frame(frame, grid_size)
                motion = motion_tracker.update(frame, timestamp)
                results.append(FrameAnalysis(
                    timestamp, result.metrics, result.confidence_map,
                    motion.magnitude_map, motion.activity_age_seconds,
                    motion.mean_magnitude, motion.camera_motion_compensated,
                    motion.scene_changed,
                ))
            index += 1
    finally:
        capture.release()
    if not results:
        raise ValueError("No readable frames were found in this video")
    return results


def prioritise_cells(confidence_map: np.ndarray, limit: int = 5) -> list[dict[str, float | int]]:
    """Return lowest-confidence grid cells first (zero-based row/column)."""
    if not isinstance(confidence_map, np.ndarray) or confidence_map.ndim != 2 or confidence_map.size == 0:
        raise ValueError("confidence_map must be a non-empty 2D NumPy array")
    if not np.issubdtype(confidence_map.dtype, np.number) or not np.isfinite(confidence_map).all():
        raise ValueError("confidence_map must contain only finite numeric values")
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
        raise ValueError("limit must be an integer")
    if limit < 1:
        return []
    indices = np.argsort(confidence_map, axis=None)[:limit]
    width = confidence_map.shape[1]
    return [
        {"row": int(i//width), "column": int(i%width),
         "confidence": float(confidence_map.flat[i]),
         "uncertainty": float(100-confidence_map.flat[i])}
        for i in indices
    ]
