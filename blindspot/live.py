"""Local webcam and RTSP sampling helpers."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from blindspot.engine import FrameAnalysis, analyse_frame
from blindspot.motion import MotionFreshnessTracker
from blindspot.tracking import FrameFreezeDetector


@dataclass(frozen=True)
class LiveSample:
    analysis: FrameAnalysis
    frame_change_score: float
    possible_frozen_feed: bool


def parse_camera_source(source: str | int) -> str | int:
    """Interpret a numeric string as a local camera index; otherwise keep URL."""
    if isinstance(source, int) and not isinstance(source, bool):
        if source < 0:
            raise ValueError("camera index must be zero or greater")
        return source
    value = str(source).strip()
    if not value:
        raise ValueError("camera source must not be empty")
    if value.isdigit():
        return int(value)
    if not (value.startswith("rtsp://") or value.startswith("rtsps://")
            or value.startswith("http://") or value.startswith("https://")):
        raise ValueError("source must be a camera index or RTSP/HTTP(S) stream URL")
    return value


def capture_live_source(
    source: str | int,
    duration_seconds: float = 10.0,
    sample_every_seconds: float = 1.0,
    max_samples: int = 30,
    grid_size: int = 8,
) -> list[LiveSample]:
    """Capture a bounded local webcam/network-stream window and sample it."""
    if not np.isfinite(duration_seconds) or not 1 <= duration_seconds <= 60:
        raise ValueError("duration_seconds must be between 1 and 60")
    if not np.isfinite(sample_every_seconds) or sample_every_seconds <= 0:
        raise ValueError("sample_every_seconds must be positive")
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer")

    camera_source = parse_camera_source(source)
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open camera source: {camera_source if isinstance(camera_source, int) else 'stream URL'}")

    detector = FrameFreezeDetector()
    motion_tracker = MotionFreshnessTracker(grid_size=grid_size)
    samples: list[LiveSample] = []
    started = time.monotonic()
    next_sample = started
    try:
        while time.monotonic() - started < duration_seconds and len(samples) < max_samples:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.02)
                continue
            now = time.monotonic()
            if now >= next_sample:
                timestamp = now - started
                analysis = analyse_frame(frame, grid_size)
                motion = motion_tracker.update(frame, timestamp)
                change, frozen = detector.update(frame)
                analysis = FrameAnalysis(
                    timestamp, analysis.metrics, analysis.confidence_map,
                    motion.magnitude_map, motion.activity_age_seconds,
                    motion.mean_magnitude, motion.camera_motion_compensated,
                    motion.scene_changed,
                )
                samples.append(LiveSample(analysis, change, frozen))
                next_sample = now + sample_every_seconds
    finally:
        capture.release()

    if not samples:
        raise ValueError("No frames received. Check camera permissions, camera index, or stream URL.")
    return samples
