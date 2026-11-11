"""Live webcam / RTSP sampling. Runs on the machine hosting Streamlit."""
from __future__ import annotations
import time
import cv2
from blindspot.engine import FrameAnalysis, analyse_frame


def sample_stream(source: int | str, duration_seconds: float = 15.0,
                  sample_every_seconds: float = 1.0, max_samples: int = 30,
                  grid_size: int = 8) -> list[FrameAnalysis]:
    """Sample a live webcam or RTSP stream for a bounded time window."""
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_every_seconds <= 0:
        raise ValueError("sample_every_seconds must be positive")
    if max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError("Could not open live source. Check camera index, RTSP URL, and network.")
    results: list[FrameAnalysis] = []
    started = time.monotonic()
    next_sample = started
    try:
        while time.monotonic() - started < duration_seconds and len(results) < max_samples:
            ok, frame = capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            if now >= next_sample:
                result = analyse_frame(frame, grid_size)
                elapsed = now - started
                results.append(FrameAnalysis(elapsed, result.metrics, result.confidence_map))
                next_sample = now + sample_every_seconds
    finally:
        capture.release()
    if not results:
        raise ValueError("No frames received from live source during capture window")
    return results
