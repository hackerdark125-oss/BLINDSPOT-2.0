"""Temporal motion activity maps using dense optical flow.

Activity age means time since measurable pixel motion, not proof that a region
was or was not observed. Static scenes can naturally have high activity age.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class MotionSnapshot:
    magnitude_map: np.ndarray
    activity_age_seconds: np.ndarray
    mean_magnitude: float


class MotionFreshnessTracker:
    """Track per-grid-cell motion activity and elapsed time since activity."""

    def __init__(
        self,
        grid_size: int = 8,
        motion_threshold: float = 0.35,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
    ) -> None:
        if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError("grid_size must be a positive integer")
        if not np.isfinite(motion_threshold) or motion_threshold < 0:
            raise ValueError("motion_threshold must be a non-negative finite number")
        if not 0 < pyr_scale < 1:
            raise ValueError("pyr_scale must be between 0 and 1")
        if levels < 1 or winsize < 3 or iterations < 1 or poly_n < 5 or poly_n % 2 == 0:
            raise ValueError("invalid optical-flow parameters")
        if not np.isfinite(poly_sigma) or poly_sigma <= 0:
            raise ValueError("poly_sigma must be positive")
        self.grid_size = grid_size
        self.motion_threshold = float(motion_threshold)
        self.parameters = (pyr_scale, levels, winsize, iterations, poly_n, poly_sigma)
        self._previous: np.ndarray | None = None
        self._last_activity: np.ndarray | None = None
        self._start_time: float | None = None
        self._last_timestamp: float | None = None

    def _gray(self, frame: np.ndarray) -> np.ndarray:
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
        size = max(self.grid_size * 24, 96)
        return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray, timestamp_seconds: float) -> MotionSnapshot:
        """Update tracker and return per-cell motion magnitude and activity age."""
        if not np.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be a non-negative finite number")
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        current = self._gray(frame)
        if self._previous is None:
            self._previous = current
            self._start_time = float(timestamp_seconds)
            self._last_timestamp = float(timestamp_seconds)
            self._last_activity = np.full(
                (self.grid_size, self.grid_size), timestamp_seconds, dtype=np.float32
            )
            zeros = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            return MotionSnapshot(zeros, zeros.copy(), 0.0)

        flow = cv2.calcOpticalFlowFarneback(
            self._previous, current, None, *self.parameters, 0
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        h, w = magnitude.shape
        cell_magnitudes = np.empty((self.grid_size, self.grid_size), dtype=np.float32)
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                patch = magnitude[
                    row*h//self.grid_size:(row+1)*h//self.grid_size,
                    col*w//self.grid_size:(col+1)*w//self.grid_size,
                ]
                cell_magnitudes[row, col] = float(np.mean(patch))

        active = cell_magnitudes >= self.motion_threshold
        self._last_activity[active] = timestamp_seconds
        ages = np.maximum(0.0, timestamp_seconds - self._last_activity).astype(np.float32)
        self._previous = current
        self._last_timestamp = float(timestamp_seconds)
        return MotionSnapshot(
            cell_magnitudes, ages, float(np.mean(cell_magnitudes))
        )
