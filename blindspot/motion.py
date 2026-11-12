"""Temporal motion diagnostics with optional global camera-motion compensation.

Activity age is time since measurable residual motion, not observation freshness.
Global motion is estimated from ORB feature correspondences and removed before
Farneback flow. Scene cuts reset temporal history rather than creating a false
motion spike. Both mechanisms are conservative fallbacks, not visual odometry.
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
    camera_motion_compensated: bool = False
    scene_changed: bool = False


class MotionFreshnessTracker:
    """Track residual motion per cell while discounting estimated global motion."""

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
        camera_motion_compensation: bool = True,
        scene_change_threshold: float = 0.35,
        min_feature_matches: int = 8,
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
        if not isinstance(camera_motion_compensation, bool):
            raise ValueError("camera_motion_compensation must be a boolean")
        if not np.isfinite(scene_change_threshold) or not 0 <= scene_change_threshold <= 1:
            raise ValueError("scene_change_threshold must be between 0 and 1")
        if isinstance(min_feature_matches, bool) or not isinstance(min_feature_matches, int) or min_feature_matches < 4:
            raise ValueError("min_feature_matches must be an integer of at least 4")
        self.grid_size = grid_size
        self.motion_threshold = float(motion_threshold)
        self.parameters = (pyr_scale, levels, winsize, iterations, poly_n, poly_sigma)
        self.camera_motion_compensation = camera_motion_compensation
        self.scene_change_threshold = float(scene_change_threshold)
        self.min_feature_matches = min_feature_matches
        self._previous: np.ndarray | None = None
        self._last_activity: np.ndarray | None = None
        self._last_timestamp: float | None = None
        self._orb = cv2.ORB_create(nfeatures=500) if camera_motion_compensation else None
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) if camera_motion_compensation else None

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

    def _is_scene_change(self, previous: np.ndarray, current: np.ndarray) -> bool:
        previous_hist = cv2.calcHist([previous], [0], None, [32], [0, 256])
        current_hist = cv2.calcHist([current], [0], None, [32], [0, 256])
        cv2.normalize(previous_hist, previous_hist)
        cv2.normalize(current_hist, current_hist)
        correlation = cv2.compareHist(previous_hist, current_hist, cv2.HISTCMP_CORREL)
        return bool(correlation < self.scene_change_threshold)

    def _align_previous(self, previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, bool]:
        """Warp previous grayscale frame into current coordinates when affine fit is reliable."""
        if self._orb is None or self._matcher is None:
            return previous, False
        key_prev, desc_prev = self._orb.detectAndCompute(previous, None)
        key_curr, desc_curr = self._orb.detectAndCompute(current, None)
        if desc_prev is None or desc_curr is None or len(key_prev) < self.min_feature_matches or len(key_curr) < self.min_feature_matches:
            return previous, False
        pairs = self._matcher.knnMatch(desc_prev, desc_curr, k=2)
        good = [first for pair in pairs if len(pair) == 2
                for first, second in [pair] if first.distance < 0.75 * second.distance]
        if len(good) < self.min_feature_matches:
            return previous, False
        points_prev = np.float32([key_prev[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        points_curr = np.float32([key_curr[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        transform, inliers = cv2.estimateAffinePartial2D(
            points_prev, points_curr, method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=1000, confidence=0.99,
        )
        if transform is None or inliers is None:
            return previous, False
        inlier_ratio = float(np.mean(inliers))
        if inlier_ratio < 0.40 or not np.isfinite(transform).all():
            return previous, False
        aligned = cv2.warpAffine(
            previous, transform, (current.shape[1], current.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )
        displacement = float(np.linalg.norm(transform[:, 2]))
        compensated = displacement >= 0.5 or abs(float(transform[0, 1])) >= 0.002
        return aligned, compensated

    def _empty_snapshot(self, timestamp_seconds: float, scene_changed: bool = False) -> MotionSnapshot:
        shape = (self.grid_size, self.grid_size)
        zeros = np.zeros(shape, dtype=np.float32)
        self._last_activity = np.full(shape, timestamp_seconds, dtype=np.float32)
        return MotionSnapshot(zeros, zeros.copy(), 0.0, False, scene_changed)

    def update(self, frame: np.ndarray, timestamp_seconds: float) -> MotionSnapshot:
        """Update tracker and return residual motion, age, and diagnostic flags."""
        if not np.isfinite(timestamp_seconds) or timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be a non-negative finite number")
        if self._last_timestamp is not None and timestamp_seconds < self._last_timestamp:
            raise ValueError("timestamps must be non-decreasing")
        current = self._gray(frame)
        if self._previous is None:
            self._previous = current
            self._last_timestamp = float(timestamp_seconds)
            return self._empty_snapshot(float(timestamp_seconds))

        if self._is_scene_change(self._previous, current):
            self._previous = current
            self._last_timestamp = float(timestamp_seconds)
            return self._empty_snapshot(float(timestamp_seconds), scene_changed=True)

        aligned_previous, compensated = self._align_previous(self._previous, current)
        flow = cv2.calcOpticalFlowFarneback(
            aligned_previous, current, None, *self.parameters, 0
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
            cell_magnitudes, ages, float(np.mean(cell_magnitudes)),
            compensated, False,
        )
