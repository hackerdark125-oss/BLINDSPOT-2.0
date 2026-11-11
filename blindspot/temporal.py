"""Temporal change signals for adjacent sampled frames.

This is a visual-change heuristic, not optical flow, object tracking, or obstacle detection.
"""
from __future__ import annotations
import cv2
import numpy as np


def frame_change(previous: np.ndarray, current: np.ndarray) -> float:
    """Return mean absolute grayscale change in [0, 100]."""
    if not isinstance(previous, np.ndarray) or previous.size == 0:
        raise ValueError("previous frame must be a non-empty image")
    if not isinstance(current, np.ndarray) or current.size == 0:
        raise ValueError("current frame must be a non-empty image")
    def gray(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image
        if image.ndim == 3 and image.shape[2] >= 3:
            return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        raise ValueError("frames must be grayscale or have at least 3 channels")
    a, b = gray(previous), gray(current)
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.mean(cv2.absdiff(a, b)) * 100.0 / 255.0)
