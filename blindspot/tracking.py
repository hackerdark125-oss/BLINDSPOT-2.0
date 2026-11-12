"""Lightweight diagnostics for detecting possibly frozen camera feeds."""
from __future__ import annotations

import cv2
import numpy as np


class FrameFreezeDetector:
    """Flag repeated near-identical frames; static scenes can also trigger this."""

    def __init__(self, threshold: float = 0.002, consecutive_samples: int = 5):
        if not 0 <= threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if isinstance(consecutive_samples, bool) or not isinstance(consecutive_samples, int) or consecutive_samples < 2:
            raise ValueError("consecutive_samples must be an integer of at least 2")
        self.threshold = float(threshold)
        self.consecutive_samples = consecutive_samples
        self._previous: np.ndarray | None = None
        self._still_count = 0

    def update(self, frame: np.ndarray) -> tuple[float, bool]:
        """Return normalized frame-change score and possible-freeze flag."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty NumPy image")
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] in (3, 4):
            gray = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("frame must be grayscale, BGR, or BGRA")

        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        if self._previous is None:
            change = 1.0
        else:
            change = float(np.mean(cv2.absdiff(small, self._previous)) / 255.0)
        self._previous = small
        self._still_count = self._still_count + 1 if change <= self.threshold else 0
        return change, self._still_count >= self.consecutive_samples
