"""Visual loop-closure candidate retrieval and geometric verification.

This module proposes revisited-frame pairs only. It deliberately does not
invent a metric camera pose or silently modify an odometry trajectory; a
candidate must be independently reviewed and converted to a measured pose
constraint before pose-graph correction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class LoopClosureCandidate:
    earlier_index: int
    later_index: int
    match_count: int
    inlier_count: int
    inlier_ratio: float
    verification_model: str
    score: float


def _gray_u8(frame: np.ndarray, index: int) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8:
        raise ValueError(f"frame {index} must be a non-empty uint8 image")
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] in (3, 4):
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"frame {index} must be grayscale, BGR, or BGRA")
    

def detect_loop_closure_candidates(
    frames: Sequence[np.ndarray],
    min_frame_gap: int = 10,
    min_matches: int = 12,
    min_inliers: int = 10,
    min_inlier_ratio: float = 0.45,
    max_features: int = 2500,
    ratio_threshold: float = 0.75,
) -> list[LoopClosureCandidate]:
    """Retrieve and geometrically verify likely revisited views.

    ORB descriptors are compared only across frame pairs separated by at
    least min_frame_gap. Fundamental-matrix RANSAC is preferred; a
    homography RANSAC fallback supports approximately planar scenes. Results
    are ranked by inlier ratio and count. They are hypotheses, not metric
    relative poses or proof of loop closure.
    """
    if not isinstance(frames, (list, tuple)) or len(frames) < 2:
        raise ValueError("frames must contain at least two images")
    for name, value, minimum in (
        ("min_frame_gap", min_frame_gap, 1),
        ("min_matches", min_matches, 4),
        ("min_inliers", min_inliers, 4),
        ("max_features", max_features, 32),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not np.isfinite(min_inlier_ratio) or not 0 < min_inlier_ratio <= 1:
        raise ValueError("min_inlier_ratio must be in (0, 1]")
    if not np.isfinite(ratio_threshold) or not 0 < ratio_threshold < 1:
        raise ValueError("ratio_threshold must be in (0, 1)")
    gray = [_gray_u8(frame, index) for index, frame in enumerate(frames)]
    if len({image.shape for image in gray}) != 1:
        raise ValueError("all frames must have the same image dimensions")

    detector = cv2.ORB_create(nfeatures=max_features)
    features = [detector.detectAndCompute(image, None) for image in gray]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    candidates: list[LoopClosureCandidate] = []

    for earlier in range(len(frames)):
        key_a, desc_a = features[earlier]
        if desc_a is None or len(key_a) < min_matches:
            continue
        for later in range(earlier + min_frame_gap, len(frames)):
            key_b, desc_b = features[later]
            if desc_b is None or len(key_b) < min_matches:
                continue
            pairs = matcher.knnMatch(desc_a, desc_b, k=2)
            good = [
                first for pair in pairs if len(pair) == 2
                for first, second in [pair]
                if first.distance < ratio_threshold * second.distance
            ]
            if len(good) < min_matches:
                continue
            points_a = np.float32([key_a[item.queryIdx].pt for item in good])
            points_b = np.float32([key_b[item.trainIdx].pt for item in good])

            model = "fundamental"
            inlier_mask = None
            try:
                _, inlier_mask = cv2.findFundamentalMat(
                    points_a, points_b, cv2.FM_RANSAC, 2.0, 0.99
                )
            except cv2.error:
                inlier_mask = None
            inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
            if inliers < min_inliers:
                model = "homography"
                try:
                    _, inlier_mask = cv2.findHomography(
                        points_a, points_b, cv2.RANSAC, 3.0
                    )
                except cv2.error:
                    inlier_mask = None
                inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
            ratio = inliers / len(good)
            if inliers < min_inliers or ratio < min_inlier_ratio:
                continue
            score = float(ratio * np.log1p(inliers))
            candidates.append(LoopClosureCandidate(
                earlier, later, len(good), inliers, float(ratio), model, score
            ))

    return sorted(candidates, key=lambda item: (-item.score, item.earlier_index, item.later_index))
