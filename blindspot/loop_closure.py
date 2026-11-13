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


@dataclass(frozen=True)
class MetricLoopClosureVerification:
    """RANSAC result for a rigid transform between paired metric 3D points."""

    rotation: np.ndarray
    translation: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    match_count: int
    inlier_ratio: float
    rmse: float
    median_error: float
    threshold: float
    verified: bool


def _fit_metric_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, singular_values, vt = np.linalg.svd(covariance)
    if len(singular_values) < 2 or singular_values[1] <= 1e-10:
        raise ValueError("3D correspondences are degenerate or collinear")
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def verify_metric_loop_closure(
    source_points: object,
    target_points: object,
    threshold: float = 0.05,
    min_inliers: int = 12,
    min_inlier_ratio: float = 0.35,
    max_iterations: int = 1000,
    random_seed: int = 0,
) -> MetricLoopClosureVerification:
    """RANSAC-verify a metric transform from matched stereo 3D feature points.

    The returned transform maps source coordinates into target coordinates.
    Distances and threshold use the same unit as the input clouds. This verifies
    a retrieved candidate only; it does not create image feature matches.
    """
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 3 or not np.isfinite(source).all():
        raise ValueError("source_points must be a finite Nx3 array with N >= 3")
    if target.ndim != 2 or target.shape != source.shape or not np.isfinite(target).all():
        raise ValueError("target_points must be a finite Nx3 array matching source_points")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be positive and finite")
    if isinstance(min_inliers, bool) or not isinstance(min_inliers, int) or min_inliers < 3:
        raise ValueError("min_inliers must be an integer >= 3")
    if not np.isfinite(min_inlier_ratio) or not 0 < min_inlier_ratio <= 1:
        raise ValueError("min_inlier_ratio must be in (0, 1]")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")
    if len(source) < min_inliers:
        raise ValueError("fewer 3D matches than min_inliers")

    rng = np.random.default_rng(random_seed)
    best_mask = np.zeros(len(source), dtype=bool)
    best_rmse = np.inf
    valid_models = 0
    for _ in range(max_iterations):
        sample = rng.choice(len(source), size=3, replace=False)
        try:
            rotation, translation = _fit_metric_rigid(source[sample], target[sample])
        except ValueError:
            continue
        valid_models += 1
        errors = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
        mask = errors <= threshold
        count = int(mask.sum())
        rmse = float(np.sqrt(np.mean(errors[mask] ** 2))) if count else np.inf
        if count > int(best_mask.sum()) or (count == int(best_mask.sum()) and rmse < best_rmse):
            best_mask, best_rmse = mask, rmse
            if count == len(source):
                break

    if valid_models == 0:
        raise ValueError("all sampled 3D correspondences are degenerate or collinear")
    if int(best_mask.sum()) < 3:
        rotation, translation = np.eye(3), np.zeros(3)
        errors = np.full(len(source), np.inf)
    else:
        rotation, translation = _fit_metric_rigid(source[best_mask], target[best_mask])
        for _ in range(5):
            errors = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
            refined = errors <= threshold
            if int(refined.sum()) < 3 or np.array_equal(refined, best_mask):
                break
            best_mask = refined
            rotation, translation = _fit_metric_rigid(source[best_mask], target[best_mask])
        errors = np.linalg.norm((rotation @ source.T).T + translation - target, axis=1)
        best_mask = errors <= threshold

    count = int(best_mask.sum())
    ratio = float(count / len(source))
    rmse = float(np.sqrt(np.mean(errors[best_mask] ** 2))) if count else float("inf")
    median = float(np.median(errors[best_mask])) if count else float("inf")
    return MetricLoopClosureVerification(
        rotation=rotation.copy(), translation=translation.copy(),
        inlier_mask=best_mask.copy(), inlier_count=count,
        match_count=len(source), inlier_ratio=ratio, rmse=rmse,
        median_error=median, threshold=float(threshold),
        verified=bool(count >= min_inliers and ratio >= min_inlier_ratio),
    )


@dataclass(frozen=True)
class StereoFrameLoopClosureResult:
    """Metric verification result plus image/depth correspondence counts."""

    verification: MetricLoopClosureVerification
    feature_match_count: int
    valid_depth_match_count: int


def verify_stereo_frame_loop_closure(
    earlier_frame: np.ndarray,
    later_frame: np.ndarray,
    earlier_depth: np.ndarray,
    later_depth: np.ndarray,
    camera_matrix: object,
    threshold: float = 0.05,
    min_inliers: int = 12,
    min_inlier_ratio: float = 0.35,
    max_features: int = 2500,
    ratio_threshold: float = 0.75,
    max_iterations: int = 1000,
    random_seed: int = 0,
) -> StereoFrameLoopClosureResult:
    """Extract ORB matches, back-project stereo depths, and verify metric closure.

    Both depth maps must be aligned with their corresponding rectified frames
    and use the same distance unit. The intrinsic matrix must describe those
    exact rectified images. The transform maps earlier-camera 3D points into
    later-camera coordinates. RANSAC rejects inconsistent matches; it cannot
    detect incorrect calibration, moving features, or repeated-pattern aliases.
    """
    gray_a = _gray_u8(earlier_frame, 0)
    gray_b = _gray_u8(later_frame, 1)
    if gray_a.shape != gray_b.shape:
        raise ValueError("earlier_frame and later_frame must have the same dimensions")
    depth_a = np.asarray(earlier_depth, dtype=np.float64)
    depth_b = np.asarray(later_depth, dtype=np.float64)
    if depth_a.shape != gray_a.shape or depth_b.shape != gray_b.shape:
        raise ValueError("both depth maps must match their corresponding frame dimensions")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError("camera_matrix must be finite 3x3 with positive focal lengths")
    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 32:
        raise ValueError("max_features must be an integer >= 32")
    if not np.isfinite(ratio_threshold) or not 0 < ratio_threshold < 1:
        raise ValueError("ratio_threshold must be in (0, 1)")

    detector = cv2.ORB_create(nfeatures=max_features)
    key_a, desc_a = detector.detectAndCompute(gray_a, None)
    key_b, desc_b = detector.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None:
        raise ValueError("insufficient ORB features for metric loop closure")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [
        first for pair in pairs if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio_threshold * second.distance
    ]
    if not good:
        raise ValueError("no reliable ORB feature matches for metric loop closure")

    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    points_a, points_b = [], []
    for match in good:
        x_a, y_a = key_a[match.queryIdx].pt
        x_b, y_b = key_b[match.trainIdx].pt
        col_a, row_a = int(round(x_a)), int(round(y_a))
        col_b, row_b = int(round(x_b)), int(round(y_b))
        if not (0 <= row_a < depth_a.shape[0] and 0 <= col_a < depth_a.shape[1]
                and 0 <= row_b < depth_b.shape[0] and 0 <= col_b < depth_b.shape[1]):
            continue
        z_a, z_b = depth_a[row_a, col_a], depth_b[row_b, col_b]
        if not np.isfinite(z_a) or not np.isfinite(z_b) or z_a <= 0 or z_b <= 0:
            continue
        points_a.append(((x_a - cx) * z_a / fx, (y_a - cy) * z_a / fy, z_a))
        points_b.append(((x_b - cx) * z_b / fx, (y_b - cy) * z_b / fy, z_b))
    if len(points_a) < min_inliers:
        raise ValueError("too few matched features have valid depth in both frames")

    verification = verify_metric_loop_closure(
        np.asarray(points_a), np.asarray(points_b),
        threshold=threshold, min_inliers=min_inliers,
        min_inlier_ratio=min_inlier_ratio, max_iterations=max_iterations,
        random_seed=random_seed,
    )
    return StereoFrameLoopClosureResult(
        verification=verification,
        feature_match_count=len(good),
        valid_depth_match_count=len(points_a),
    )
