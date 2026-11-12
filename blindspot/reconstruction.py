"""Stereo point-cloud reconstruction and monocular visual-motion estimation.

Point clouds use the same distance unit as the supplied depth map. Monocular
motion returns translation direction only: its magnitude/scale is unobservable
without a known baseline, depth, or another metric reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualMotion:
    rotation_matrix: list[list[float]]
    translation_direction: list[float]
    inlier_count: int
    match_count: int
    inlier_ratio: float
    metric_scale_known: bool = False


def depth_to_point_cloud(
    depth_map: np.ndarray,
    camera_matrix: np.ndarray | list[list[float]],
    color_image: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    stride: int = 2,
    max_depth: float | None = None,
    world_rotation: np.ndarray | list[list[float]] | None = None,
    world_translation: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Back-project a depth image into an Nx6 XYZRGB point array."""
    depth = np.asarray(depth_map, dtype=np.float64)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if depth.ndim != 2 or depth.size == 0:
        raise ValueError("depth_map must be a non-empty 2D array")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= 0):
        raise ValueError("max_depth must be positive and finite")
    if color_image is not None:
        colors = np.asarray(color_image)
        if colors.shape != (*depth.shape, 3) or colors.dtype != np.uint8:
            raise ValueError("color_image must be a uint8 BGR/RGB image matching depth_map")
    else:
        colors = np.zeros((*depth.shape, 3), dtype=np.uint8)
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != depth.shape or mask.dtype != np.bool_:
            raise ValueError("valid_mask must be a boolean array matching depth_map")
        valid &= mask
    rows, cols = np.indices(depth.shape)
    valid &= (rows % stride == 0) & (cols % stride == 0)
    y, x = np.nonzero(valid)
    z = depth[y, x]
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    xyz = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    if world_rotation is not None or world_translation is not None:
        rotation = np.eye(3) if world_rotation is None else np.asarray(world_rotation, dtype=float)
        translation = np.zeros(3) if world_translation is None else np.asarray(world_translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("world transform must contain a finite 3x3 rotation and 3-vector translation")
        xyz = (rotation @ xyz.T).T + translation
    rgb = colors[y, x, ::-1] if color_image is not None else colors[y, x]
    return np.column_stack((xyz, rgb.astype(np.float64))).astype(np.float32)


def write_ply(path: str | Path, points: np.ndarray) -> Path:
    """Write an Nx6 XYZRGB array as an interoperable ASCII PLY point cloud."""
    cloud = np.asarray(points)
    if cloud.ndim != 2 or cloud.shape[1] != 6 or not np.isfinite(cloud).all():
        raise ValueError("points must be a finite Nx6 XYZRGB array")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(cloud)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for row in cloud:
            r, g, b = np.clip(np.rint(row[3:6]), 0, 255).astype(int)
            stream.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {r} {g} {b}\n")
    return destination


def estimate_visual_motion(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    camera_matrix: np.ndarray | list[list[float]],
    min_inliers: int = 8,
    max_features: int = 2000,
) -> VisualMotion:
    """Estimate relative monocular camera rotation and translation direction.

    This is a two-frame visual-motion estimate, not globally corrected visual
    odometry or absolute localisation. Translation scale is inherently unknown.
    """
    frames = []
    for name, frame in (("previous_frame", previous_frame), ("current_frame", current_frame)):
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8:
            raise ValueError(f"{name} must be a non-empty uint8 image")
        if frame.ndim == 2:
            frames.append(frame)
        elif frame.ndim == 3 and frame.shape[2] in (3, 4):
            frames.append(cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY))
        else:
            raise ValueError(f"{name} must be grayscale, BGR, or BGRA")
    if frames[0].shape != frames[1].shape:
        raise ValueError("both frames must have the same dimensions")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera_matrix must be a finite 3x3 matrix with positive focal lengths")
    if isinstance(min_inliers, bool) or not isinstance(min_inliers, int) or min_inliers < 5:
        raise ValueError("min_inliers must be an integer >= 5")
    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < min_inliers:
        raise ValueError("max_features must be an integer >= min_inliers")

    detector = cv2.ORB_create(nfeatures=max_features)
    key_a, desc_a = detector.detectAndCompute(frames[0], None)
    key_b, desc_b = detector.detectAndCompute(frames[1], None)
    if desc_a is None or desc_b is None or len(key_a) < min_inliers or len(key_b) < min_inliers:
        raise ValueError("insufficient visual features for motion estimation")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        raise ValueError("insufficient reliable feature matches for motion estimation")
    points_a = np.float32([key_a[m.queryIdx].pt for m in good])
    points_b = np.float32([key_b[m.trainIdx].pt for m in good])
    essential, mask = cv2.findEssentialMat(
        points_a, points_b, matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if essential is None or mask is None:
        raise ValueError("essential-matrix estimation failed; scene may lack parallax or texture")
    _, rotation, translation, pose_mask = cv2.recoverPose(essential, points_a, points_b, matrix, mask=mask)
    inliers = int(np.count_nonzero(pose_mask))
    if inliers < min_inliers:
        raise ValueError(f"only {inliers} geometric inliers; need at least {min_inliers}")
    return VisualMotion(
        rotation_matrix=rotation.tolist(),
        translation_direction=(translation.reshape(3) / np.linalg.norm(translation)).tolist(),
        inlier_count=inliers,
        match_count=len(good),
        inlier_ratio=float(inliers / len(good)),
    )
