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


def select_moving_detection_regions(
    detections: list[Any] | tuple[Any, ...],
    previous_centers: dict[int, tuple[float, float]],
    min_displacement_pixels: float = 4.0,
) -> tuple[list[tuple[int, int, int, int]], dict[int, tuple[float, float]]]:
    """Select tracked detections whose centers moved enough since the last sample.

    Detections without a ByteTrack ID are ignored because motion cannot be
    distinguished from a new/static object without persistent identity.
    """
    if not np.isfinite(min_displacement_pixels) or min_displacement_pixels < 0:
        raise ValueError("min_displacement_pixels must be non-negative and finite")
    if not isinstance(previous_centers, dict):
        raise ValueError("previous_centers must be a dictionary")
    centers = dict(previous_centers)
    moving_boxes: list[tuple[int, int, int, int]] = []
    for item in detections:
        track_id = getattr(item, "track_id", None)
        if track_id is None:
            continue
        if isinstance(track_id, bool) or not isinstance(track_id, (int, np.integer)):
            raise ValueError("detection track_id must be an integer or None")
        x, y = getattr(item, "x", None), getattr(item, "y", None)
        width, height = getattr(item, "width", None), getattr(item, "height", None)
        values = (x, y, width, height)
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError("detection boxes must contain integer x, y, width, and height")
        if width <= 0 or height <= 0:
            raise ValueError("detection box dimensions must be positive")
        center = (float(x) + float(width) / 2.0, float(y) + float(height) / 2.0)
        old_center = centers.get(int(track_id))
        if old_center is not None:
            displacement = float(np.hypot(center[0] - old_center[0], center[1] - old_center[1]))
            if displacement >= float(min_displacement_pixels):
                moving_boxes.append((int(x), int(y), int(width), int(height)))
        centers[int(track_id)] = center
    return moving_boxes, centers


def exclude_detection_regions(
    valid_mask: np.ndarray,
    boxes: list[tuple[int, int, int, int]] | tuple[tuple[int, int, int, int], ...],
    padding: int = 6,
) -> np.ndarray:
    """Return a copy of a boolean depth mask with detected object boxes removed.

    Boxes are (x, y, width, height) in image pixels. Padding expands each
    region to reduce dynamic-object boundary leakage into stereo VO and fusion.
    """
    mask = np.asarray(valid_mask)
    if mask.ndim != 2 or mask.size == 0 or mask.dtype != np.bool_:
        raise ValueError("valid_mask must be a non-empty 2D boolean array")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    output = mask.copy()
    height, width = output.shape
    for box in boxes:
        if not isinstance(box, (tuple, list)) or len(box) != 4:
            raise ValueError("each box must be an (x, y, width, height) tuple")
        x, y, box_width, box_height = box
        values = (x, y, box_width, box_height)
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError("box coordinates must be integers")
        if box_width <= 0 or box_height <= 0:
            raise ValueError("box width and height must be positive")
        x1 = max(0, int(x) - padding)
        y1 = max(0, int(y) - padding)
        x2 = min(width, int(x) + int(box_width) + padding)
        y2 = min(height, int(y) + int(box_height) + padding)
        if x1 < x2 and y1 < y2:
            output[y1:y2, x1:x2] = False
    return output


@dataclass(frozen=True)
class StereoVisualMotion:
    """Metric relative pose from previous camera coordinates to current coordinates."""
    rotation_matrix: list[list[float]]
    translation_vector: list[float]
    inlier_count: int
    match_count: int
    inlier_ratio: float
    metric_scale_known: bool = True


def estimate_stereo_visual_motion(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous_depth: np.ndarray,
    camera_matrix: np.ndarray | list[list[float]],
    min_inliers: int = 8,
    max_features: int = 3000,
    reprojection_error: float = 2.0,
    previous_valid_mask: np.ndarray | None = None,
) -> StereoVisualMotion:
    """Estimate metric frame-to-frame motion using previous-frame stereo depth.

    ORB matches are converted to 3D points with the previous frame's depth,
    then PnP-RANSAC estimates the transform mapping previous-camera 3D points
    into current-camera coordinates. Depth and translation use the same unit.
    """
    for name, frame in (("previous_frame", previous_frame), ("current_frame", current_frame)):
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.dtype != np.uint8:
            raise ValueError(f"{name} must be a non-empty uint8 image")
        if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[2] not in (3, 4)):
            raise ValueError(f"{name} must be grayscale, BGR, or BGRA")
    if previous_frame.shape[:2] != current_frame.shape[:2]:
        raise ValueError("both frames must have the same dimensions")
    depth = np.asarray(previous_depth, dtype=np.float64)
    if depth.shape != previous_frame.shape[:2]:
        raise ValueError("previous_depth must match the previous frame dimensions")
    if previous_valid_mask is not None:
        previous_valid_mask = np.asarray(previous_valid_mask)
        if previous_valid_mask.shape != depth.shape or previous_valid_mask.dtype != np.bool_:
            raise ValueError("previous_valid_mask must be boolean and match previous_depth")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera_matrix must be finite 3x3 with positive focal lengths")
    if isinstance(min_inliers, bool) or not isinstance(min_inliers, int) or min_inliers < 6:
        raise ValueError("min_inliers must be an integer >= 6")
    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < min_inliers:
        raise ValueError("max_features must be an integer >= min_inliers")
    if not np.isfinite(reprojection_error) or reprojection_error <= 0:
        raise ValueError("reprojection_error must be positive and finite")

    def grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)

    detector = cv2.ORB_create(nfeatures=max_features)
    feature_mask = None if previous_valid_mask is None else previous_valid_mask.astype(np.uint8) * 255
    key_a, desc_a = detector.detectAndCompute(grayscale(previous_frame), feature_mask)
    key_b, desc_b = detector.detectAndCompute(grayscale(current_frame), None)
    if desc_a is None or desc_b is None:
        raise ValueError("insufficient visual features for stereo visual odometry")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        raise ValueError("insufficient reliable feature matches for stereo visual odometry")

    object_points, image_points = [], []
    height, width = depth.shape
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    for match in good:
        u, v = key_a[match.queryIdx].pt
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < width and 0 <= y < height):
            continue
        z = depth[y, x]
        if not np.isfinite(z) or z <= 0:
            continue
        object_points.append([(u - cx) * z / fx, (v - cy) * z / fy, z])
        image_points.append(key_b[match.trainIdx].pt)
    if len(object_points) < min_inliers:
        raise ValueError("too few matched features have valid stereo depth")
    object_array = np.asarray(object_points, dtype=np.float32)
    image_array = np.asarray(image_points, dtype=np.float32)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_array, image_array, matrix, None,
        iterationsCount=200, reprojectionError=float(reprojection_error),
        confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    inlier_count = 0 if inliers is None else len(inliers)
    if not success or inlier_count < min_inliers:
        raise ValueError(f"stereo PnP failed or produced too few inliers ({inlier_count})")
    rotation, _ = cv2.Rodrigues(rvec)
    return StereoVisualMotion(
        rotation_matrix=rotation.tolist(),
        translation_vector=tvec.reshape(3).tolist(),
        inlier_count=int(inlier_count),
        match_count=len(object_points),
        inlier_ratio=float(inlier_count / len(object_points)),
    )


def compose_camera_to_world_pose(
    previous_rotation: np.ndarray | list[list[float]],
    previous_translation: np.ndarray | list[float],
    relative_rotation: np.ndarray | list[list[float]],
    relative_translation: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compose camera-to-world pose with a previous-to-current relative transform."""
    r_world_previous = np.asarray(previous_rotation, dtype=np.float64)
    t_world_previous = np.asarray(previous_translation, dtype=np.float64)
    r_current_previous = np.asarray(relative_rotation, dtype=np.float64)
    t_current_previous = np.asarray(relative_translation, dtype=np.float64)
    if (r_world_previous.shape != (3, 3) or r_current_previous.shape != (3, 3)
            or t_world_previous.shape != (3,) or t_current_previous.shape != (3,)
            or not all(np.isfinite(x).all() for x in
                       (r_world_previous, r_current_previous, t_world_previous, t_current_previous))):
        raise ValueError("poses must contain finite 3x3 rotations and 3D translations")
    r_world_current = r_world_previous @ r_current_previous.T
    t_world_current = t_world_previous - r_world_current @ t_current_previous
    return r_world_current, t_world_current


def fuse_point_clouds(
    point_clouds: list[np.ndarray],
    camera_to_world_poses: list[tuple[np.ndarray | list, np.ndarray | list]],
    voxel_size: float | None = None,
) -> np.ndarray:
    """Transform camera-local XYZRGB clouds into one world frame and optionally voxel-average."""
    if len(point_clouds) != len(camera_to_world_poses) or not point_clouds:
        raise ValueError("provide the same non-zero number of point clouds and poses")
    transformed = []
    for cloud, pose in zip(point_clouds, camera_to_world_poses):
        points = np.asarray(cloud, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 6 or not np.isfinite(points).all():
            raise ValueError("each point cloud must be a finite Nx6 XYZRGB array")
        rotation = np.asarray(pose[0], dtype=np.float64)
        translation = np.asarray(pose[1], dtype=np.float64)
        if rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise ValueError("each pose must contain a finite 3x3 rotation and 3D translation")
        xyz_world = (rotation @ points[:, :3].T).T + translation
        transformed.append(np.column_stack((xyz_world, points[:, 3:6])))
    merged = np.vstack(transformed)
    if voxel_size is None:
        return merged.astype(np.float32)
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    keys = np.floor(merged[:, :3] / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 6), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, merged)
    return (sums / counts[:, None]).astype(np.float32)
