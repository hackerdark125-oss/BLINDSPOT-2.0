"""Planar ground-footprint coverage from calibrated camera intrinsics and poses.

This is a 2D plane-intersection diagnostic. It is not a 3D reconstruction,
visual-localization system, or guarantee that image pixels contain useful data.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from typing import Any

import cv2
import numpy as np

from blindspot.calibration import CameraCalibration
from blindspot.stereo import StereoCalibration


@dataclass(frozen=True)
class CameraPose:
    """Camera-to-world rigid transform: X_world = R @ X_camera + t."""
    rotation_matrix: list[list[float]]
    translation_vector: list[float]

    def to_json(self) -> str:
        return json.dumps({
            "rotation_matrix": self.rotation_matrix,
            "translation_vector": self.translation_vector,
        }, indent=2)


@dataclass(frozen=True)
class PlanarCoverageResult:
    polygon_a: np.ndarray
    polygon_b: np.ndarray
    coverage_count_map: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    overlap_area: float
    union_area: float
    camera_a_area: float
    camera_b_area: float
    plane_z: float


def load_camera_pose_json(content: str) -> CameraPose:
    """Load and validate a camera-to-world rigid transform."""
    try:
        data = json.loads(content)
        rotation = np.asarray(data["rotation_matrix"], dtype=np.float64)
        translation = np.asarray(data["translation_vector"], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid camera pose JSON") from exc
    if (
        rotation.shape != (3, 3) or translation.size != 3
        or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    ):
        raise ValueError("camera pose must contain a finite 3x3 rotation and 3D translation")
    return CameraPose(rotation.tolist(), translation.tolist())


def _pose_arrays(pose: CameraPose) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(pose.rotation_matrix, dtype=np.float64)
    translation = np.asarray(pose.translation_vector, dtype=np.float64).reshape(-1)
    if (
        rotation.shape != (3, 3) or translation.size != 3
        or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)
    ):
        raise ValueError("camera pose contains an invalid rigid transform")
    return rotation, translation


def camera_footprint_on_plane(
    intrinsics: CameraCalibration,
    pose: CameraPose,
    plane_z: float = 0.0,
) -> np.ndarray:
    """Intersect the four image-corner rays with a horizontal world plane."""
    if not isfinite(plane_z):
        raise ValueError("plane_z must be finite")
    matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    if (
        matrix.shape != (3, 3) or not np.isfinite(matrix).all()
        or abs(float(np.linalg.det(matrix))) < 1e-12
        or intrinsics.image_width < 2 or intrinsics.image_height < 2
    ):
        raise ValueError("camera intrinsics are invalid")
    rotation, translation = _pose_arrays(pose)
    inverse = np.linalg.inv(matrix)
    pixels = np.asarray([
        [0.0, 0.0, 1.0],
        [intrinsics.image_width - 1.0, 0.0, 1.0],
        [intrinsics.image_width - 1.0, intrinsics.image_height - 1.0, 1.0],
        [0.0, intrinsics.image_height - 1.0, 1.0],
    ], dtype=np.float64)
    rays_camera = (inverse @ pixels.T).T
    rays_world = (rotation @ rays_camera.T).T
    if np.any(np.abs(rays_world[:, 2]) < 1e-9):
        raise ValueError("one or more image-corner rays are parallel to the coverage plane")
    distances = (plane_z - translation[2]) / rays_world[:, 2]
    if np.any(distances <= 0) or not np.isfinite(distances).all():
        raise ValueError("coverage plane must be in front of all image-corner rays")
    points = translation[None, :] + distances[:, None] * rays_world
    return points[:, :2].astype(np.float32)


def _polygon_area(polygon: np.ndarray) -> float:
    return abs(float(cv2.contourArea(polygon.reshape(-1, 1, 2))))


def estimate_stereo_planar_coverage(
    intrinsics_a: CameraCalibration,
    intrinsics_b: CameraCalibration,
    stereo: StereoCalibration,
    camera_a_pose: CameraPose,
    plane_z: float = 0.0,
    grid_size: int = 160,
) -> PlanarCoverageResult:
    """Estimate two camera footprints and their overlap on a known world plane.

    OpenCV stereo calibration convention is X_b = R_ab @ X_a + T_ab.
    The relative transform is inverted and composed with camera A's world pose.
    All pose, translation, and plane units must be consistent.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or not 32 <= grid_size <= 512:
        raise ValueError("grid_size must be an integer between 32 and 512")
    if (intrinsics_a.image_width, intrinsics_a.image_height) != (stereo.image_width, stereo.image_height):
        raise ValueError("camera A intrinsics do not match stereo calibration dimensions")
    if (intrinsics_b.image_width, intrinsics_b.image_height) != (stereo.image_width, stereo.image_height):
        raise ValueError("camera B intrinsics do not match stereo calibration dimensions")
    rotation_ab = np.asarray(stereo.rotation_matrix, dtype=np.float64)
    translation_ab = np.asarray(stereo.translation_vector, dtype=np.float64).reshape(-1)
    if (
        rotation_ab.shape != (3, 3) or translation_ab.size != 3
        or not np.isfinite(rotation_ab).all() or not np.isfinite(translation_ab).all()
    ):
        raise ValueError("stereo relative pose is invalid")
    rotation_a, translation_a = _pose_arrays(camera_a_pose)
    rotation_b = rotation_a @ rotation_ab.T
    translation_b = translation_a - rotation_b @ translation_ab
    pose_b = CameraPose(rotation_b.tolist(), translation_b.tolist())

    polygon_a = camera_footprint_on_plane(intrinsics_a, camera_a_pose, plane_z)
    polygon_b = camera_footprint_on_plane(intrinsics_b, pose_b, plane_z)
    area_a = _polygon_area(polygon_a)
    area_b = _polygon_area(polygon_b)
    overlap, _ = cv2.intersectConvexConvex(polygon_a, polygon_b)
    overlap_area = max(0.0, float(overlap))
    union_area = max(0.0, area_a + area_b - overlap_area)

    all_points = np.vstack((polygon_a, polygon_b))
    x_min, y_min = all_points.min(axis=0).astype(float)
    x_max, y_max = all_points.max(axis=0).astype(float)
    if x_max - x_min <= 1e-9 or y_max - y_min <= 1e-9:
        raise ValueError("camera footprints have a degenerate ground-plane extent")
    count_map = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for polygon in (polygon_a, polygon_b):
        points = np.empty_like(polygon, dtype=np.int32)
        points[:, 0] = np.round((polygon[:, 0] - x_min) / (x_max - x_min) * (grid_size - 1)).astype(np.int32)
        points[:, 1] = np.round((y_max - polygon[:, 1]) / (y_max - y_min) * (grid_size - 1)).astype(np.int32)
        mask = np.zeros_like(count_map)
        cv2.fillPoly(mask, [points], 1)
        count_map += mask

    return PlanarCoverageResult(
        polygon_a, polygon_b, count_map,
        x_min, x_max, y_min, y_max,
        overlap_area, union_area, area_a, area_b, float(plane_z),
    )
