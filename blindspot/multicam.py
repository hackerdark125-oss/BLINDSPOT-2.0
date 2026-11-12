"""Multi-camera point-cloud registration and memory-aware voxel fusion.

All camera-to-world poses must already be expressed in one shared world frame.
This module does not estimate inter-camera extrinsics or synchronise sensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CameraFusionStats:
    camera_id: str
    input_points: int
    valid_points: int


@dataclass(frozen=True)
class MultiCameraFusionResult:
    points: np.ndarray
    cameras: tuple[CameraFusionStats, ...]
    input_points: int
    output_points: int
    voxel_size: float | None


def _validate_rotation(rotation: object, camera_id: str) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"camera {camera_id!r} rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
        raise ValueError(f"camera {camera_id!r} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-3):
        raise ValueError(f"camera {camera_id!r} rotation must have determinant +1")
    return matrix


def fuse_multicamera_point_clouds(
    camera_point_clouds: Mapping[str, np.ndarray],
    camera_to_world_poses: Mapping[str, tuple[object, object]],
    voxel_size: float | None = None,
) -> MultiCameraFusionResult:
    """Register named camera-local XYZRGB clouds and fuse them in one world frame.

    Point colours are RGB values in [0, 255]. Each pose is a
    (camera_to_world_rotation, camera_to_world_translation) pair. Translation
    and point coordinates must use the same distance unit for every camera.

    With voxel_size set, each camera is voxel-aggregated before cross-camera
    aggregation, reducing peak memory compared with stacking all raw points.
    Each input point has equal weight; output XYZ and RGB values are arithmetic
    means within each occupied world-frame voxel.
    """
    if not isinstance(camera_point_clouds, Mapping) or not camera_point_clouds:
        raise ValueError("camera_point_clouds must be a non-empty mapping")
    if not isinstance(camera_to_world_poses, Mapping):
        raise ValueError("camera_to_world_poses must be a mapping")
    cloud_ids = set(camera_point_clouds)
    pose_ids = set(camera_to_world_poses)
    if cloud_ids != pose_ids:
        raise ValueError("camera cloud IDs and pose IDs must match exactly")
    if any(not isinstance(camera_id, str) or not camera_id.strip() for camera_id in cloud_ids):
        raise ValueError("camera IDs must be non-empty strings")
    if voxel_size is not None and (not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("voxel_size must be positive and finite")

    camera_stats: list[CameraFusionStats] = []
    transformed_clouds: list[np.ndarray] = []
    voxel_keys: list[np.ndarray] = []
    voxel_sums: list[np.ndarray] = []
    voxel_counts: list[np.ndarray] = []
    total_input = 0

    for camera_id in sorted(cloud_ids):
        cloud = np.asarray(camera_point_clouds[camera_id], dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 6 or not np.isfinite(cloud).all():
            raise ValueError(f"camera {camera_id!r} cloud must be a finite Nx6 XYZRGB array")
        pose = camera_to_world_poses[camera_id]
        if not isinstance(pose, (tuple, list)) or len(pose) != 2:
            raise ValueError(f"camera {camera_id!r} pose must be (rotation, translation)")
        rotation = _validate_rotation(pose[0], camera_id)
        translation = np.asarray(pose[1], dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"camera {camera_id!r} translation must be a finite 3-vector")

        total_input += len(cloud)
        camera_stats.append(CameraFusionStats(camera_id, len(cloud), len(cloud)))
        world = cloud.copy()
        if len(world):
            world[:, :3] = (rotation @ world[:, :3].T).T + translation

        if voxel_size is None:
            transformed_clouds.append(world)
            continue

        if not len(world):
            continue
        keys = np.floor(world[:, :3] / float(voxel_size)).astype(np.int64)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        sums = np.zeros((len(unique_keys), 6), dtype=np.float64)
        counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.int64)
        np.add.at(sums, inverse, world)
        voxel_keys.append(unique_keys)
        voxel_sums.append(sums)
        voxel_counts.append(counts)

    if voxel_size is None:
        fused = np.vstack(transformed_clouds).astype(np.float32)
    elif not voxel_keys:
        fused = np.empty((0, 6), dtype=np.float32)
    else:
        # Merge per-camera voxel accumulators. Repeated voxel keys across cameras
        # are weighted by their original point counts, not by camera count.
        all_keys = np.vstack(voxel_keys)
        all_sums = np.vstack(voxel_sums)
        all_counts = np.concatenate(voxel_counts)
        merged_keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)
        merged_sums = np.zeros((len(merged_keys), 6), dtype=np.float64)
        merged_counts = np.zeros(len(merged_keys), dtype=np.int64)
        np.add.at(merged_sums, inverse, all_sums)
        np.add.at(merged_counts, inverse, all_counts)
        fused = (merged_sums / merged_counts[:, None]).astype(np.float32)

    return MultiCameraFusionResult(
        points=fused,
        cameras=tuple(camera_stats),
        input_points=total_input,
        output_points=len(fused),
        voxel_size=None if voxel_size is None else float(voxel_size),
    )
