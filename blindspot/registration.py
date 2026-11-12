"""Coarse-to-fine point-to-point ICP for rigid XYZ point-cloud registration.

The implementation uses deterministic voxel sampling and bounded-memory
nearest-neighbour search. ICP is local: callers should provide a reasonable
initial alignment and inspect fitness/RMSE before trusting the transform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ICPRegistrationResult:
    rotation: np.ndarray
    translation: np.ndarray
    fitness: float
    rmse: float
    correspondences: int
    iterations: int
    converged: bool


def _xyz(points: object, name: str) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] not in (3, 6) or len(cloud) < 3:
        raise ValueError(f"{name} must contain at least three XYZ or XYZRGB points")
    cloud = cloud[:, :3]
    if not np.isfinite(cloud).all():
        raise ValueError(f"{name} must contain finite coordinates")
    return cloud


def _voxel_sample(points: np.ndarray, voxel_size: float, limit: int) -> np.ndarray:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    sampled = points[np.sort(first)]
    if len(sampled) > limit:
        indices = np.linspace(0, len(sampled) - 1, limit, dtype=np.int64)
        sampled = sampled[indices]
    return sampled


def _nearest(source: np.ndarray, target: np.ndarray, chunk_size: int = 256):
    indices = np.empty(len(source), dtype=np.int64)
    distances = np.empty(len(source), dtype=np.float64)
    target_norm = np.sum(target * target, axis=1)
    for start in range(0, len(source), chunk_size):
        chunk = source[start:start + chunk_size]
        squared = (np.sum(chunk * chunk, axis=1)[:, None]
                   + target_norm[None, :] - 2.0 * chunk @ target.T)
        nearest = np.argmin(squared, axis=1)
        indices[start:start + len(chunk)] = nearest
        distances[start:start + len(chunk)] = np.sqrt(
            np.maximum(squared[np.arange(len(chunk)), nearest], 0.0)
        )
    return indices, distances


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def register_point_cloud_icp(
    source_points: object,
    target_points: object,
    initial_rotation: object | None = None,
    initial_translation: object | None = None,
    max_correspondence_distance: float = 0.25,
    voxel_size: float = 0.03,
    trim_fraction: float = 0.8,
    max_iterations: int = 40,
    tolerance: float = 1e-6,
    max_points: int = 4000,
) -> ICPRegistrationResult:
    """Estimate the rigid transform mapping source XYZ into target XYZ.

    Distances and voxel_size use the point clouds' shared metric unit.
    A supplied initial transform maps source coordinates into target coordinates.
    Correspondences are distance-gated and trimmed to reduce outlier influence.
    """
    source = _xyz(source_points, "source_points")
    target = _xyz(target_points, "target_points")
    if not np.isfinite(max_correspondence_distance) or max_correspondence_distance <= 0:
        raise ValueError("max_correspondence_distance must be positive and finite")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")
    if not 0 < trim_fraction <= 1 or not np.isfinite(trim_fraction):
        raise ValueError("trim_fraction must be in (0, 1]")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 3:
        raise ValueError("max_points must be an integer >= 3")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite")

    rotation = np.eye(3) if initial_rotation is None else np.asarray(initial_rotation, dtype=np.float64)
    translation = np.zeros(3) if initial_translation is None else np.asarray(initial_translation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("initial_rotation must be a proper finite rotation matrix")
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("initial_translation must be a finite 3-vector")

    source = _voxel_sample(source, voxel_size, max_points)
    target = _voxel_sample(target, voxel_size, max_points)
    previous_rmse = np.inf
    converged = False
    correspondences = 0
    rmse = np.inf

    for iteration in range(1, max_iterations + 1):
        transformed = (rotation @ source.T).T + translation
        nearest, distances = _nearest(transformed, target)
        valid = distances <= max_correspondence_distance
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 3:
            raise ValueError("ICP found fewer than three valid correspondences; improve initial alignment")
        keep_count = max(3, int(np.floor(len(valid_indices) * trim_fraction)))
        selected = valid_indices[np.argsort(distances[valid_indices])[:keep_count]]
        matched_source = transformed[selected]
        matched_target = target[nearest[selected]]
        delta_rotation, delta_translation = _rigid_fit(matched_source, matched_target)
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation

        updated = (rotation @ source.T).T + translation
        updated_nearest, updated_distances = _nearest(updated, target)
        updated_valid = updated_distances <= max_correspondence_distance
        updated_indices = np.flatnonzero(updated_valid)
        if len(updated_indices) < 3:
            raise ValueError("ICP refinement lost valid correspondences")
        updated_keep = max(3, int(np.floor(len(updated_indices) * trim_fraction)))
        chosen = updated_indices[np.argsort(updated_distances[updated_indices])[:updated_keep]]
        correspondences = len(chosen)
        rmse = float(np.sqrt(np.mean(updated_distances[chosen] ** 2)))
        if abs(previous_rmse - rmse) < tolerance:
            converged = True
            break
        previous_rmse = rmse

    fitness = float(correspondences / len(source))
    return ICPRegistrationResult(
        rotation=rotation, translation=translation, fitness=fitness, rmse=rmse,
        correspondences=correspondences, iterations=iteration, converged=converged,
    )
