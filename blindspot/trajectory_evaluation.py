"""Metric trajectory evaluation against independently measured ground truth."""
from __future__ import annotations

from math import sqrt
from typing import Sequence

import numpy as np


def _poses(values: object, name: str) -> np.ndarray:
    poses = np.asarray(values, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"{name} must be an Nx4x4 pose array with N >= 2")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} must contain finite values")
    if not np.allclose(poses[:, 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(f"{name} contains invalid homogeneous bottom rows")
    rotations = poses[:, :3, :3]
    if (not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3)):
        raise ValueError(f"{name} rotations must be proper orthonormal matrices")
    return poses


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cosine))


def evaluate_trajectory(
    estimated_poses: Sequence[object],
    ground_truth_poses: Sequence[object],
) -> dict[str, float | int | str]:
    """Compute first-pose-aligned ATE and consecutive-frame relative pose error.

    No scale fitting is performed: estimated and reference translations must
    use the same metric unit. Pose matrices are camera-to-world transforms.
    """
    estimated = _poses(estimated_poses, "estimated_poses")
    ground_truth = _poses(ground_truth_poses, "ground_truth_poses")
    if len(estimated) != len(ground_truth):
        raise ValueError("estimated and ground-truth trajectories must have equal lengths")

    # Rigidly align the first estimated camera pose to the first reference pose.
    alignment_rotation = ground_truth[0, :3, :3] @ estimated[0, :3, :3].T
    alignment_translation = (
        ground_truth[0, :3, 3] - alignment_rotation @ estimated[0, :3, 3]
    )
    aligned = estimated.copy()
    aligned[:, :3, :3] = alignment_rotation @ estimated[:, :3, :3]
    aligned[:, :3, 3] = (
        alignment_rotation @ estimated[:, :3, 3].T
    ).T + alignment_translation

    position_errors = np.linalg.norm(
        aligned[:, :3, 3] - ground_truth[:, :3, 3], axis=1
    )
    relative_translation_errors = []
    relative_rotation_errors = []
    for index in range(len(estimated) - 1):
        estimated_relative = np.linalg.inv(estimated[index]) @ estimated[index + 1]
        reference_relative = np.linalg.inv(ground_truth[index]) @ ground_truth[index + 1]
        error = np.linalg.inv(reference_relative) @ estimated_relative
        relative_translation_errors.append(float(np.linalg.norm(error[:3, 3])))
        relative_rotation_errors.append(_rotation_angle(error[:3, :3]))

    return {
        "poses_evaluated": int(len(estimated)),
        "ate_rmse": float(sqrt(np.mean(position_errors ** 2))),
        "ate_mean": float(np.mean(position_errors)),
        "ate_median": float(np.median(position_errors)),
        "ate_max": float(np.max(position_errors)),
        "rpe_translation_rmse": float(sqrt(np.mean(np.square(relative_translation_errors)))),
        "rpe_rotation_rmse_degrees": float(
            np.degrees(sqrt(np.mean(np.square(relative_rotation_errors))))
        ),
        "translation_unit": "same as input pose translations",
    }
