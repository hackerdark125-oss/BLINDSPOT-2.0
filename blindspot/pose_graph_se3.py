"""Full SE(3) pose-graph optimisation for short stereo-odometry trajectories.

Poses are camera-to-world transforms. Relative measurements map coordinates
from pose i into pose j. Pose zero is fixed to remove gauge freedom.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseGraphSE3Result:
    poses: tuple[tuple[np.ndarray, np.ndarray], ...]
    iterations: int
    initial_cost: float
    final_cost: float
    converged: bool


def _rotation(value: object, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-3):
        raise ValueError(f"{label} rotation must have determinant +1")
    return matrix


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    vector, _ = cv2.Rodrigues(rotation)
    return vector.reshape(3).astype(np.float64)


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(vector, dtype=np.float64).reshape(3, 1))
    return rotation


def optimize_pose_graph_se3(
    camera_to_world_poses: Sequence[tuple[object, object]],
    loop_closures: Sequence[tuple[int, int, object, object, float]] = (),
    odometry_weight: float = 1.0,
    max_iterations: int = 30,
    huber_delta: float = 1.0,
    tolerance: float = 1e-7,
) -> PoseGraphSE3Result:
    """Jointly optimise rotations and translations using relative SE(3) edges.

    Each loop closure is (i, j, measured_rotation_i_to_j,
    measured_translation_i_to_j, weight). Measurements must be metric and
    independently verified. Consecutive relative measurements are derived
    from the supplied initial trajectory and retained as odometry edges.

    A damped Gauss-Newton solver with central-difference Jacobians and a
    Huber robust kernel is used. This dense solver is intended for short
    trajectories; it is not a large-scale SLAM backend.
    """
    if not isinstance(camera_to_world_poses, (list, tuple)) or len(camera_to_world_poses) < 2:
        raise ValueError("camera_to_world_poses must contain at least two poses")
    if (not np.isfinite(odometry_weight) or odometry_weight <= 0
            or not np.isfinite(huber_delta) or huber_delta <= 0
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("weights, huber_delta, and tolerance must be positive and finite")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    rotations, translations = [], []
    for index, pose in enumerate(camera_to_world_poses):
        if not isinstance(pose, (tuple, list)) or len(pose) != 2:
            raise ValueError(f"pose {index} must be (rotation, translation)")
        rotations.append(_rotation(pose[0], f"pose {index}"))
        translation = np.asarray(pose[1], dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"pose {index} translation must be a finite 3-vector")
        translations.append(translation)

    count = len(rotations)
    # Edge measurements map coordinates in i into coordinates in j.
    edges: list[tuple[int, int, np.ndarray, np.ndarray, float]] = []
    for i in range(count - 1):
        relative_rotation = rotations[i + 1].T @ rotations[i]
        relative_translation = rotations[i + 1].T @ (translations[i] - translations[i + 1])
        edges.append((i, i + 1, relative_rotation, relative_translation, float(odometry_weight)))

    for edge in loop_closures:
        if not isinstance(edge, (tuple, list)) or len(edge) != 5:
            raise ValueError("each loop closure must be (i, j, rotation, translation, weight)")
        i, j, measured_rotation, measured_translation, weight = edge
        if (isinstance(i, bool) or not isinstance(i, (int, np.integer))
                or isinstance(j, bool) or not isinstance(j, (int, np.integer))
                or not 0 <= i < j < count):
            raise ValueError("loop closure indices must satisfy 0 <= i < j < pose count")
        measured_rotation = _rotation(measured_rotation, "loop closure")
        measured_translation = np.asarray(measured_translation, dtype=np.float64)
        if measured_translation.shape != (3,) or not np.isfinite(measured_translation).all():
            raise ValueError("loop closure translation must be a finite 3-vector")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("loop closure weight must be positive and finite")
        edges.append((int(i), int(j), measured_rotation, measured_translation, float(weight)))

    initial_rotations = list(rotations)
    initial_translations = list(translations)
    parameters = np.concatenate([
        np.concatenate((_so3_log(rotations[i]), translations[i]))
        for i in range(1, count)
    ])

    def unpack(values: np.ndarray):
        rs = [initial_rotations[0]]
        ts = [initial_translations[0]]
        for index in range(count - 1):
            chunk = values[index * 6:(index + 1) * 6]
            rs.append(_so3_exp(chunk[:3]))
            ts.append(chunk[3:6])
        return rs, ts

    def residual(values: np.ndarray) -> np.ndarray:
        rs, ts = unpack(values)
        rows = []
        for i, j, measured_r, measured_t, weight in edges:
            predicted_r = rs[j].T @ rs[i]
            predicted_t = rs[j].T @ (ts[i] - ts[j])
            rotation_error = _so3_log(measured_r.T @ predicted_r)
            translation_error = predicted_t - measured_t
            rows.append(np.concatenate((rotation_error, translation_error)) * np.sqrt(weight))
        return np.concatenate(rows)

    def robust_cost(vector: np.ndarray) -> float:
        blocks = vector.reshape(-1, 6)
        norms = np.linalg.norm(blocks, axis=1)
        delta = float(huber_delta)
        return float(np.sum(np.where(norms <= delta, 0.5 * norms ** 2,
                                     delta * (norms - 0.5 * delta))))

    current = parameters.copy()
    raw = residual(current)
    initial_cost = robust_cost(raw)
    current_cost = initial_cost
    damping = 1e-4
    converged = False
    iterations = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        raw = residual(current)
        blocks = raw.reshape(-1, 6)
        norms = np.linalg.norm(blocks, axis=1)
        robust_weights = np.ones_like(norms)
        outliers = norms > huber_delta
        robust_weights[outliers] = huber_delta / np.maximum(norms[outliers], 1e-12)
        row_weights = np.repeat(np.sqrt(robust_weights), 6)
        weighted_residual = raw * row_weights

        jacobian = np.empty((len(raw), len(current)), dtype=np.float64)
        step = 1e-6
        for column in range(len(current)):
            offset = np.zeros_like(current)
            offset[column] = step
            jacobian[:, column] = (residual(current + offset) - residual(current - offset)) / (2 * step)
        jacobian *= row_weights[:, None]

        hessian = jacobian.T @ jacobian
        gradient = jacobian.T @ weighted_residual
        try:
            update = np.linalg.solve(hessian + damping * np.eye(len(current)), -gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.lstsq(hessian + damping * np.eye(len(current)), -gradient, rcond=None)[0]
        if not np.isfinite(update).all():
            raise RuntimeError("SE(3) pose-graph solver produced a non-finite update")
        if np.linalg.norm(update) < tolerance:
            converged = True
            break

        accepted = False
        scale = 1.0
        for _ in range(12):
            candidate = current + scale * update
            candidate_cost = robust_cost(residual(candidate))
            if candidate_cost < current_cost:
                current, current_cost = candidate, candidate_cost
                damping = max(damping * 0.3, 1e-10)
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            damping *= 10.0
            if damping > 1e8:
                break
        elif abs(candidate_cost - current_cost) < tolerance:
            converged = True
            break

    final_rotations, final_translations = unpack(current)
    result_poses = tuple((r.copy(), t.copy()) for r, t in zip(final_rotations, final_translations))
    return PoseGraphSE3Result(
        poses=result_poses, iterations=iterations, initial_cost=initial_cost,
        final_cost=robust_cost(residual(current)), converged=converged,
    )
