"""Independent residual metrics for validating SE(3) pose-graph constraints."""
from __future__ import annotations

from math import acos, degrees
from typing import Sequence

import numpy as np


def evaluate_pose_graph_constraints(
    camera_to_world_poses: Sequence[tuple[object, object]],
    constraints: Sequence[tuple[int, int, object, object] | tuple[int, int, object, object, float]],
) -> dict[str, object]:
    """Measure how well camera-to-world poses satisfy relative i-to-j constraints.

    Each constraint is (i, j, R_ij, t_ij[, weight]) and follows
    X_j = R_ij @ X_i + t_ij. Rotation error is reported in degrees and
    translation error in the same unit as pose translations.
    """
    if not isinstance(camera_to_world_poses, (list, tuple)) or len(camera_to_world_poses) < 2:
        raise ValueError("camera_to_world_poses must contain at least two poses")
    rotations: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    for index, pose in enumerate(camera_to_world_poses):
        if not isinstance(pose, (list, tuple)) or len(pose) != 2:
            raise ValueError(f"pose {index} must be (rotation, translation)")
        rotation = np.asarray(pose[0], dtype=np.float64)
        translation = np.asarray(pose[1], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError(f"pose {index} rotation must be finite 3x3")
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
            raise ValueError(f"pose {index} rotation must be a proper rotation")
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError(f"pose {index} translation must be a finite 3-vector")
        rotations.append(rotation)
        translations.append(translation)

    if not constraints:
        raise ValueError("at least one relative-pose constraint is required")
    rows = []
    weighted_translation_squared = 0.0
    weighted_rotation_squared = 0.0
    total_weight = 0.0
    for edge_index, edge in enumerate(constraints):
        if not isinstance(edge, (list, tuple)) or len(edge) not in (4, 5):
            raise ValueError(f"constraint {edge_index} must contain 4 or 5 values")
        i, j, measured_rotation, measured_translation = edge[:4]
        weight = float(edge[4]) if len(edge) == 5 else 1.0
        if (isinstance(i, bool) or not isinstance(i, (int, np.integer))
                or isinstance(j, bool) or not isinstance(j, (int, np.integer))
                or not 0 <= i < j < len(rotations)):
            raise ValueError(f"constraint {edge_index} indices must satisfy 0 <= i < j < pose count")
        measured_rotation = np.asarray(measured_rotation, dtype=np.float64)
        measured_translation = np.asarray(measured_translation, dtype=np.float64)
        if measured_rotation.shape != (3, 3) or not np.isfinite(measured_rotation).all():
            raise ValueError(f"constraint {edge_index} rotation must be finite 3x3")
        if (not np.allclose(measured_rotation.T @ measured_rotation, np.eye(3), atol=1e-3)
                or not np.isclose(np.linalg.det(measured_rotation), 1.0, atol=1e-3)):
            raise ValueError(f"constraint {edge_index} rotation must be a proper rotation")
        if measured_translation.shape != (3,) or not np.isfinite(measured_translation).all():
            raise ValueError(f"constraint {edge_index} translation must be a finite 3-vector")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError(f"constraint {edge_index} weight must be positive and finite")

        predicted_rotation = rotations[j].T @ rotations[i]
        predicted_translation = rotations[j].T @ (translations[i] - translations[j])
        rotation_delta = measured_rotation.T @ predicted_rotation
        cosine = float(np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0))
        rotation_error = degrees(acos(cosine))
        translation_vector = predicted_translation - measured_translation
        translation_error = float(np.linalg.norm(translation_vector))
        rows.append({
            "i": int(i), "j": int(j), "weight": weight,
            "rotation_error_degrees": rotation_error,
            "translation_error": translation_error,
            "translation_error_vector": translation_vector.tolist(),
        })
        weighted_rotation_squared += weight * rotation_error ** 2
        weighted_translation_squared += weight * translation_error ** 2
        total_weight += weight

    return {
        "constraints_evaluated": len(rows),
        "edges": rows,
        "weighted_rotation_rmse_degrees": float(np.sqrt(weighted_rotation_squared / total_weight)),
        "weighted_translation_rmse": float(np.sqrt(weighted_translation_squared / total_weight)),
        "translation_unit": "same as input pose translations",
        "interpretation": (
            "Constraint consistency only; low residuals do not prove the measurements "
            "are accurate or independently verified."
        ),
    }
