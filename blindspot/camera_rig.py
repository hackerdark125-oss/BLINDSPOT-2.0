"""Compose pairwise stereo extrinsics into a shared multi-camera rig frame."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CameraRigResult:
    anchor_camera_id: str
    camera_poses: dict[str, tuple[np.ndarray, np.ndarray]]
    connected_cameras: tuple[str, ...]
    pairwise_edges: int
    mean_rotation_cycle_error_degrees: float
    mean_translation_cycle_error: float


def _rotation(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-3)):
        raise ValueError("pairwise rotation must be a proper finite 3x3 rotation")
    return matrix


def estimate_camera_rig_poses(
    camera_ids: Sequence[str],
    pairwise_extrinsics: Sequence[tuple[str, str, object, object]],
    anchor_camera_id: str,
) -> CameraRigResult:
    """Estimate camera-to-world poses by composing calibrated pairwise edges.

    Each edge is (camera_i, camera_j, R_ji, t_ji) with the convention
    X_j = R_ji @ X_i + t_ji. Translation units must be identical for every
    edge. The anchor camera defines the world frame with identity pose.

    A breadth-first spanning tree initialises all connected cameras. Every
    supplied edge is then checked against the resulting rig for cycle residuals.
    This composes pairwise stereo extrinsics; it does not detect chessboards,
    synchronise capture, or refine all edges in a global nonlinear adjustment.
    """
    if not isinstance(camera_ids, (list, tuple)) or len(camera_ids) < 2:
        raise ValueError("camera_ids must contain at least two cameras")
    ids = list(camera_ids)
    if any(not isinstance(item, str) or not item.strip() for item in ids):
        raise ValueError("camera IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("camera IDs must be unique")
    if anchor_camera_id not in ids:
        raise ValueError("anchor_camera_id must be one of camera_ids")

    adjacency: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {item: [] for item in ids}
    checked_edges = []
    seen_pairs = set()
    for edge in pairwise_extrinsics:
        if not isinstance(edge, (tuple, list)) or len(edge) != 4:
            raise ValueError("each edge must be (camera_i, camera_j, rotation_i_to_j, translation_i_to_j)")
        camera_i, camera_j, raw_rotation, raw_translation = edge
        if camera_i not in adjacency or camera_j not in adjacency or camera_i == camera_j:
            raise ValueError("pairwise edge must reference two distinct known cameras")
        pair = (camera_i, camera_j)
        if pair in seen_pairs or (camera_j, camera_i) in seen_pairs:
            raise ValueError("duplicate pairwise camera edge")
        seen_pairs.add(pair)
        rotation = _rotation(raw_rotation)
        translation = np.asarray(raw_translation, dtype=np.float64)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("pairwise translation must be a finite 3-vector")
        # Forward maps i coordinates to j coordinates. The camera-j world pose
        # follows from X_world = R_wi X_i + t_wi = R_wj X_j + t_wj.
        adjacency[camera_i].append((camera_j, rotation, translation))
        adjacency[camera_j].append((camera_i, rotation.T, -rotation.T @ translation))
        checked_edges.append((camera_i, camera_j, rotation, translation))

    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {
        anchor_camera_id: (np.eye(3), np.zeros(3))
    }
    queue = deque([anchor_camera_id])
    while queue:
        current = queue.popleft()
        rotation_world_current, translation_world_current = poses[current]
        for neighbour, rotation_neighbour_current, translation_neighbour_current in adjacency[current]:
            if neighbour in poses:
                continue
            rotation_world_neighbour = rotation_world_current @ rotation_neighbour_current.T
            translation_world_neighbour = (
                translation_world_current
                - rotation_world_neighbour @ translation_neighbour_current
            )
            poses[neighbour] = (rotation_world_neighbour, translation_world_neighbour)
            queue.append(neighbour)

    if len(poses) != len(ids):
        missing = sorted(set(ids) - set(poses))
        raise ValueError(f"camera extrinsic graph is disconnected; no path to: {', '.join(missing)}")

    rotation_errors, translation_errors = [], []
    for camera_i, camera_j, measured_rotation, measured_translation in checked_edges:
        rotation_i, translation_i = poses[camera_i]
        rotation_j, translation_j = poses[camera_j]
        predicted_rotation = rotation_j.T @ rotation_i
        predicted_translation = rotation_j.T @ (translation_i - translation_j)
        delta_rotation = measured_rotation.T @ predicted_rotation
        angle = np.arccos(np.clip((np.trace(delta_rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(float(np.degrees(angle)))
        translation_errors.append(float(np.linalg.norm(predicted_translation - measured_translation)))

    return CameraRigResult(
        anchor_camera_id=anchor_camera_id,
        camera_poses=poses,
        connected_cameras=tuple(item for item in ids if item in poses),
        pairwise_edges=len(checked_edges),
        mean_rotation_cycle_error_degrees=float(np.mean(rotation_errors)) if rotation_errors else 0.0,
        mean_translation_cycle_error=float(np.mean(translation_errors)) if translation_errors else 0.0,
    )
