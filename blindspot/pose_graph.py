"""Lightweight translation-only pose-graph optimisation for stereo odometry.

This module does not detect loop closures. Callers must supply independently
verified loop-closure constraints in the same world-coordinate distance unit
as the odometry trajectory. Rotations are intentionally left unchanged.
"""
from __future__ import annotations

import numpy as np


def optimize_pose_graph_translations(
    translations: list[list[float] | tuple[float, float, float] | np.ndarray],
    loop_closures: list[tuple[int, int, list[float] | tuple[float, float, float], float]]
    | tuple[tuple[int, int, list[float] | tuple[float, float, float], float], ...],
    odometry_weight: float = 1.0,
) -> list[list[float]]:
    """Distribute translation drift across a trajectory using loop constraints.

    Each loop closure is (i, j, measured_delta, weight), where measured_delta
    is the independently estimated world-frame displacement from pose i to j.
    Consecutive odometry deltas are retained as weighted constraints. The first
    pose is fixed to remove gauge freedom.

    This is a small dense least-squares solver intended for short trajectories,
    not a replacement for full SE(3) bundle adjustment or a loop detector.
    """
    if not isinstance(translations, (list, tuple)) or len(translations) < 2:
        raise ValueError("translations must contain at least two 3D positions")
    positions = np.asarray(translations, dtype=np.float64)
    if positions.shape != (len(translations), 3) or not np.isfinite(positions).all():
        raise ValueError("translations must be finite 3D positions")
    if not np.isfinite(odometry_weight) or odometry_weight <= 0:
        raise ValueError("odometry_weight must be positive and finite")

    count = len(positions)
    constraints: list[tuple[int, int, np.ndarray, float]] = []
    for index in range(count - 1):
        constraints.append((
            index, index + 1,
            positions[index + 1] - positions[index],
            float(odometry_weight),
        ))

    for closure in loop_closures:
        if not isinstance(closure, (tuple, list)) or len(closure) != 4:
            raise ValueError("each loop closure must be (i, j, measured_delta, weight)")
        i, j, delta, weight = closure
        if (isinstance(i, bool) or not isinstance(i, (int, np.integer))
                or isinstance(j, bool) or not isinstance(j, (int, np.integer))
                or not (0 <= i < j < count)):
            raise ValueError("loop closure indices must satisfy 0 <= i < j < pose count")
        vector = np.asarray(delta, dtype=np.float64)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("loop closure displacement must be a finite 3-vector")
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("loop closure weight must be positive and finite")
        constraints.append((int(i), int(j), vector, float(weight)))

    # Pose zero is anchored. Solve for the remaining 3D positions.
    rows = []
    targets = []
    for i, j, delta, weight in constraints:
        row = np.zeros(count - 1, dtype=np.float64)
        if i > 0:
            row[i - 1] -= 1.0
        if j > 0:
            row[j - 1] += 1.0
        scale = np.sqrt(weight)
        rows.append(row * scale)
        targets.append((delta + positions[i] - positions[j]) * scale)

    design = np.vstack(rows)
    target = np.vstack(targets)
    corrected = positions.copy()
    corrected[1:] += np.linalg.lstsq(design, target, rcond=None)[0]
    return corrected.tolist()
