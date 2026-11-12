import numpy as np
import pytest

from blindspot.pose_graph import optimize_pose_graph_translations


def test_pose_graph_without_loop_closures_preserves_odometry():
    poses = [[0, 0, 0], [1, 0, 0], [2, 1, 0]]
    result = optimize_pose_graph_translations(poses, [])
    assert np.allclose(result, poses)


def test_pose_graph_distributes_translation_drift_with_loop_closure():
    poses = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    corrected = np.asarray(optimize_pose_graph_translations(
        poses, [(0, 3, [2.7, 0, 0], 100.0)]
    ))
    assert np.allclose(corrected[0], poses[0])
    assert corrected[3, 0] == pytest.approx(2.7, abs=0.01)
    assert corrected[1, 0] < 1.0
    assert corrected[2, 0] < 2.0
    assert np.all(np.diff(corrected[:, 0]) > 0)


def test_pose_graph_rejects_invalid_trajectory_and_constraints():
    with pytest.raises(ValueError, match="at least two"):
        optimize_pose_graph_translations([[0, 0, 0]], [])
    with pytest.raises(ValueError, match="3D positions"):
        optimize_pose_graph_translations([[0, 0], [1, 1]], [])
    with pytest.raises(ValueError, match="indices"):
        optimize_pose_graph_translations(
            [[0, 0, 0], [1, 0, 0]], [(1, 0, [1, 0, 0], 1.0)]
        )
    with pytest.raises(ValueError, match="weight"):
        optimize_pose_graph_translations(
            [[0, 0, 0], [1, 0, 0]], [(0, 1, [1, 0, 0], 0)]
        )
