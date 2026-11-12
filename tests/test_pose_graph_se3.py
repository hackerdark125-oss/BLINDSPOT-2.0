import numpy as np
import pytest

from blindspot.pose_graph_se3 import optimize_pose_graph_se3


def _pose(x, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), np.array([x, 0.0, 0.0])


def test_se3_pose_graph_preserves_anchor_and_optimizes_rotation_and_translation():
    poses = [_pose(0.0), _pose(1.1, 0.08), _pose(2.2, 0.16), _pose(3.3, 0.24)]
    # A verified revisit measurement closes the path to the origin.
    result = optimize_pose_graph_se3(
        poses,
        loop_closures=[(0, 3, np.eye(3), np.array([-3.0, 0.0, 0.0]), 20.0)],
    )
    assert np.allclose(result.poses[0][0], np.eye(3))
    assert np.allclose(result.poses[0][1], np.zeros(3))
    assert result.final_cost < result.initial_cost
    assert result.iterations > 1
    assert not np.allclose(result.poses[1][0], poses[1][0])
    assert result.poses[1][1][0] < poses[1][1][0]


def test_se3_without_loop_keeps_relative_odometry():
    poses = [_pose(0.0), _pose(1.0, 0.1), _pose(2.0, 0.2)]
    result = optimize_pose_graph_se3(poses)
    for actual, original in zip(result.poses, poses):
        assert np.allclose(actual[0], original[0], atol=1e-5)
        assert np.allclose(actual[1], original[1], atol=1e-5)


@pytest.mark.parametrize("bad", [np.zeros((3, 3)), np.full((3, 3), np.nan)])
def test_se3_rejects_invalid_rotation(bad):
    with pytest.raises(ValueError):
        optimize_pose_graph_se3([_pose(0), (bad, np.zeros(3))])


def test_se3_rejects_bad_loop_constraint():
    with pytest.raises(ValueError):
        optimize_pose_graph_se3([_pose(0), _pose(1)], [(0, 1, np.eye(3), [0, 0], 1.0)])
