"""Regression coverage for geometric registration and SE(3) optimisation."""

import numpy as np
import pytest

from blindspot.pose_graph_se3 import optimize_pose_graph_se3
from blindspot.registration import register_point_cloud_icp


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_icp_recovers_small_rigid_transform_from_nearby_clouds():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(180, 3)) * 0.3
    expected_rotation = _rotation_z(0.08)
    expected_translation = np.array([0.12, -0.07, 0.04])
    target = (expected_rotation @ source.T).T + expected_translation

    result = register_point_cloud_icp(
        source, target,
        max_correspondence_distance=0.8,
        voxel_size=0.002,
        trim_fraction=1.0,
        max_iterations=60,
        tolerance=1e-10,
        max_points=1000,
    )

    assert result.fitness > 0.95
    assert result.rmse < 1e-5
    assert np.allclose(result.rotation, expected_rotation, atol=1e-4)
    assert np.allclose(result.translation, expected_translation, atol=1e-4)


def test_icp_accepts_xyzrgb_and_ignores_color_columns_for_alignment():
    rng = np.random.default_rng(7)
    xyz = rng.normal(size=(80, 3))
    source = np.column_stack([xyz, np.zeros((len(xyz), 3))])
    target = np.column_stack([xyz + [0.02, 0.0, 0.0], np.full((len(xyz), 3), 255)])

    result = register_point_cloud_icp(
        source, target, max_correspondence_distance=0.2,
        voxel_size=0.001, trim_fraction=1.0, max_iterations=10,
    )

    assert np.allclose(result.translation, [0.02, 0.0, 0.0], atol=1e-6)
    assert result.rmse < 1e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_correspondence_distance": 0},
        {"voxel_size": -1},
        {"trim_fraction": 0},
        {"max_iterations": 0},
        {"max_points": 2},
        {"tolerance": 0},
    ],
)
def test_icp_rejects_invalid_configuration(kwargs):
    cloud = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    with pytest.raises(ValueError):
        register_point_cloud_icp(cloud, cloud, **kwargs)


def test_icp_rejects_insufficient_correspondences():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    target = source + 100.0
    with pytest.raises(ValueError, match="fewer than three"):
        register_point_cloud_icp(
            source, target, max_correspondence_distance=0.01,
            voxel_size=0.001,
        )


def test_se3_pose_graph_without_loop_closure_preserves_input_trajectory():
    poses = [
        (np.eye(3), np.array([float(i), 0.0, 0.0]))
        for i in range(4)
    ]
    result = optimize_pose_graph_se3(poses)

    assert result.final_cost <= result.initial_cost + 1e-12
    assert np.allclose(result.poses[0][0], poses[0][0])
    assert np.allclose(result.poses[0][1], poses[0][1])
    for actual, expected in zip(result.poses, poses):
        assert np.allclose(actual[0], expected[0], atol=1e-6)
        assert np.allclose(actual[1], expected[1], atol=1e-6)


def test_se3_loop_constraint_reduces_translation_closure_error_and_anchors_first_pose():
    poses = [
        (np.eye(3), np.array([0.0, 0.0, 0.0])),
        (np.eye(3), np.array([1.0, 0.0, 0.0])),
        (np.eye(3), np.array([2.0, 0.0, 0.0])),
        (np.eye(3), np.array([3.6, 0.0, 0.0])),
    ]
    # Relative camera-coordinate transform from pose 0 to pose 3.
    closure = (0, 3, np.eye(3), np.array([-3.0, 0.0, 0.0]), 8.0)
    result = optimize_pose_graph_se3(poses, loop_closures=[closure])

    initial_error = np.linalg.norm(poses[0][1] - poses[3][1] - [-3.0, 0.0, 0.0])
    final_error = np.linalg.norm(
        result.poses[0][1] - result.poses[3][1] - [-3.0, 0.0, 0.0]
    )
    assert result.final_cost < result.initial_cost
    assert final_error < initial_error
    assert np.allclose(result.poses[0][1], poses[0][1])


def test_se3_pose_graph_rejects_invalid_rotation_and_loop_edge():
    poses = [
        (np.eye(3), np.zeros(3)),
        (np.eye(3), np.array([1.0, 0.0, 0.0])),
    ]
    invalid_rotation = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="determinant"):
        optimize_pose_graph_se3([(invalid_rotation, np.zeros(3)), poses[1]])

    with pytest.raises(ValueError, match="indices"):
        optimize_pose_graph_se3(
            poses, loop_closures=[(1, 0, np.eye(3), np.zeros(3), 1.0)]
        )
