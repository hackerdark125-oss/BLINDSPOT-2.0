import numpy as np
import pytest

from blindspot.registration import register_point_cloud_icp


def test_icp_recovers_small_translation_from_overlapping_cloud():
    rng = np.random.default_rng(12)
    target = rng.uniform(-1.0, 1.0, size=(500, 3))
    source = target - np.array([0.08, -0.04, 0.03])
    result = register_point_cloud_icp(
        source, target, max_correspondence_distance=0.3,
        voxel_size=0.005, trim_fraction=0.95, max_iterations=30,
    )
    assert np.allclose(result.rotation, np.eye(3), atol=1e-4)
    assert np.allclose(result.translation, [0.08, -0.04, 0.03], atol=2e-3)
    assert result.fitness > 0.9
    assert result.rmse < 0.01


def test_icp_accepts_xyzrgb_and_initial_transform():
    rng = np.random.default_rng(4)
    xyz = rng.normal(size=(120, 3))
    colors = np.full((120, 3), 128)
    target = np.column_stack((xyz, colors))
    source = target.copy()
    result = register_point_cloud_icp(
        source, target, initial_translation=[0.01, 0, 0],
        max_correspondence_distance=0.2, voxel_size=0.01,
    )
    assert np.allclose(result.translation, np.zeros(3), atol=1e-3)


def test_icp_rejects_invalid_cloud_and_parameters():
    with pytest.raises(ValueError):
        register_point_cloud_icp([[0, 0, 0]], np.zeros((4, 3)))
    with pytest.raises(ValueError):
        register_point_cloud_icp(np.zeros((4, 3)), np.zeros((4, 3)), trim_fraction=0)
