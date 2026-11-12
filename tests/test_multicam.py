import numpy as np
import pytest

from blindspot.multicam import fuse_multicamera_point_clouds


def test_multicamera_fusion_registers_camera_clouds_in_world_frame():
    cloud_a = np.array([[0, 0, 1, 10, 20, 30]], dtype=float)
    cloud_b = np.array([[0, 0, 1, 30, 40, 50]], dtype=float)
    poses = {
        "left": (np.eye(3), [0, 0, 0]),
        "right": (np.eye(3), [2, 0, 0]),
    }
    result = fuse_multicamera_point_clouds(
        {"left": cloud_a, "right": cloud_b}, poses
    )
    assert result.points.shape == (2, 6)
    assert np.allclose(result.points[:, :3], [[0, 0, 1], [2, 0, 1]])
    assert result.input_points == 2
    assert result.output_points == 2
    assert [item.camera_id for item in result.cameras] == ["left", "right"]


def test_multicamera_fusion_voxel_averages_across_cameras_by_point_count():
    cloud_a = np.array([[0.01, 0, 1, 10, 20, 30],
                        [0.02, 0, 1, 20, 30, 40]], dtype=float)
    cloud_b = np.array([[0.03, 0, 1, 40, 50, 60]], dtype=float)
    poses = {"a": (np.eye(3), [0, 0, 0]), "b": (np.eye(3), [0, 0, 0])}
    result = fuse_multicamera_point_clouds(
        {"a": cloud_a, "b": cloud_b}, poses, voxel_size=0.1
    )
    assert result.points.shape == (1, 6)
    assert np.allclose(result.points[0], [0.02, 0, 1, 70 / 3, 100 / 3, 130 / 3])
    assert result.output_points == 1
    assert result.voxel_size == 0.1


def test_multicamera_fusion_supports_empty_camera_cloud():
    result = fuse_multicamera_point_clouds(
        {"a": np.empty((0, 6)), "b": np.array([[1, 2, 3, 4, 5, 6]])},
        {"a": (np.eye(3), [0, 0, 0]), "b": (np.eye(3), [0, 0, 0])},
        voxel_size=0.25,
    )
    assert result.points.shape == (1, 6)
    assert result.input_points == 1


def test_multicamera_fusion_rejects_mismatched_ids_and_bad_rotation():
    cloud = np.ones((1, 6))
    with pytest.raises(ValueError, match="IDs must match"):
        fuse_multicamera_point_clouds({"left": cloud}, {"right": (np.eye(3), [0, 0, 0])})
    invalid_rotation = np.diag([2.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="orthonormal"):
        fuse_multicamera_point_clouds(
            {"left": cloud}, {"left": (invalid_rotation, [0, 0, 0])}
        )


def test_multicamera_fusion_rejects_invalid_cloud_pose_and_voxel_size():
    cloud = np.ones((1, 6))
    pose = {"cam": (np.eye(3), [0, 0, 0])}
    with pytest.raises(ValueError, match="Nx6"):
        fuse_multicamera_point_clouds({"cam": np.ones((1, 5))}, pose)
    with pytest.raises(ValueError, match="translation"):
        fuse_multicamera_point_clouds({"cam": cloud}, {"cam": (np.eye(3), [0, 0])})
    with pytest.raises(ValueError, match="voxel_size"):
        fuse_multicamera_point_clouds({"cam": cloud}, pose, voxel_size=0)
