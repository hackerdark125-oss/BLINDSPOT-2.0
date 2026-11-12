import numpy as np
import pytest

from blindspot.reconstruction import depth_to_point_cloud, write_ply, estimate_visual_motion


def test_depth_backprojection_and_stride():
    depth = np.full((4, 4), 10.0)
    k = np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]])
    cloud = depth_to_point_cloud(depth, k, stride=2)
    assert cloud.shape == (4, 6)
    assert np.allclose(cloud[0, :3], [-5.0, -5.0, 10.0])


def test_depth_point_cloud_filters_invalid_and_far_pixels():
    depth = np.array([[1.0, np.nan], [4.0, 8.0]])
    cloud = depth_to_point_cloud(depth, np.eye(3), max_depth=5, stride=1)
    assert cloud.shape == (2, 6)


def test_depth_point_cloud_world_transform_and_colors():
    depth = np.ones((2, 2), dtype=float)
    colors = np.zeros((2, 2, 3), dtype=np.uint8)
    colors[0, 0] = [10, 20, 30]
    cloud = depth_to_point_cloud(
        depth, np.eye(3), colors, stride=1,
        world_translation=[1, 2, 3],
    )
    assert np.allclose(cloud[0, :3], [1, 2, 4])
    assert np.allclose(cloud[0, 3:], [30, 20, 10])


def test_point_cloud_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="camera_matrix"):
        depth_to_point_cloud(np.ones((2, 2)), np.eye(2))
    with pytest.raises(ValueError, match="stride"):
        depth_to_point_cloud(np.ones((2, 2)), np.eye(3), stride=0)


def test_write_ply(tmp_path):
    points = np.array([[1, 2, 3, 255, 0, 10]], dtype=float)
    path = write_ply(tmp_path / "nested" / "scene.ply", points)
    content = path.read_text()
    assert "element vertex 1" in content
    assert "1.000000 2.000000 3.000000 255 0 10" in content


def test_visual_motion_rejects_blank_frames():
    blank = np.zeros((80, 80), dtype=np.uint8)
    with pytest.raises(ValueError, match="insufficient visual features"):
        estimate_visual_motion(blank, blank, np.eye(3))
