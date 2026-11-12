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


from blindspot.reconstruction import (
    compose_camera_to_world_pose, fuse_point_clouds, estimate_stereo_visual_motion,
)


def test_compose_camera_pose_identity_relative_motion():
    rotation, translation = compose_camera_to_world_pose(
        np.eye(3), [0, 0, 0], np.eye(3), [1, 0, 0]
    )
    assert np.allclose(rotation, np.eye(3))
    assert np.allclose(translation, [-1, 0, 0])


def test_fuse_point_clouds_transforms_and_voxel_averages():
    cloud_a = np.array([[0, 0, 1, 10, 20, 30]], dtype=float)
    cloud_b = np.array([[0.01, 0, 1, 30, 40, 50]], dtype=float)
    poses = [(np.eye(3), np.zeros(3)), (np.eye(3), np.zeros(3))]
    fused = fuse_point_clouds([cloud_a, cloud_b], poses, voxel_size=0.1)
    assert fused.shape == (1, 6)
    assert np.allclose(fused[0], [0.005, 0, 1, 20, 30, 40])


def test_fuse_point_clouds_rejects_mismatched_pose_count():
    with pytest.raises(ValueError, match="same non-zero"):
        fuse_point_clouds([np.zeros((1, 6))], [])


def test_stereo_visual_odometry_rejects_depth_shape_mismatch():
    frame = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="match the previous frame"):
        estimate_stereo_visual_motion(frame, frame, np.ones((10, 10)), np.eye(3))


def test_exclude_detection_regions_masks_boxes_with_padding_and_clipping():
    from blindspot.reconstruction import exclude_detection_regions

    valid = np.ones((10, 12), dtype=bool)
    masked = exclude_detection_regions(valid, [(2, 3, 4, 3)], padding=1)
    assert masked.sum() == valid.sum() - (6 * 5)
    assert not masked[2:7, 1:7].any()
    assert valid.all()  # input mask is not mutated


def test_exclude_detection_regions_clips_boxes_at_image_edges():
    from blindspot.reconstruction import exclude_detection_regions

    valid = np.ones((5, 5), dtype=bool)
    masked = exclude_detection_regions(valid, [(-3, -2, 5, 4)], padding=1)
    assert not masked[:3, :3].any()
    assert masked[4, 4]


def test_exclude_detection_regions_validates_inputs():
    from blindspot.reconstruction import exclude_detection_regions

    with pytest.raises(ValueError, match="boolean"):
        exclude_detection_regions(np.ones((4, 4), dtype=np.uint8), [])
    with pytest.raises(ValueError, match="positive"):
        exclude_detection_regions(np.ones((4, 4), dtype=bool), [(1, 1, 0, 2)])
    with pytest.raises(ValueError, match="padding"):
        exclude_detection_regions(np.ones((4, 4), dtype=bool), [], padding=-1)



def test_stereo_visual_odometry_validates_previous_feature_mask():
    frame = np.zeros((20, 20), dtype=np.uint8)
    with pytest.raises(ValueError, match="previous_valid_mask"):
        estimate_stereo_visual_motion(
            frame, frame, np.ones((20, 20)), np.eye(3),
            previous_valid_mask=np.ones((10, 10), dtype=bool),
        )
    with pytest.raises(ValueError, match="previous_valid_mask"):
        estimate_stereo_visual_motion(
            frame, frame, np.ones((20, 20)), np.eye(3),
            previous_valid_mask=np.ones((20, 20), dtype=np.uint8),
        )
