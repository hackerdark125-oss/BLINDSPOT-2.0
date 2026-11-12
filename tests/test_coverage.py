import json

import numpy as np
import pytest

from blindspot.calibration import CameraCalibration
from blindspot.coverage import (
    CameraPose,
    camera_footprint_on_plane,
    estimate_stereo_planar_coverage,
    load_camera_pose_json,
)
from blindspot.stereo import StereoCalibration


def intrinsics():
    return CameraCalibration(
        camera_matrix=[[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]],
        distortion_coefficients=[0, 0, 0, 0, 0],
        rms_reprojection_error=0.1,
        image_width=100,
        image_height=100,
        board_columns=9,
        board_rows=6,
        square_size=1.0,
        accepted_views=5,
        rejected_views=0,
    )


def stereo():
    return StereoCalibration(
        rotation_matrix=np.eye(3).tolist(),
        translation_vector=[-2.0, 0.0, 0.0],
        essential_matrix=[[0, 0, 0], [0, 0, -2], [0, 2, 0]],
        fundamental_matrix=[[0, 0, 0], [0, 0, -0.02], [0, 0.02, 0]],
        rms_reprojection_error=0.1,
        image_width=100,
        image_height=100,
        accepted_pairs=5,
        rejected_pairs=0,
        translation_unit="m",
    )


def downward_pose():
    return CameraPose(
        rotation_matrix=[[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        translation_vector=[0, 0, 10],
    )


def test_camera_footprint_intersects_ground_plane():
    polygon = camera_footprint_on_plane(intrinsics(), downward_pose(), plane_z=0)
    assert polygon.shape == (4, 2)
    assert np.isclose(abs(cv2_area(polygon)), 98.01, atol=0.1)


def cv2_area(polygon):
    import cv2
    return cv2.contourArea(polygon)


def test_stereo_coverage_reports_expected_overlap_and_union():
    result = estimate_stereo_planar_coverage(
        intrinsics(), intrinsics(), stereo(), downward_pose(), grid_size=64
    )
    assert result.camera_a_area == pytest.approx(98.01, abs=0.1)
    assert result.camera_b_area == pytest.approx(100.0, abs=0.1)
    assert result.overlap_area == pytest.approx(78.21, abs=0.1)
    assert result.union_area == pytest.approx(117.81, abs=0.1)
    assert result.coverage_count_map.shape == (64, 64)
    assert set(np.unique(result.coverage_count_map)).issubset({0, 1, 2})


def test_camera_pose_json_validates_rotation_and_translation():
    pose = downward_pose()
    loaded = load_camera_pose_json(pose.to_json())
    assert loaded.translation_vector == pose.translation_vector
    bad = {"rotation_matrix": np.zeros((3, 3)).tolist(), "translation_vector": [0, 0, 1]}
    with pytest.raises(ValueError, match="finite 3x3 rotation"):
        load_camera_pose_json(json.dumps(bad))


def test_coverage_rejects_camera_looking_away_from_plane():
    upward = CameraPose(np.eye(3).tolist(), [0, 0, 10])
    with pytest.raises(ValueError, match="in front"):
        camera_footprint_on_plane(intrinsics(), upward)


def test_coverage_rejects_invalid_grid_size():
    with pytest.raises(ValueError, match="grid_size"):
        estimate_stereo_planar_coverage(
            intrinsics(), intrinsics(), stereo(), downward_pose(), grid_size=8
        )
