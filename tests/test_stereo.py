import numpy as np
import pytest

from blindspot.calibration import CameraCalibration
from blindspot.stereo import calibrate_stereo_cameras


def make_intrinsics():
    return CameraCalibration(
        camera_matrix=[[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]],
        distortion_coefficients=[0, 0, 0, 0, 0],
        rms_reprojection_error=0.2,
        image_width=160,
        image_height=120,
        board_columns=9,
        board_rows=6,
        square_size=1.0,
        accepted_views=5,
        rejected_views=0,
    )


def test_stereo_requires_matching_pair_counts():
    with pytest.raises(ValueError, match="same number"):
        calibrate_stereo_cameras(
            [np.zeros((120, 160), dtype=np.uint8)],
            [],
            make_intrinsics(),
            make_intrinsics(),
        )


def test_stereo_rejects_mismatched_intrinsic_resolutions():
    other = make_intrinsics()
    changed = CameraCalibration(
        **{**other.__dict__, "image_width": 320}
    )
    with pytest.raises(ValueError, match="same image resolution"):
        calibrate_stereo_cameras(
            [np.zeros((120, 160), dtype=np.uint8)] * 3,
            [np.zeros((120, 160), dtype=np.uint8)] * 3,
            make_intrinsics(),
            changed,
        )


def test_stereo_rejects_wrong_board_configuration():
    intrinsics = make_intrinsics()
    with pytest.raises(ValueError, match="different chessboard size"):
        calibrate_stereo_cameras(
            [np.zeros((120, 160), dtype=np.uint8)] * 3,
            [np.zeros((120, 160), dtype=np.uint8)] * 3,
            intrinsics,
            intrinsics,
            board_size=(8, 6),
        )


def test_stereo_requires_detectable_paired_boards():
    images = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match="Need at least"):
        calibrate_stereo_cameras(
            images, images, make_intrinsics(), make_intrinsics()
        )

def make_stereo():
    from blindspot.stereo import StereoCalibration

    return StereoCalibration(
        rotation_matrix=np.eye(3).tolist(),
        translation_vector=[-10.0, 0.0, 0.0],
        essential_matrix=[[0, 0, 0], [0, 0, -10], [0, 10, 0]],
        fundamental_matrix=[[0, 0, 0], [0, 0, -0.1], [0, 0.1, 0]],
        rms_reprojection_error=0.3,
        image_width=160,
        image_height=120,
        accepted_pairs=5,
        rejected_pairs=0,
        translation_unit="cm",
    )


def test_stereo_calibration_json_round_trip():
    from blindspot.stereo import load_stereo_calibration_json

    original = make_stereo()
    loaded = load_stereo_calibration_json(original.to_json())
    assert loaded.translation_vector == original.translation_vector
    assert loaded.translation_unit == "cm"


def test_stereo_depth_rejects_wrong_image_resolution():
    from blindspot.stereo import rectify_and_compute_depth

    image = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="match the stereo calibration resolution"):
        rectify_and_compute_depth(
            image, image, make_intrinsics(), make_intrinsics(), make_stereo()
        )


def test_stereo_depth_rejects_invalid_disparity_range():
    from blindspot.stereo import rectify_and_compute_depth

    image = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="multiple of 16"):
        rectify_and_compute_depth(
            image, image, make_intrinsics(), make_intrinsics(), make_stereo(),
            num_disparities=15,
        )

def test_stereo_json_rejects_zero_baseline():
    from blindspot.stereo import load_stereo_calibration_json

    data = make_stereo().__dict__.copy()
    data["translation_vector"] = [0.0, 0.0, 0.0]
    import json
    with pytest.raises(ValueError, match="invalid parameters"):
        load_stereo_calibration_json(json.dumps(data))
