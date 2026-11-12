import json

import numpy as np
import pytest

from blindspot.calibration import calibrate_camera, load_calibration_json


def test_calibration_requires_multiple_views():
    with pytest.raises(ValueError, match="at least three"):
        calibrate_camera([])


def test_calibration_rejects_invalid_board_size():
    with pytest.raises(ValueError, match="board_size"):
        calibrate_camera([np.zeros((100, 100), dtype=np.uint8)] * 3, board_size=(1, 6))


def test_calibration_rejects_nonpositive_square_size():
    with pytest.raises(ValueError, match="square_size"):
        calibrate_camera([np.zeros((100, 100), dtype=np.uint8)] * 3, square_size=0)


def test_calibration_rejects_insufficient_chessboard_views():
    images = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError, match="Need at least"):
        calibrate_camera(images)


def test_load_calibration_json_rejects_bad_content():
    with pytest.raises(ValueError, match="invalid calibration JSON"):
        load_calibration_json("{not json")


def test_load_calibration_json_rejects_invalid_matrix():
    data = {
        "camera_matrix": [[1, 2], [3, 4]],
        "distortion_coefficients": [0, 0, 0, 0],
        "rms_reprojection_error": 0.1,
        "image_width": 100,
        "image_height": 100,
        "board_columns": 9,
        "board_rows": 6,
        "square_size": 1.0,
        "accepted_views": 3,
        "rejected_views": 0,
    }
    with pytest.raises(ValueError, match="invalid camera parameters"):
        load_calibration_json(json.dumps(data))
