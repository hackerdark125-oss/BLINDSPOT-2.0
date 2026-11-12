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

def _synthetic_chessboard_views():
    import cv2

    square = 40
    board = np.full((7 * square, 10 * square), 255, dtype=np.uint8)
    for row in range(7):
        for col in range(10):
            if (row + col) % 2 == 0:
                board[row*square:(row+1)*square, col*square:(col+1)*square] = 0
    source = np.float32([[0, 0], [399, 0], [399, 279], [0, 279]])
    destinations = [
        [[85, 65], [500, 75], [485, 370], [75, 355]],
        [[110, 80], [520, 55], [505, 355], [95, 370]],
        [[70, 95], [470, 60], [515, 345], [105, 365]],
        [[100, 55], [490, 90], [510, 375], [65, 340]],
        [[125, 85], [515, 100], [480, 360], [90, 350]],
    ]
    views = []
    for points in destinations:
        transform = cv2.getPerspectiveTransform(source, np.float32(points))
        views.append(cv2.warpPerspective(
            board, transform, (640, 480), borderValue=255
        ))
    return views


def test_calibrate_camera_with_synthetic_chessboard_views():
    result = calibrate_camera(
        _synthetic_chessboard_views(),
        board_size=(9, 6),
        square_size=24.0,
        min_views=3,
    )
    assert result.accepted_views >= 3
    assert result.image_width == 640
    assert result.image_height == 480
    assert np.asarray(result.camera_matrix).shape == (3, 3)
    assert np.isfinite(result.rms_reprojection_error)


