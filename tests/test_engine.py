import numpy as np
import pytest

from blindspot.engine import analyse_frame, prioritise_cells


def test_analysis_returns_bounded_grid():
    result = analyse_frame(np.full((240, 320, 3), 120, dtype=np.uint8), 8)
    assert result.confidence_map.shape == (8, 8)
    assert np.all((result.confidence_map >= 0) & (result.confidence_map <= 100))
    assert result.metrics.uncertainty == pytest.approx(100-result.metrics.confidence)


def test_textured_image_scores_above_blank():
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    textured = np.random.default_rng(4).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    assert analyse_frame(textured).metrics.confidence > analyse_frame(blank).metrics.confidence


def test_prioritise_lowest_first():
    result = prioritise_cells(np.array([[80, 10], [50, 20]], dtype=np.float32), 2)
    assert [(x["row"], x["column"]) for x in result] == [(0, 1), (1, 1)]


def test_invalid_frame_rejected():
    with pytest.raises(ValueError):
        analyse_frame(np.array([]))


@pytest.mark.parametrize("grid_size", [0, -1, 1.5, True])
def test_invalid_grid_size_rejected(grid_size):
    with pytest.raises(ValueError, match="grid_size"):
        analyse_frame(np.zeros((16, 16), dtype=np.uint8), grid_size)


def test_non_uint8_frame_rejected():
    with pytest.raises(ValueError, match="uint8"):
        analyse_frame(np.zeros((16, 16), dtype=np.float32))


def test_invalid_confidence_map_rejected():
    with pytest.raises(ValueError):
        prioritise_cells(np.array([[1.0, np.nan]]))


def test_non_integer_limit_rejected():
    with pytest.raises(ValueError, match="limit"):
        prioritise_cells(np.ones((2, 2)), 1.5)
