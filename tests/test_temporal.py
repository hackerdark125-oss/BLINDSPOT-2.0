import numpy as np
import pytest
from blindspot.temporal import frame_change


def test_identical_frames_have_zero_change():
    image = np.full((32, 32, 3), 90, dtype=np.uint8)
    assert frame_change(image, image) == pytest.approx(0.0)


def test_different_frames_have_positive_change():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    b = np.full((32, 32, 3), 255, dtype=np.uint8)
    assert frame_change(a, b) == pytest.approx(100.0)


def test_empty_frame_rejected():
    with pytest.raises(ValueError):
        frame_change(np.array([]), np.zeros((4, 4), dtype=np.uint8))
