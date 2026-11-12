import numpy as np
import pytest

from blindspot.tracking import FrameFreezeDetector
from blindspot.live import parse_camera_source


def test_freeze_detector_flags_repeated_frames():
    detector = FrameFreezeDetector(threshold=0.002, consecutive_samples=3)
    frame = np.full((80, 100, 3), 120, dtype=np.uint8)
    assert detector.update(frame)[1] is False
    assert detector.update(frame)[1] is False
    assert detector.update(frame)[1] is True


def test_freeze_detector_resets_after_change():
    detector = FrameFreezeDetector(threshold=0.002, consecutive_samples=2)
    still = np.zeros((80, 100), dtype=np.uint8)
    changed = np.full((80, 100), 255, dtype=np.uint8)
    detector.update(still)
    detector.update(still)
    assert detector.update(changed)[1] is False


def test_parse_camera_source():
    assert parse_camera_source("0") == 0
    assert parse_camera_source(1) == 1
    assert parse_camera_source("rtsp://camera.local/live") == "rtsp://camera.local/live"


@pytest.mark.parametrize("source", ["", "-1", "camera.local", True])
def test_parse_camera_source_rejects_invalid_values(source):
    with pytest.raises(ValueError):
        parse_camera_source(source)
