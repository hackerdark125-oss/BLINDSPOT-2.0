import numpy as np
import pytest

from blindspot.motion import MotionFreshnessTracker


def test_first_frame_initializes_motion_and_age_maps():
    tracker = MotionFreshnessTracker(grid_size=4)
    result = tracker.update(np.zeros((120, 160), dtype=np.uint8), 2.0)
    assert result.magnitude_map.shape == (4, 4)
    assert np.all(result.magnitude_map == 0)
    assert np.all(result.activity_age_seconds == 0)


def test_static_frames_increase_activity_age():
    tracker = MotionFreshnessTracker(grid_size=4, motion_threshold=0.35)
    frame = np.zeros((120, 160), dtype=np.uint8)
    tracker.update(frame, 0.0)
    result = tracker.update(frame, 3.0)
    assert np.allclose(result.magnitude_map, 0)
    assert np.allclose(result.activity_age_seconds, 3.0)


def test_motion_resets_activity_age_in_moving_scene():
    tracker = MotionFreshnessTracker(grid_size=4, motion_threshold=0.05)
    first = np.zeros((160, 160), dtype=np.uint8)
    second = first.copy()
    cv2 = pytest.importorskip("cv2")
    cv2.rectangle(first, (15, 45), (55, 85), 255, -1)
    cv2.rectangle(second, (35, 45), (75, 85), 255, -1)
    tracker.update(first, 0.0)
    result = tracker.update(second, 2.0)
    assert result.mean_magnitude > 0
    assert np.any(result.magnitude_map > 0)


def test_timestamps_must_not_go_backwards():
    tracker = MotionFreshnessTracker()
    tracker.update(np.zeros((32, 32), dtype=np.uint8), 2.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        tracker.update(np.zeros((32, 32), dtype=np.uint8), 1.0)


def test_invalid_frame_rejected():
    tracker = MotionFreshnessTracker()
    with pytest.raises(ValueError):
        tracker.update(np.array([]), 0.0)
