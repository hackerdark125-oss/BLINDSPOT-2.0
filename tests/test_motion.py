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


def test_scene_change_resets_motion_history():
    tracker = MotionFreshnessTracker(grid_size=4, camera_motion_compensation=False)
    dark = np.zeros((160, 160), dtype=np.uint8)
    bright = np.full((160, 160), 255, dtype=np.uint8)
    tracker.update(dark, 0.0)
    result = tracker.update(bright, 1.0)
    assert result.scene_changed
    assert not result.camera_motion_compensated
    assert np.all(result.magnitude_map == 0)
    assert np.all(result.activity_age_seconds == 0)
    later = tracker.update(bright, 3.0)
    assert not later.scene_changed
    assert np.allclose(later.activity_age_seconds, 2.0)


def test_camera_motion_compensation_reports_success_on_textured_translation():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, size=(240, 240), dtype=np.uint8)
    second = cv2.warpAffine(
        first, np.float32([[1, 0, 6], [0, 1, 3]]), (240, 240),
        borderMode=cv2.BORDER_REFLECT,
    )
    tracker = MotionFreshnessTracker(grid_size=4, motion_threshold=0.1)
    tracker.update(first, 0.0)
    result = tracker.update(second, 1.0)
    assert result.camera_motion_compensated


def test_camera_motion_compensation_can_be_disabled():
    tracker = MotionFreshnessTracker(grid_size=4, camera_motion_compensation=False)
    frame = np.zeros((120, 120), dtype=np.uint8)
    tracker.update(frame, 0.0)
    result = tracker.update(frame, 1.0)
    assert not result.camera_motion_compensated
