import numpy as np
import pytest
from blindspot.people import CentroidTracker, Detection, HOGPeopleDetector

def person(x, y=20):
    return Detection(x, y, 30, 80, 1.0)

def test_tracker_keeps_id_for_nearby_detection():
    tracker = CentroidTracker(max_distance=50)
    first = tracker.update([person(10)])[0]
    assert tracker.update([person(18)])[0].track_id == first.track_id

def test_tracker_registers_new_person_when_far_away():
    tracker = CentroidTracker(max_distance=20)
    first = tracker.update([person(0)])[0]
    assert tracker.update([person(200)])[0].track_id != first.track_id

def test_tracker_expires_disappeared_track():
    tracker = CentroidTracker(max_disappeared=1)
    first = tracker.update([person(10)])[0]
    tracker.update([])
    tracker.update([])
    assert tracker.update([person(10)])[0].track_id != first.track_id

def test_tracker_handles_multiple_people_and_order_changes():
    tracker = CentroidTracker(max_distance=60)
    initial = tracker.update([person(0), person(200)])
    updated = tracker.update([person(205), person(5)])
    assert {x.track_id for x in initial} == {x.track_id for x in updated}

@pytest.mark.parametrize("kwargs", [
    {"max_distance": 0}, {"max_disappeared": -1}, {"max_disappeared": True},
])
def test_tracker_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        CentroidTracker(**kwargs)

def test_tracker_rejects_invalid_detections():
    with pytest.raises(ValueError):
        CentroidTracker().update([object()])

def test_hog_detector_validates_frame_format():
    detector = HOGPeopleDetector()
    with pytest.raises(ValueError, match="uint8"):
        detector.detect(np.zeros((160, 80, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="three-channel"):
        detector.detect(np.zeros((160, 80), dtype=np.uint8))

def test_hog_detector_handles_too_small_frame():
    assert HOGPeopleDetector().detect(np.zeros((40, 40, 3), dtype=np.uint8)) == []
