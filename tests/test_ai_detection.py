from types import SimpleNamespace

import numpy as np
import pytest

from blindspot.ai_detection import YOLODetector


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[5, 8, 70, 100], [-10, 5, 20, 30]])
        self.cls = FakeTensor([0, 2])
        self.conf = FakeTensor([0.91, 0.77])
        self.id = FakeTensor([11, 12])

    def __len__(self):
        return 2


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def track(self, **kwargs):
        self.kwargs = kwargs
        return [SimpleNamespace(boxes=FakeBoxes(), names={0: "person", 2: "car"})]


def test_yolo_detector_maps_boxes_and_track_ids():
    model = FakeModel()
    detector = YOLODetector(model=model)
    frame = np.zeros((120, 100, 3), dtype=np.uint8)
    detections = detector.track_frame(frame, classes=[0, 2], confidence=0.4)
    assert len(detections) == 2
    assert detections[0].class_name == "person"
    assert detections[0].track_id == 11
    assert detections[0].width == 65
    assert detections[1].x == 0
    assert model.kwargs["persist"] is True
    assert model.kwargs["tracker"] == "bytetrack.yaml"
    assert model.kwargs["classes"] == [0, 2]


def test_yolo_detector_handles_missing_track_ids():
    model = FakeModel()
    result = SimpleNamespace(boxes=FakeBoxes(), names={0: "person", 2: "car"})
    result.boxes.id = None
    model.track = lambda **kwargs: [result]
    detections = YOLODetector(model=model).track_frame(
        np.zeros((120, 100, 3), dtype=np.uint8)
    )
    assert all(item.track_id is None for item in detections)


@pytest.mark.parametrize("frame", [
    np.zeros((0, 0, 3), dtype=np.uint8),
    np.zeros((100, 100), dtype=np.uint8),
    np.zeros((100, 100, 3), dtype=np.float32),
])
def test_yolo_detector_rejects_invalid_frames(frame):
    with pytest.raises(ValueError):
        YOLODetector(model=FakeModel()).track_frame(frame)


@pytest.mark.parametrize("kwargs", [
    {"confidence": -0.1}, {"confidence": 1.1}, {"image_size": 10},
    {"classes": [-1]},
])
def test_yolo_detector_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        YOLODetector(model=FakeModel()).track_frame(
            np.zeros((120, 100, 3), dtype=np.uint8), **kwargs
        )


def test_yolo_detector_handles_empty_result():
    model = FakeModel()
    model.track = lambda **kwargs: [SimpleNamespace(boxes=None, names={})]
    assert YOLODetector(model=model).track_frame(
        np.zeros((120, 100, 3), dtype=np.uint8)
    ) == []


def test_reset_tracker_clears_each_persistent_tracker():
    class FakeTracker:
        def __init__(self):
            self.reset_count = 0

        def reset(self):
            self.reset_count += 1

    trackers = [FakeTracker(), FakeTracker()]
    model = FakeModel()
    model.predictor = SimpleNamespace(trackers=trackers)
    detector = YOLODetector(model=model)
    detector.reset_tracker()
    assert [tracker.reset_count for tracker in trackers] == [1, 1]


def test_reset_tracker_is_safe_before_predictor_initialization():
    YOLODetector(model=FakeModel()).reset_tracker()


def test_process_stream_samples_live_frames_and_releases_capture(monkeypatch):
    from blindspot.ai_detection import process_stream

    class FakeCapture:
        def __init__(self):
            self.index = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            frame = np.full((120, 100, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

        def release(self):
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr("blindspot.ai_detection.cv2.VideoCapture", lambda source: capture)
    model = FakeModel()
    results = process_stream(
        "0", duration_seconds=60, sample_every_frames=2, max_samples=3,
        classes=[0, 2], detector=YOLODetector(model=model),
    )
    assert [result.frame_index for result in results] == [0, 2, 4]
    assert all(result.annotated_frame.shape == (120, 100, 3) for result in results)
    assert all(result.inference_time_ms >= 0 for result in results)
    assert capture.released is True


@pytest.mark.parametrize("source", ["", "camera-name", "ftp://camera", -1, True])
def test_process_stream_rejects_invalid_source(source):
    from blindspot.ai_detection import process_stream

    with pytest.raises(ValueError):
        process_stream(source, detector=YOLODetector(model=FakeModel()))


def test_process_stream_rejects_invalid_duration():
    from blindspot.ai_detection import process_stream

    with pytest.raises(ValueError, match="duration_seconds"):
        process_stream("0", duration_seconds=0, detector=YOLODetector(model=FakeModel()))


def test_coco_class_catalogue_contains_all_eighty_unique_ids():
    from blindspot.ai_detection import COCO_CLASSES

    assert len(COCO_CLASSES) == 80
    assert sorted(COCO_CLASSES.values()) == list(range(80))
