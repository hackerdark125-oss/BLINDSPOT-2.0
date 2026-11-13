"""Regression tests for tracking metrics and camera-freeze diagnostics."""

import numpy as np
import pytest

from blindspot.evaluation import evaluate_tracking
from blindspot.tracking import FrameFreezeDetector


def _row(frame, track, x=10, class_id=0):
    return {
        "frame_index": frame,
        "track_id": track,
        "class_id": class_id,
        "x": x,
        "y": 10,
        "width": 20,
        "height": 30,
    }


def test_tracking_evaluation_perfect_match_reports_perfect_detection_metrics():
    truth = [_row(0, 4), _row(1, 4)]
    predictions = [_row(0, 9), _row(1, 9)]

    result = evaluate_tracking(truth, predictions)

    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["mean_matched_iou"] == 1.0
    assert result["id_switches"] == 0


def test_tracking_evaluation_counts_false_positives_and_missed_detections():
    truth = [_row(0, 1), _row(0, 2, x=100)]
    predictions = [_row(0, 7), _row(0, 8, x=300)]

    result = evaluate_tracking(truth, predictions)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_tracking_evaluation_counts_identity_switch_on_consecutive_frames():
    truth = [_row(0, 1), _row(1, 1)]
    predictions = [_row(0, 10), _row(1, 11)]

    result = evaluate_tracking(truth, predictions)

    assert result["id_switches"] == 1


@pytest.mark.parametrize(
    "truth,predictions,threshold",
    [
        ([], [], 0),
        ([{"frame_index": 0}], [], 0.5),
        ([_row(0, 1, x=float("nan"))], [], 0.5),
    ],
)
def test_tracking_evaluation_rejects_invalid_inputs(truth, predictions, threshold):
    with pytest.raises(ValueError):
        evaluate_tracking(truth, predictions, iou_threshold=threshold)


def test_tracking_evaluation_reports_latency_and_rejects_invalid_timing():
    result = evaluate_tracking(
        [_row(0, 1)], [_row(0, 2)], processing_times_ms=[10.0, 20.0]
    )

    assert result["mean_processing_ms"] == pytest.approx(15.0)
    assert result["estimated_processing_fps"] == pytest.approx(1000.0 / 15.0)
    with pytest.raises(ValueError, match="positive milliseconds"):
        evaluate_tracking([], [], processing_times_ms=[0.0])


def test_freeze_detector_flags_repeated_frames_and_recovers_on_change():
    detector = FrameFreezeDetector(threshold=0.0, consecutive_samples=3)
    still = np.zeros((80, 80, 3), dtype=np.uint8)

    _, first = detector.update(still)
    _, second = detector.update(still)
    _, third = detector.update(still)

    assert first is False
    assert second is False
    assert third is True

    changed = np.full((80, 80, 3), 255, dtype=np.uint8)
    _, recovered = detector.update(changed)
    assert recovered is False


def test_freeze_detector_rejects_invalid_frames_and_settings():
    with pytest.raises(ValueError):
        FrameFreezeDetector(threshold=1.0)
    with pytest.raises(ValueError):
        FrameFreezeDetector(consecutive_samples=1)

    detector = FrameFreezeDetector()
    with pytest.raises(ValueError):
        detector.update(np.empty((0, 0), dtype=np.uint8))
