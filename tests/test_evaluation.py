import pytest

from blindspot.evaluation import evaluate_tracking


def row(frame, track, x=0, class_id=0):
    return {"frame_index": frame, "track_id": track, "class_id": class_id,
            "x": x, "y": 0, "width": 10, "height": 10}


def test_perfect_detection_and_stable_tracking():
    gt = [row(0, 7), row(1, 7)]
    result = evaluate_tracking(gt, [row(0, 3), row(1, 3)], inference_times_ms=[20, 30])
    assert result["precision"] == 1
    assert result["recall"] == 1
    assert result["f1"] == 1
    assert result["id_switches"] == 0
    assert result["estimated_fps"] == pytest.approx(40)


def test_counts_false_positives_false_negatives_and_id_switches():
    gt = [row(0, 7), row(1, 7), row(1, 8, x=30)]
    pred = [row(0, 3), row(1, 4), row(1, 9, x=80)]
    result = evaluate_tracking(gt, pred)
    assert result["true_positives"] == 2
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["id_switches"] == 1


def test_class_mismatch_is_not_a_match():
    result = evaluate_tracking([row(0, 1)], [row(0, 1, class_id=2)])
    assert result["true_positives"] == 0
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1])
def test_rejects_invalid_iou_threshold(threshold):
    with pytest.raises(ValueError):
        evaluate_tracking([], [], iou_threshold=threshold)


def test_rejects_invalid_box_and_latency():
    bad = row(0, 1)
    bad["width"] = 0
    with pytest.raises(ValueError):
        evaluate_tracking([bad], [])
    with pytest.raises(ValueError):
        evaluate_tracking([], [], inference_times_ms=[0])


def test_id_switch_counts_consecutive_sampled_frames_with_stride():
    gt = [row(0, 7), row(3, 7)]
    pred = [row(0, 3), row(3, 4)]
    assert evaluate_tracking(gt, pred)["id_switches"] == 1
