import pytest

from blindspot.synchronization import synchronize_timestamps


def test_synchronize_camera_streams_with_offsets():
    matches = synchronize_timestamps(
        {"camera_a": [0.0, 1.0, 2.0], "camera_b": [0.2, 1.2, 2.2]},
        offsets_seconds={"camera_b": -0.2},
        tolerance_seconds=0.01,
    )
    assert len(matches) == 3
    assert matches[1]["samples"]["camera_b"]["sample_index"] == 1
    assert matches[1]["samples"]["camera_b"]["skew_seconds"] == pytest.approx(0.0)


def test_synchronization_does_not_reuse_samples():
    matches = synchronize_timestamps(
        {"a": [0.0, 0.1, 0.2], "b": [0.05]},
        tolerance_seconds=0.1,
    )
    assert len(matches) == 1


def test_synchronization_returns_empty_when_streams_do_not_overlap():
    assert synchronize_timestamps(
        {"a": [0.0, 1.0], "b": [10.0, 11.0]},
        tolerance_seconds=0.1,
    ) == []


@pytest.mark.parametrize("streams", [
    {"only_one": [0.0]},
    {"a": [0.0, 1.0], "b": [1.0, 0.0]},
    {"a": [0.0], "b": [float("nan")]},
])
def test_synchronization_rejects_invalid_streams(streams):
    with pytest.raises(ValueError):
        synchronize_timestamps(streams)


def test_failed_multi_camera_match_does_not_consume_partial_samples():
    matches = synchronize_timestamps(
        {"a": [0.0, 0.05], "b": [0.05], "c": [10.0]},
        tolerance_seconds=0.1,
    )
    assert matches == []
