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
        {"a": [0.0, 0.05], "b": [0.05], "c": [0.05]},
        tolerance_seconds=0.01,
    )
    assert len(matches) == 1
    assert matches[0]["reference_time_seconds"] == pytest.approx(0.05)
    assert matches[0]["samples"]["b"]["sample_index"] == 0


@pytest.mark.parametrize("streams", [
    {"a": ["0.0", "1.0"], "b": [0.0, 1.0]},
    {"a": [True, 1.0], "b": [0.0, 1.0]},
    {"a": [0.0, 1.0], "b": [0.0, float("inf")]},
])
def test_synchronization_rejects_non_numeric_or_non_finite_timestamps(streams):
    with pytest.raises(ValueError, match="timestamps"):
        synchronize_timestamps(streams)


@pytest.mark.parametrize("offsets", [
    {"b": "0.1"},
    {"b": True},
    {"b": float("nan")},
    {"unknown": 0.0},
])
def test_synchronization_rejects_invalid_offsets(offsets):
    with pytest.raises(ValueError):
        synchronize_timestamps({"a": [0.0], "b": [0.0]}, offsets_seconds=offsets)


@pytest.mark.parametrize("tolerance", [True, -0.1, float("nan"), float("inf"), "0.1"])
def test_synchronization_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance_seconds"):
        synchronize_timestamps({"a": [0.0], "b": [0.0]}, tolerance_seconds=tolerance)


def test_synchronization_rejects_offset_overflow():
    with pytest.raises(ValueError, match="aligned timestamps"):
        synchronize_timestamps(
            {"a": [1e308], "b": [1e308]},
            offsets_seconds={"b": 1e308},
        )
