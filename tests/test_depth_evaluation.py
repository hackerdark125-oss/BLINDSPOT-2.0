import math

import pytest

from blindspot.depth_evaluation import evaluate_depth_samples


def test_depth_metrics_and_valid_pixel_coverage():
    result = evaluate_depth_samples([
        {"known_distance": 10, "measured_distance": 11, "valid_pixels": 90, "total_pixels": 100},
        {"known_distance": 20, "measured_distance": 18, "valid_pixels": 80, "total_pixels": 100},
    ])
    assert result["samples_evaluated"] == 2
    assert result["mean_absolute_error"] == 1.5
    assert math.isclose(result["root_mean_squared_error"], math.sqrt(2.5))
    assert result["median_absolute_error"] == 1.5
    assert result["mean_absolute_relative_error_percent"] == 10.0
    assert result["valid_pixel_percent"] == 85.0


def test_depth_benchmark_counts_samples_with_no_valid_pixels():
    result = evaluate_depth_samples([
        {"known_distance": 10, "measured_distance": 10, "valid_pixels": 50, "total_pixels": 100},
        {"known_distance": 20, "measured_distance": None, "valid_pixels": 0, "total_pixels": 100},
    ])
    assert result["samples_evaluated"] == 1
    assert result["samples_without_valid_depth"] == 1
    assert result["valid_pixel_percent"] == 25.0


@pytest.mark.parametrize("sample", [
    {"known_distance": 0, "measured_distance": 1, "valid_pixels": 1, "total_pixels": 1},
    {"known_distance": 1, "measured_distance": 1, "valid_pixels": 2, "total_pixels": 1},
    {"known_distance": 1, "measured_distance": None, "valid_pixels": 1, "total_pixels": 1},
])
def test_depth_benchmark_rejects_invalid_samples(sample):
    with pytest.raises(ValueError):
        evaluate_depth_samples([sample])


def test_depth_benchmark_requires_at_least_one_valid_measurement():
    with pytest.raises(ValueError, match="no sample"):
        evaluate_depth_samples([
            {"known_distance": 10, "measured_distance": None, "valid_pixels": 0, "total_pixels": 100}
        ])


def test_depth_benchmark_rejects_fractional_pixel_counts():
    with pytest.raises(ValueError, match="pixel counts"):
        evaluate_depth_samples([
            {"known_distance": 10, "measured_distance": 10, "valid_pixels": 1.5, "total_pixels": 10}
        ])
