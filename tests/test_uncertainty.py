import math
import pytest

from blindspot.uncertainty import (
    evaluate_calibration, fit_isotonic_calibrator, apply_isotonic_calibrator,
)


def test_calibration_metrics_perfect_predictions():
    samples = [
        {"predicted_confidence": 0.0, "outcome": 0},
        {"predicted_confidence": 1.0, "outcome": 1},
    ]
    result = evaluate_calibration(samples, bins=5)
    assert result["samples"] == 2
    assert result["brier_score"] == 0.0
    assert result["expected_calibration_error"] == 0.0
    assert math.isfinite(result["log_loss"])


def test_calibration_metrics_detect_overconfidence():
    samples = [{"predicted_confidence": 0.9, "outcome": 0}] * 8
    result = evaluate_calibration(samples)
    assert result["expected_calibration_error"] == pytest.approx(0.9)
    assert result["brier_score"] == pytest.approx(0.81)


def test_isotonic_calibrator_is_monotone_and_applicable():
    samples = [
        {"predicted_confidence": 0.2, "outcome": 0},
        {"predicted_confidence": 0.4, "outcome": 1},
        {"predicted_confidence": 0.6, "outcome": 0},
        {"predicted_confidence": 0.8, "outcome": 1},
    ]
    calibrator = fit_isotonic_calibrator(samples)
    values = apply_isotonic_calibrator([0.1, 0.3, 0.5, 0.7, 0.9], calibrator)
    assert values == sorted(values)
    assert all(0 <= value <= 1 for value in values)


@pytest.mark.parametrize("sample", [
    {"predicted_confidence": -0.1, "outcome": 0},
    {"predicted_confidence": 0.5, "outcome": 2},
    {"predicted_confidence": float("nan"), "outcome": 1},
])
def test_calibration_rejects_invalid_labels(sample):
    with pytest.raises(ValueError):
        evaluate_calibration([sample])


def test_calibration_rejects_empty_data():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_calibration([])
