import pytest

from scripts.benchmark_uncertainty import build_report


def test_uncertainty_cli_fits_on_train_and_reports_heldout_metrics():
    training = [
        {"predicted_confidence": 0.1, "outcome": 0},
        {"predicted_confidence": 0.2, "outcome": 0},
        {"predicted_confidence": 0.8, "outcome": 1},
        {"predicted_confidence": 0.9, "outcome": 1},
    ]
    heldout = [
        {"predicted_confidence": 0.15, "outcome": 0},
        {"predicted_confidence": 0.85, "outcome": 1},
    ]
    report = build_report(training, heldout, bins=5)
    assert report["training_samples"] == 4
    assert report["heldout_samples"] == 2
    assert report["isotonic_calibrator"]["training_samples"] == 4
    assert report["raw_heldout_evaluation"]["samples"] == 2
    assert report["calibrated_heldout_evaluation"]["samples"] == 2


def test_uncertainty_cli_requires_both_splits():
    with pytest.raises(ValueError, match="both contain"):
        build_report([], [{"predicted_confidence": 0.5, "outcome": 1}])
