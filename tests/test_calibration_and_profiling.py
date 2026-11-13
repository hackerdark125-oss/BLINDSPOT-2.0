"""Regression tests for score calibration diagnostics and profiling CLI."""
import pytest

from blindspot.quality_evaluation import evaluate_quality_scores
from scripts.profile_reconstruction import run_profile


def test_quality_score_calibration_diagnostics_are_explicitly_non_probabilistic():
    result = evaluate_quality_scores([
        {"predicted_score": 10, "reference_score": 20},
        {"predicted_score": 90, "reference_score": 80},
    ], bins=2)

    diagnostics = result["score_calibration_diagnostics"]
    assert diagnostics["expected_absolute_score_gap"] == pytest.approx(0.1)
    assert diagnostics["maximum_absolute_score_gap"] == pytest.approx(0.1)
    assert diagnostics["normalised_brier_like_error"] == pytest.approx(0.01)
    assert "not probabilistic calibration" in diagnostics["interpretation"]


def test_quality_score_calibration_rejects_out_of_range_scores():
    with pytest.raises(ValueError, match="0 to 100"):
        evaluate_quality_scores([
            {"predicted_score": 101, "reference_score": 50}
        ])


def test_reconstruction_profile_reports_time_memory_and_quality_metrics():
    report = run_profile(point_count=120, pose_count=4)

    assert report["schema_version"] == 1
    assert len(report["workloads"]) == 2
    for workload in report["workloads"]:
        assert workload["wall_time_seconds"] >= 0
        assert workload["python_tracemalloc_peak_bytes"] > 0
    assert report["workloads"][0]["fitness"] > 0.95
    assert report["workloads"][0]["rmse"] < 1e-4
    assert report["workloads"][1]["final_cost"] <= report["workloads"][1]["initial_cost"]


@pytest.mark.parametrize("kwargs", [{"point_count": 99}, {"pose_count": 2}])
def test_reconstruction_profile_rejects_invalid_workload_sizes(kwargs):
    with pytest.raises(ValueError):
        run_profile(**kwargs)
