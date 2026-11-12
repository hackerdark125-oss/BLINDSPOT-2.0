import math

import pytest

from blindspot.quality_evaluation import evaluate_quality_scores


def test_quality_evaluator_reports_error_and_correlation():
    result = evaluate_quality_scores([
        {"predicted_score": 10, "reference_score": 20},
        {"predicted_score": 50, "reference_score": 60},
        {"predicted_score": 90, "reference_score": 80},
    ], bins=5)
    assert result["samples"] == 3
    assert math.isclose(result["mean_absolute_error"], 13.3333333333, rel_tol=1e-8)
    assert result["pearson_correlation"] > 0.9
    assert result["spearman_rank_correlation"] == 1.0
    assert len(result["score_bins"]) == 3


def test_quality_evaluator_handles_tied_ranks():
    result = evaluate_quality_scores([
        {"predicted_score": 50, "reference_score": 20},
        {"predicted_score": 50, "reference_score": 80},
        {"predicted_score": 90, "reference_score": 90},
    ])
    assert result["spearman_rank_correlation"] is not None


@pytest.mark.parametrize("sample", [
    {"predicted_score": -1, "reference_score": 20},
    {"predicted_score": 50, "reference_score": 101},
    {"predicted_score": float("nan"), "reference_score": 50},
])
def test_quality_evaluator_rejects_invalid_scores(sample):
    with pytest.raises(ValueError):
        evaluate_quality_scores([sample])


def test_quality_evaluator_rejects_empty_data():
    with pytest.raises(ValueError, match="at least one"):
        evaluate_quality_scores([])
