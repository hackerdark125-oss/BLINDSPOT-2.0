"""Agreement metrics for heuristic image-quality scores and human labels."""
from __future__ import annotations

from math import isfinite, sqrt
from typing import Any, Iterable, Mapping


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average_rank
        cursor = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator <= 1e-12:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def evaluate_quality_scores(
    samples: Iterable[Mapping[str, Any]], bins: int = 10
) -> dict[str, Any]:
    """Compare app quality scores with human reference scores, both on 0..100.

    The calibration diagnostics treat human ratings as soft reference outcomes.
    They are score-agreement diagnostics, not probabilistic calibration metrics.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 20:
        raise ValueError("bins must be an integer between 2 and 20")
    predicted: list[float] = []
    reference: list[float] = []
    for index, sample in enumerate(samples, start=1):
        try:
            prediction = float(sample["predicted_score"])
            target = float(sample["reference_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"sample {index} requires numeric predicted_score and reference_score") from exc
        if not isfinite(prediction) or not isfinite(target) or not 0 <= prediction <= 100 or not 0 <= target <= 100:
            raise ValueError(f"sample {index} scores must be finite values from 0 to 100")
        predicted.append(prediction)
        reference.append(target)
    if not predicted:
        raise ValueError("at least one labelled quality sample is required")

    errors = [prediction - target for prediction, target in zip(predicted, reference)]
    absolute_errors = [abs(error) for error in errors]
    mse = sum(error * error for error in errors) / len(errors)
    spearman = _correlation(_average_ranks(predicted), _average_ranks(reference))
    bin_rows = []
    for bin_index in range(bins):
        low = bin_index * 100.0 / bins
        high = (bin_index + 1) * 100.0 / bins
        members = [
            index for index, score in enumerate(predicted)
            if low <= score < high or (bin_index == bins - 1 and score == 100)
        ]
        if members:
            mean_predicted = sum(predicted[i] for i in members) / len(members)
            mean_reference = sum(reference[i] for i in members) / len(members)
            bin_rows.append({
                "bin_low": low,
                "bin_high": high,
                "count": len(members),
                "mean_predicted_score": mean_predicted,
                "mean_reference_score": mean_reference,
                "absolute_gap": abs(mean_predicted - mean_reference),
            })
    expected_gap = sum(
        row["count"] / len(predicted) * row["absolute_gap"] for row in bin_rows
    ) / 100.0
    max_gap = max((row["absolute_gap"] for row in bin_rows), default=0.0) / 100.0
    return {
        "samples": len(predicted),
        "mean_absolute_error": sum(absolute_errors) / len(errors),
        "root_mean_squared_error": sqrt(mse),
        "mean_bias_predicted_minus_reference": sum(errors) / len(errors),
        "pearson_correlation": _correlation(predicted, reference),
        "spearman_rank_correlation": spearman,
        "score_bins": bin_rows,
        "score_calibration_diagnostics": {
            "expected_absolute_score_gap": expected_gap,
            "maximum_absolute_score_gap": max_gap,
            "normalised_brier_like_error": mse / 10000.0,
            "interpretation": (
                "Exploratory agreement with normalized human ratings; not "
                "probabilistic calibration or a claim that scores are probabilities."
            ),
        },
        "interpretation": "Agreement with human labels; scores are not calibrated probabilities.",
    }
}
