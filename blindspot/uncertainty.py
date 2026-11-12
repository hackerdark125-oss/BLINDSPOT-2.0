"""Label-driven calibration and evaluation for confidence estimates.

Metrics are only meaningful when outcomes are independently labelled and the
evaluation set is held out from any calibration fitting.
"""
from __future__ import annotations

from math import log
from typing import Iterable, Mapping


def _validated_samples(samples: Iterable[Mapping[str, float]]) -> tuple[list[float], list[int]]:
    confidence, outcomes = [], []
    for sample in samples:
        try:
            score = float(sample["predicted_confidence"])
            outcome_raw = float(sample["outcome"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each sample needs numeric predicted_confidence and outcome") from exc
        if not 0.0 <= score <= 1.0:
            raise ValueError("predicted_confidence values must be in [0, 1]")
        if outcome_raw not in (0.0, 1.0):
            raise ValueError("outcome values must be binary 0 or 1")
        confidence.append(score)
        outcomes.append(int(outcome_raw))
    if not confidence:
        raise ValueError("at least one labelled sample is required")
    return confidence, outcomes


def evaluate_calibration(samples: Iterable[Mapping[str, float]], bins: int = 10) -> dict:
    """Return Brier score, log loss, ECE, and reliability-bin summaries."""
    confidence, outcomes = _validated_samples(samples)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    count = len(confidence)
    brier = sum((p - y) ** 2 for p, y in zip(confidence, outcomes)) / count
    eps = 1e-15
    log_loss = -sum(y * log(max(p, eps)) + (1 - y) * log(max(1 - p, eps))
                    for p, y in zip(confidence, outcomes)) / count
    rows, ece = [], 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = [i for i, p in enumerate(confidence)
                    if lower <= p < upper or (index == bins - 1 and p == 1.0)]
        if not selected:
            continue
        mean_conf = sum(confidence[i] for i in selected) / len(selected)
        observed = sum(outcomes[i] for i in selected) / len(selected)
        gap = abs(mean_conf - observed)
        ece += len(selected) / count * gap
        rows.append({
            "lower_bound": lower, "upper_bound": upper, "samples": len(selected),
            "mean_confidence": mean_conf, "observed_frequency": observed,
            "absolute_gap": gap,
        })
    return {"samples": count, "brier_score": brier, "log_loss": log_loss,
            "expected_calibration_error": ece, "reliability_bins": rows}


def fit_isotonic_calibrator(samples: Iterable[Mapping[str, float]]) -> dict:
    """Fit a monotone piecewise-constant confidence map using PAVA."""
    confidence, outcomes = _validated_samples(samples)
    ordered = sorted(zip(confidence, outcomes), key=lambda item: item[0])
    blocks: list[list[float]] = []
    for score, outcome in ordered:
        if blocks and score == blocks[-1][1]:
            blocks[-1][2] += outcome
            blocks[-1][3] += 1
        else:
            blocks.append([score, score, float(outcome), 1.0])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[2] / left[3] <= right[2] / right[3]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2], left[3] + right[3]]]
    return {
        "method": "isotonic_regression_pava",
        "training_samples": len(confidence),
        "blocks": [{"min_score": b[0], "max_score": b[1],
                    "calibrated_confidence": b[2] / b[3], "samples": int(b[3])}
                   for b in blocks],
    }


def apply_isotonic_calibrator(scores: Iterable[float], calibrator: Mapping) -> list[float]:
    """Apply a fitted isotonic map; values outside training range use edge blocks."""
    blocks = calibrator.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("calibrator must contain non-empty isotonic blocks")
    result = []
    for raw in scores:
        score = float(raw)
        if not 0.0 <= score <= 1.0:
            raise ValueError("scores must be in [0, 1]")
        selected = blocks[-1]
        for index, block in enumerate(blocks[:-1]):
            boundary = (float(block["max_score"]) + float(blocks[index + 1]["min_score"])) / 2.0
            if score <= boundary:
                selected = block
                break
        value = float(selected["calibrated_confidence"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("calibrator contains invalid calibrated confidence")
        result.append(value)
    return result
