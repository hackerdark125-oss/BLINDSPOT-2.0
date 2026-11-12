"""Quantitative evaluation of stereo depth estimates against known distances."""
from __future__ import annotations

from math import isfinite, sqrt
from statistics import median
from typing import Iterable, Mapping, Any


def evaluate_depth_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, float | int]:
    """Evaluate per-target median depth and aggregate valid-pixel coverage.

    Each row requires known_distance, valid_pixels, and total_pixels. A missing
    measured_distance is allowed only for a sample with zero valid pixels.
    Distances must all use one consistent unit supplied by the caller.
    """
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    total_valid = 0
    total_pixels = 0
    sample_count = 0
    skipped_samples = 0

    for index, sample in enumerate(samples, start=1):
        try:
            known = float(sample["known_distance"])
            valid = int(sample["valid_pixels"])
            total = int(sample["total_pixels"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"sample {index} is missing valid numeric fields") from exc
        if not isfinite(known) or known <= 0:
            raise ValueError(f"sample {index} known_distance must be positive and finite")
        if valid < 0 or total <= 0 or valid > total:
            raise ValueError(f"sample {index} must satisfy 0 <= valid_pixels <= total_pixels")
        total_valid += valid
        total_pixels += total
        measured_raw = sample.get("measured_distance")
        if measured_raw in (None, ""):
            if valid != 0:
                raise ValueError(f"sample {index} has valid pixels but no measured_distance")
            skipped_samples += 1
            continue
        try:
            measured = float(measured_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sample {index} measured_distance must be numeric") from exc
        if not isfinite(measured) or measured <= 0:
            raise ValueError(f"sample {index} measured_distance must be positive and finite")
        error = abs(measured - known)
        absolute_errors.append(error)
        relative_errors.append(error / known * 100.0)
        sample_count += 1

    if total_pixels == 0:
        raise ValueError("at least one depth sample is required")
    if not sample_count:
        raise ValueError("no sample contains a valid measured depth")

    mae = sum(absolute_errors) / sample_count
    rmse = sqrt(sum(error * error for error in absolute_errors) / sample_count)
    return {
        "samples_evaluated": sample_count,
        "samples_without_valid_depth": skipped_samples,
        "mean_absolute_error": mae,
        "root_mean_squared_error": rmse,
        "median_absolute_error": float(median(absolute_errors)),
        "mean_absolute_relative_error_percent": sum(relative_errors) / sample_count,
        "valid_pixel_percent": total_valid / total_pixels * 100.0,
    }
