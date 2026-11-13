"""Timestamp alignment utilities for independently recorded camera streams."""
from __future__ import annotations

from bisect import bisect_left
from math import isfinite
from numbers import Real
from typing import Mapping, Sequence, Any


def synchronize_timestamps(
    streams: Mapping[str, Sequence[float]],
    offsets_seconds: Mapping[str, float] | None = None,
    tolerance_seconds: float = 0.25,
) -> list[dict[str, Any]]:
    """Match nearest samples across camera timelines after manual time offsets.

    The first stream is the reference timeline. Offsets are added to source
    timestamps before matching. Each sample from a non-reference camera can be
    used at most once. This aligns timestamps only; it does not synchronize
    camera clocks or guarantee that frames depict the same instant.
    """
    if not isinstance(streams, Mapping) or len(streams) < 2:
        raise ValueError("at least two camera streams are required")
    if (isinstance(tolerance_seconds, bool) or not isinstance(tolerance_seconds, Real)
            or not isfinite(tolerance_seconds) or tolerance_seconds < 0):
        raise ValueError("tolerance_seconds must be a non-negative finite number")
    if offsets_seconds is None:
        offsets_seconds = {}
    if not isinstance(offsets_seconds, Mapping):
        raise ValueError("offsets_seconds must be a mapping")
    unknown_offsets = set(offsets_seconds) - set(streams)
    if unknown_offsets:
        raise ValueError(f"offsets supplied for unknown camera streams: {sorted(unknown_offsets)}")
    normalized: dict[str, list[float]] = {}
    original: dict[str, list[float]] = {}
    for name, timestamps in streams.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("camera stream names must be non-empty strings")
        try:
            raw_values = list(timestamps)
        except TypeError as exc:
            raise ValueError(f"camera stream {name!r} timestamps must be a sequence") from exc
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
            raise ValueError(f"camera stream {name!r} timestamps must be numeric")
        values = [float(value) for value in raw_values]
        if not values:
            raise ValueError(f"camera stream {name!r} has no timestamps")
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError(f"camera stream {name!r} timestamps must be finite and non-negative")
        if any(values[index] < values[index - 1] for index in range(1, len(values))):
            raise ValueError(f"camera stream {name!r} timestamps must be non-decreasing")
        raw_offset = offsets_seconds.get(name, 0.0)
        if isinstance(raw_offset, bool) or not isinstance(raw_offset, Real):
            raise ValueError(f"camera stream {name!r} offset must be numeric and finite")
        offset = float(raw_offset)
        if not isfinite(offset):
            raise ValueError(f"camera stream {name!r} offset must be numeric and finite")
        aligned_values = [value + offset for value in values]
        if not all(isfinite(value) for value in aligned_values):
            raise ValueError(f"camera stream {name!r} aligned timestamps must remain finite")
        original[name] = values
        normalized[name] = aligned_values

    names = list(streams)
    reference_name = names[0]
    cursors = {name: 0 for name in names[1:]}
    aligned: list[dict[str, Any]] = []
    for reference_index, reference_time in enumerate(normalized[reference_name]):
        matched: dict[str, Any] = {
            reference_name: {
                "sample_index": reference_index,
                "source_time_seconds": original[reference_name][reference_index],
                "aligned_time_seconds": reference_time,
                "skew_seconds": 0.0,
            }
        }
        proposed_cursors = dict(cursors)
        for name in names[1:]:
            timeline = normalized[name]
            cursor = proposed_cursors[name]
            position = bisect_left(timeline, reference_time, lo=cursor)
            candidates = [index for index in (position - 1, position) if cursor <= index < len(timeline)]
            if not candidates:
                matched = {}
                break
            best = min(candidates, key=lambda index: abs(timeline[index] - reference_time))
            skew = timeline[best] - reference_time
            if abs(skew) > tolerance_seconds:
                matched = {}
                break
            matched[name] = {
                "sample_index": best,
                "source_time_seconds": original[name][best],
                "aligned_time_seconds": timeline[best],
                "skew_seconds": skew,
            }
            proposed_cursors[name] = best + 1
        if matched:
            cursors = proposed_cursors
            aligned.append({
                "reference_time_seconds": reference_time,
                "samples": matched,
            })
    return aligned
