"""Ground-truth evaluation helpers for object detection and multi-object tracking.

CSV rows use frame_index, track_id, class_id, x, y, width, height.
Ground-truth and prediction files must describe the same sampled frame indices.
"""
from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Iterable, Mapping, Any


_REQUIRED = {"frame_index", "track_id", "class_id", "x", "y", "width", "height"}


def _normalise(rows: Iterable[Mapping[str, Any]], label: str) -> list[dict[str, float | int | None]]:
    normalised = []
    for row_number, row in enumerate(rows, start=1):
        missing = _REQUIRED.difference(row)
        if missing:
            raise ValueError(f"{label} row {row_number} is missing: {', '.join(sorted(missing))}")
        try:
            item = {
                "frame_index": int(row["frame_index"]),
                "track_id": None if row["track_id"] in (None, "", "None") else int(row["track_id"]),
                "class_id": int(row["class_id"]),
                "x": float(row["x"]), "y": float(row["y"]),
                "width": float(row["width"]), "height": float(row["height"]),
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} row {row_number} contains an invalid numeric value") from exc
        if item["frame_index"] < 0 or item["class_id"] < 0:
            raise ValueError(f"{label} row {row_number} has a negative frame or class ID")
        if not all(isfinite(item[key]) for key in ("x", "y", "width", "height")):
            raise ValueError(f"{label} row {row_number} has a non-finite bounding-box value")
        if item["width"] <= 0 or item["height"] <= 0:
            raise ValueError(f"{label} row {row_number} has a non-positive bounding-box size")
        normalised.append(item)
    return normalised


def _iou(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    left, top = max(a["x"], b["x"]), max(a["y"], b["y"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_tracking(
    ground_truth: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    iou_threshold: float = 0.5,
    inference_times_ms: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Compute greedy class-aware detection metrics and consecutive-frame ID switches."""
    if not isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")
    gt = _normalise(ground_truth, "ground truth")
    pred = _normalise(predictions, "prediction")
    gt_frames: dict[int, list[dict]] = defaultdict(list)
    pred_frames: dict[int, list[dict]] = defaultdict(list)
    for item in gt:
        gt_frames[item["frame_index"]].append(item)
    for item in pred:
        pred_frames[item["frame_index"]].append(item)

    tp = fp = fn = id_switches = 0
    iou_sum = 0.0
    previous_identity: dict[int, tuple[int, int]] = {}
    for frame_position, frame_index in enumerate(sorted(set(gt_frames) | set(pred_frames))):
        truths, candidates = gt_frames[frame_index], pred_frames[frame_index]
        pairs = sorted(
            ((_iou(truth, candidate), gi, pi)
             for gi, truth in enumerate(truths)
             for pi, candidate in enumerate(candidates)
             if truth["class_id"] == candidate["class_id"]),
            reverse=True,
        )
        used_gt: set[int] = set()
        used_pred: set[int] = set()
        for overlap, gi, pi in pairs:
            if overlap < iou_threshold or gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            tp += 1
            iou_sum += overlap
            truth, candidate = truths[gi], candidates[pi]
            gt_id, pred_id = truth["track_id"], candidate["track_id"]
            if gt_id is not None and pred_id is not None:
                previous = previous_identity.get(gt_id)
                if previous is not None and previous[0] == frame_position - 1 and previous[1] != pred_id:
                    id_switches += 1
                previous_identity[gt_id] = (frame_position, pred_id)
        fn += len(truths) - len(used_gt)
        fp += len(candidates) - len(used_pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    timings = []
    if inference_times_ms is not None:
        for value in inference_times_ms:
            timing = float(value)
            if not isfinite(timing) or timing <= 0:
                raise ValueError("inference times must be positive milliseconds")
            timings.append(timing)
    mean_latency = sum(timings) / len(timings) if timings else None
    return {
        "frames_evaluated": len(set(gt_frames) | set(pred_frames)),
        "ground_truth_detections": len(gt),
        "predicted_detections": len(pred),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_matched_iou": iou_sum / tp if tp else 0.0,
        "id_switches": id_switches,
        "mean_inference_ms": mean_latency,
        "estimated_fps": 1000.0 / mean_latency if mean_latency else None,
        "iou_threshold": iou_threshold,
        "matching_note": "Greedy class-aware IoU matching; ID switches count changes on consecutive evaluated frames only.",
    }
