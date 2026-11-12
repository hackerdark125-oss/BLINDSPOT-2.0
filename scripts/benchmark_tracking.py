"""Evaluate prediction CSV files against manually labelled ground truth."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from blindspot.evaluation import evaluate_tracking


def read_rows(path: str):
    """Read benchmark CSVs or the user-friendly CSV exported by the Streamlit app."""
    aliases = {
        "Frame index": "frame_index", "Track ID": "track_id",
        "Class ID": "class_id", "X": "x", "Y": "y",
        "Width": "width", "Height": "height",
    }
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return [
            {aliases.get(key, key): value for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, help="CSV with frame_index,track_id,class_id,x,y,width,height")
    parser.add_argument("--predictions", required=True, help="CSV with the same schema or the app's AI tracking export")
    parser.add_argument("--output", default="benchmark.json", help="Output JSON path")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU matching threshold (default: 0.5)")
    args = parser.parse_args()
    prediction_rows = read_rows(args.predictions)
    # The app exports one row per detected object, so a frame's latency is
    # repeated when several objects are present. Count each sampled frame once.
    timing_by_frame = {}
    for row in prediction_rows:
        if row.get("Inference (ms)") not in (None, ""):
            timing_by_frame.setdefault(row["frame_index"], float(row["Inference (ms)"]))
    inference_times = list(timing_by_frame.values())
    metrics = evaluate_tracking(
        read_rows(args.ground_truth), prediction_rows, args.iou,
        inference_times_ms=inference_times or None,
    )
    output = Path(args.output)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved benchmark: {output}")


if __name__ == "__main__":
    main()
