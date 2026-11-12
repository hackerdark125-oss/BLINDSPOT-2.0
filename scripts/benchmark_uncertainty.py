"""Evaluate and fit confidence calibration from independently labelled CSV data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from blindspot.uncertainty import evaluate_calibration, fit_isotonic_calibrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with predicted_confidence,outcome")
    parser.add_argument("--output", default="uncertainty_calibration.json")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    required = {"predicted_confidence", "outcome"}
    if not required.issubset(frame.columns):
        parser.error("input CSV must contain predicted_confidence and outcome columns")
    samples = frame[list(required)].to_dict(orient="records")
    report = {
        "evaluation": evaluate_calibration(samples, bins=args.bins),
        "isotonic_calibrator": fit_isotonic_calibrator(samples),
        "warning": "Fit on training labels only; evaluate calibration metrics on a separate held-out set.",
    }
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["evaluation"], indent=2))


if __name__ == "__main__":
    main()
