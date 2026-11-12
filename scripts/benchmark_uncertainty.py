"""Fit confidence calibration on training labels and evaluate on held-out labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

# Ensure direct script execution resolves the repository package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot.uncertainty import (
    apply_isotonic_calibrator,
    evaluate_calibration,
    fit_isotonic_calibrator,
)


def load_samples(path: str) -> list[dict]:
    frame = pd.read_csv(path)
    required = {"predicted_confidence", "outcome"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain predicted_confidence and outcome columns")
    return frame[["predicted_confidence", "outcome"]].to_dict(orient="records")


def build_report(training_samples: list[dict], heldout_samples: list[dict], bins: int = 10) -> dict:
    if not training_samples or not heldout_samples:
        raise ValueError("training and held-out datasets must both contain labelled samples")
    calibrator = fit_isotonic_calibrator(training_samples)
    raw_metrics = evaluate_calibration(heldout_samples, bins=bins)
    calibrated_scores = apply_isotonic_calibrator(
        [sample["predicted_confidence"] for sample in heldout_samples], calibrator
    )
    calibrated_samples = [
        {"predicted_confidence": score, "outcome": sample["outcome"]}
        for score, sample in zip(calibrated_scores, heldout_samples)
    ]
    return {
        "training_samples": len(training_samples),
        "heldout_samples": len(heldout_samples),
        "raw_heldout_evaluation": raw_metrics,
        "calibrated_heldout_evaluation": evaluate_calibration(calibrated_samples, bins=bins),
        "isotonic_calibrator": calibrator,
        "warning": "The calibrator was fitted only on training labels; all reported evaluation metrics use the separate held-out file.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Training CSV with predicted_confidence,outcome")
    parser.add_argument("--eval", required=True, help="Separate held-out CSV with predicted_confidence,outcome")
    parser.add_argument("--output", default="uncertainty_calibration.json")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    try:
        report = build_report(load_samples(args.train), load_samples(args.eval), bins=args.bins)
    except ValueError as exc:
        parser.error(str(exc))
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "training_samples": report["training_samples"],
        "heldout_samples": report["heldout_samples"],
        "raw_heldout_evaluation": report["raw_heldout_evaluation"],
        "calibrated_heldout_evaluation": report["calibrated_heldout_evaluation"],
    }, indent=2))


if __name__ == "__main__":
    main()
