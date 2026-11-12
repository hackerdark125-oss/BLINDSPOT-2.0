"""Evaluate estimated camera poses against independent ground truth JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure direct script execution resolves the repository package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot.trajectory_evaluation import evaluate_trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimated", required=True, help="JSON file containing Nx4x4 camera-to-world matrices")
    parser.add_argument("--ground-truth", required=True, help="JSON file containing matching Nx4x4 reference matrices")
    parser.add_argument("--output", default="trajectory_benchmark.json")
    args = parser.parse_args()
    estimated = json.loads(Path(args.estimated).read_text(encoding="utf-8"))
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    report = evaluate_trajectory(estimated, ground_truth)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
