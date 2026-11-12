"""CLI to compare exported BLINDSPOT per-cell scores with human quality labels."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from blindspot.quality_evaluation import evaluate_quality_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with predicted_score and reference_score")
    parser.add_argument("--output", default="quality_benchmark.json", help="Output JSON path")
    parser.add_argument("--bins", type=int, default=10, help="Number of score bins (2-20)")
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"predicted_score", "reference_score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("CSV must contain predicted_score and reference_score columns")
        metrics = evaluate_quality_scores(reader, bins=args.bins)
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
