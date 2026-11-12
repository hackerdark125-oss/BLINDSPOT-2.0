"""CLI for evaluating stereo depth against manually measured target distances."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

# Ensure direct script execution resolves the repository package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot.depth_evaluation import evaluate_depth_samples


def read_samples(path: str) -> tuple[list[dict[str, object]], str]:
    samples: list[dict[str, object]] = []
    units: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"known_distance", "measured_distance", "valid_pixels", "total_pixels"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            units.add((row.get("unit") or "unspecified").strip())
            samples.append({
                "known_distance": row["known_distance"],
                "measured_distance": row["measured_distance"] or None,
                "valid_pixels": row["valid_pixels"],
                "total_pixels": row["total_pixels"],
            })
    if len(units) > 1:
        raise ValueError("all rows must use the same distance unit")
    return samples, next(iter(units), "unspecified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of known and measured target distances")
    parser.add_argument("--output", default="depth_benchmark.json", help="Output JSON path")
    args = parser.parse_args()
    samples, unit = read_samples(args.input)
    metrics = evaluate_depth_samples(samples)
    metrics["distance_unit"] = unit
    Path(args.output).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
