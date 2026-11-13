"""Evaluate SE(3) relative-pose constraint residuals from JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot.pose_graph_evaluation import evaluate_pose_graph_constraints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", required=True, help="JSON list of {rotation_matrix, translation_vector}")
    parser.add_argument("--constraints", required=True, help="JSON list of {i, j, rotation_matrix, translation_vector, weight?}")
    parser.add_argument("--output", default="pose_graph_validation.json")
    args = parser.parse_args()
    pose_rows = json.loads(Path(args.poses).read_text(encoding="utf-8"))
    constraint_rows = json.loads(Path(args.constraints).read_text(encoding="utf-8"))
    poses = [(row["rotation_matrix"], row["translation_vector"]) for row in pose_rows]
    constraints = [
        (row["i"], row["j"], row["rotation_matrix"], row["translation_vector"], row.get("weight", 1.0))
        for row in constraint_rows
    ]
    report = evaluate_pose_graph_constraints(poses, constraints)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
