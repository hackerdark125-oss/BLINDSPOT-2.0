"""Profile deterministic synthetic ICP and SE(3) workloads.

This is a reproducible software microbenchmark, not a substitute for profiling
representative camera footage or target deployment hardware.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot.pose_graph_se3 import optimize_pose_graph_se3
from blindspot.registration import register_point_cloud_icp


def _rss_bytes() -> int | None:
    """Return current resident set size when optional psutil is installed."""
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return None


def _profile(label: str, operation: Callable[[], Any]) -> tuple[dict[str, Any], Any]:
    rss_before = _rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    result = operation()
    elapsed = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_bytes()
    return ({
        "name": label,
        "wall_time_seconds": elapsed,
        "python_tracemalloc_current_bytes": int(current_bytes),
        "python_tracemalloc_peak_bytes": int(peak_bytes),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": (
            rss_after - rss_before
            if rss_before is not None and rss_after is not None else None
        ),
    }, result)


def run_profile(point_count: int = 800, pose_count: int = 8) -> dict[str, Any]:
    """Run deterministic registration and loop-closure profiling workloads."""
    if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 100:
        raise ValueError("point_count must be an integer >= 100")
    if isinstance(pose_count, bool) or not isinstance(pose_count, int) or pose_count < 3:
        raise ValueError("pose_count must be an integer >= 3")

    rng = np.random.default_rng(20261113)
    source = rng.uniform(-2.0, 2.0, size=(point_count, 3))
    angle = 0.004
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = np.array([0.015, -0.012, 0.008])
    target = (rotation @ source.T).T + translation

    icp_metrics, icp_result = _profile(
        "synthetic_icp_registration",
        lambda: register_point_cloud_icp(
            source, target,
            max_correspondence_distance=0.5,
            voxel_size=0.005,
            trim_fraction=1.0,
            max_iterations=30,
            max_points=max(point_count, 100),
        ),
    )
    icp_metrics.update({
        "input_points": point_count,
        "fitness": float(icp_result.fitness),
        "rmse": float(icp_result.rmse),
        "iterations": int(icp_result.iterations),
        "converged": bool(icp_result.converged),
    })

    poses = [
        (np.eye(3), np.array([float(index) + (0.15 if index == pose_count - 1 else 0.0), 0.0, 0.0]))
        for index in range(pose_count)
    ]
    closure_translation = np.array([-(pose_count - 1.0), 0.0, 0.0])
    pose_metrics, pose_result = _profile(
        "synthetic_se3_pose_graph_with_loop_closure",
        lambda: optimize_pose_graph_se3(
            poses,
            loop_closures=[(0, pose_count - 1, np.eye(3), closure_translation, 8.0)],
            max_iterations=20,
        ),
    )
    pose_metrics.update({
        "pose_count": pose_count,
        "iterations": int(pose_result.iterations),
        "initial_cost": float(pose_result.initial_cost),
        "final_cost": float(pose_result.final_cost),
        "converged": bool(pose_result.converged),
    })

    return {
        "schema_version": 1,
        "benchmark": "BLINDSPOT deterministic synthetic reconstruction profiling",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "notes": [
            "Synthetic workload timings are comparative diagnostics, not field performance guarantees.",
            "tracemalloc reports Python-tracked allocations and may not capture every native allocation.",
            "RSS fields are populated only when optional psutil is installed.",
            "Use representative footage and target hardware before making deployment claims.",
        ],
        "workloads": [icp_metrics, pose_metrics],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=800, help="Synthetic point count (>=100)")
    parser.add_argument("--poses", type=int, default=8, help="Synthetic trajectory length (>=3)")
    parser.add_argument("--output", default="profile_report.json")
    args = parser.parse_args()
    report = run_profile(args.points, args.poses)
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved profile report: {output}")


if __name__ == "__main__":
    main()
