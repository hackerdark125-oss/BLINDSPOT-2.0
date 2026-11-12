"""Reproducible synthetic benchmark for multi-camera point-cloud fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

# Make direct execution (python scripts/benchmark_fusion.py) resolve the repo package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot.multicam import fuse_multicamera_point_clouds


def run_benchmark(
    camera_count: int = 2,
    points_per_camera: int = 100_000,
    voxel_size: float = 0.05,
    repeats: int = 3,
    seed: int = 42,
) -> dict:
    if camera_count < 1 or points_per_camera < 1 or repeats < 1:
        raise ValueError("camera_count, points_per_camera, and repeats must be positive")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive and finite")

    rng = np.random.default_rng(seed)
    clouds = {}
    poses = {}
    for index in range(camera_count):
        camera_id = f"camera_{index + 1}"
        xyz = rng.uniform(-5, 5, size=(points_per_camera, 3))
        rgb = rng.integers(0, 256, size=(points_per_camera, 3))
        clouds[camera_id] = np.column_stack((xyz, rgb)).astype(np.float32)
        poses[camera_id] = (np.eye(3), [float(index) * 0.25, 0.0, 0.0])

    timings = []
    output_points = 0
    for _ in range(repeats):
        started = time.perf_counter()
        result = fuse_multicamera_point_clouds(clouds, poses, voxel_size=voxel_size)
        timings.append(time.perf_counter() - started)
        output_points = result.output_points

    total_points = camera_count * points_per_camera
    mean_seconds = float(np.mean(timings))
    return {
        "camera_count": camera_count,
        "points_per_camera": points_per_camera,
        "input_points": total_points,
        "output_points": output_points,
        "voxel_size": voxel_size,
        "repeats": repeats,
        "mean_seconds": mean_seconds,
        "median_seconds": float(np.median(timings)),
        "points_per_second": float(total_points / mean_seconds),
        "seed": seed,
        "note": "Synthetic benchmark only; record hardware and real-scene results separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--points-per-camera", type=int, default=100_000)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("fusion_benchmark.json"))
    args = parser.parse_args()
    report = run_benchmark(
        args.cameras, args.points_per_camera, args.voxel_size, args.repeats, args.seed
    )
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
