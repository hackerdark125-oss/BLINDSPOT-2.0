"""Build an ASCII PLY point cloud from a saved depth array and intrinsics JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from blindspot.reconstruction import depth_to_point_cloud, write_ply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", required=True, help="Path to a 2D .npy depth map")
    parser.add_argument("--intrinsics", required=True, help="JSON file containing camera_matrix")
    parser.add_argument("--output", default="blindspot_scene.ply")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float)
    args = parser.parse_args()
    depth = np.load(args.depth, allow_pickle=False)
    calibration = json.loads(Path(args.intrinsics).read_text(encoding="utf-8"))
    if "camera_matrix" not in calibration:
        parser.error("intrinsics JSON must contain camera_matrix")
    cloud = depth_to_point_cloud(
        depth, calibration["camera_matrix"], stride=args.stride, max_depth=args.max_depth
    )
    write_ply(args.output, cloud)
    print(json.dumps({"output": str(args.output), "points": int(len(cloud)),
                      "format": "PLY ASCII"}, indent=2))


if __name__ == "__main__":
    main()
