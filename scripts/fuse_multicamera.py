"""Fuse pre-calibrated camera-local XYZRGB PLY clouds from the command line."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure direct script execution resolves the repository package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blindspot.coverage import load_camera_pose_json
from blindspot.multicam import fuse_multicamera_point_clouds
from blindspot.ply import read_ascii_ply
from blindspot.reconstruction import write_ply


def _camera_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("camera input must use CAMERA_ID=path.ply")
    camera_id, path_text = value.split("=", 1)
    camera_id = camera_id.strip()
    path = Path(path_text).expanduser()
    if not camera_id or not path_text.strip():
        raise argparse.ArgumentTypeError("camera ID and PLY path must both be non-empty")
    return camera_id, path


def fuse_files(
    cloud_inputs: list[tuple[str, Path]],
    pose_path: Path,
    output_path: Path,
    voxel_size: float | None = None,
    summary_path: Path | None = None,
) -> dict:
    """Load, validate, transform, and fuse named point clouds."""
    if len(cloud_inputs) < 2:
        raise ValueError("at least two camera PLY inputs are required")
    camera_ids = [camera_id for camera_id, _ in cloud_inputs]
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("camera IDs must be unique")
    if any(not path.is_file() for _, path in cloud_inputs):
        missing = [str(path) for _, path in cloud_inputs if not path.is_file()]
        raise ValueError(f"PLY input file(s) not found: {', '.join(missing)}")
    if not pose_path.is_file():
        raise ValueError(f"pose JSON file not found: {pose_path}")

    try:
        document = json.loads(pose_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read pose JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("pose JSON must contain a JSON object")
    distance_unit = document.get("distance_unit")
    camera_documents = document.get("cameras")
    if not isinstance(distance_unit, str) or not distance_unit.strip():
        raise ValueError("pose JSON must define a non-empty distance_unit")
    if not isinstance(camera_documents, dict) or not camera_documents:
        raise ValueError("pose JSON must contain a non-empty cameras object")
    if set(camera_ids) != set(camera_documents):
        raise ValueError("camera IDs from PLY inputs must match pose JSON camera IDs exactly")

    clouds = {}
    poses = {}
    for camera_id, path in cloud_inputs:
        try:
            clouds[camera_id] = read_ascii_ply(path)
            pose = load_camera_pose_json(json.dumps(camera_documents[camera_id]))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"camera {camera_id!r}: {exc}") from exc
        poses[camera_id] = (pose.rotation_matrix, pose.translation_vector)

    result = fuse_multicamera_point_clouds(clouds, poses, voxel_size=voxel_size)
    write_ply(output_path, result.points)
    summary = {
        "distance_unit": distance_unit,
        "camera_count": len(result.cameras),
        "input_points": result.input_points,
        "output_points": result.output_points,
        "voxel_size": result.voxel_size,
        "output_ply": str(output_path),
        "cameras": [
            {
                "camera_id": item.camera_id,
                "input_points": item.input_points,
                "valid_points": item.valid_points,
            }
            for item in result.cameras
        ],
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse at least two XYZRGB PLY clouds into a shared world frame. "
            "All camera-to-world poses must be independently calibrated and use "
            "the same coordinate frame and distance unit."
        )
    )
    parser.add_argument(
        "--cloud", action="append", type=_camera_input, required=True,
        metavar="CAMERA_ID=FILE.ply",
        help="camera cloud input; repeat once per camera",
    )
    parser.add_argument("--poses", type=Path, required=True, help="shared camera-to-world poses JSON")
    parser.add_argument("--output", type=Path, required=True, help="fused output PLY path")
    parser.add_argument(
        "--voxel-size", type=float, default=None,
        help="optional positive world-frame voxel size in the pose distance unit",
    )
    parser.add_argument("--summary", type=Path, default=None, help="optional fusion summary JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = fuse_files(
            args.cloud, args.poses, args.output, args.voxel_size, args.summary
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
