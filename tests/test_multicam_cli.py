import json

import numpy as np
import pytest

from blindspot.reconstruction import write_ply
from scripts.fuse_multicamera import _camera_input, fuse_files


def _pose_document():
    return {
        "distance_unit": "meters",
        "cameras": {
            "left": {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [0, 0, 0],
            },
            "right": {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_vector": [1, 0, 0],
            },
        },
    }


def test_camera_input_parses_id_and_path():
    camera_id, path = _camera_input("front=clouds/front camera.ply")
    assert camera_id == "front"
    assert str(path) == "clouds/front camera.ply"


@pytest.mark.parametrize("value", ["missing-separator.ply", "=cloud.ply", "camera="])
def test_camera_input_rejects_invalid_values(value):
    with pytest.raises(Exception):
        _camera_input(value)


def test_fuse_files_writes_cloud_and_summary(tmp_path):
    left = tmp_path / "left.ply"
    right = tmp_path / "right.ply"
    write_ply(left, np.array([[0, 0, 2, 10, 20, 30]], dtype=float))
    write_ply(right, np.array([[0, 0, 2, 40, 50, 60]], dtype=float))
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps(_pose_document()), encoding="utf-8")
    output = tmp_path / "out" / "fused.ply"
    summary_path = tmp_path / "reports" / "summary.json"

    summary = fuse_files(
        [("left", left), ("right", right)],
        poses,
        output,
        voxel_size=0.1,
        summary_path=summary_path,
    )

    assert output.is_file()
    assert summary_path.is_file()
    assert summary["distance_unit"] == "meters"
    assert summary["camera_count"] == 2
    assert summary["input_points"] == 2
    assert summary["output_points"] == 2
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary


def test_fuse_files_rejects_camera_pose_mismatch(tmp_path):
    cloud = tmp_path / "cloud.ply"
    write_ply(cloud, np.array([[0, 0, 2, 10, 20, 30]], dtype=float))
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps(_pose_document()), encoding="utf-8")
    with pytest.raises(ValueError, match="must match pose JSON"):
        fuse_files([("left", cloud), ("unknown", cloud)], poses, tmp_path / "out.ply")


def test_fuse_files_rejects_invalid_camera_transform(tmp_path):
    cloud = tmp_path / "cloud.ply"
    write_ply(cloud, np.array([[0, 0, 2, 10, 20, 30]], dtype=float))
    document = _pose_document()
    document["cameras"]["right"]["rotation_matrix"] = np.zeros((3, 3)).tolist()
    poses = tmp_path / "poses.json"
    poses.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="camera 'right'"):
        fuse_files([("left", cloud), ("right", cloud)], poses, tmp_path / "out.ply")
