import numpy as np
import pytest

from blindspot.camera_rig import estimate_camera_rig_poses


def _edge_from_world_poses(i_pose, j_pose):
    ri, ti = i_pose
    rj, tj = j_pose
    return rj.T @ ri, rj.T @ (ti - tj)


def test_camera_rig_composes_pairwise_extrinsics_into_anchor_frame():
    poses = {
        "cam_a": (np.eye(3), np.array([0.0, 0.0, 0.0])),
        "cam_b": (np.eye(3), np.array([0.3, 0.0, 0.0])),
        "cam_c": (np.eye(3), np.array([0.3, 0.2, 0.0])),
    }
    ab = _edge_from_world_poses(poses["cam_a"], poses["cam_b"])
    bc = _edge_from_world_poses(poses["cam_b"], poses["cam_c"])
    ac = _edge_from_world_poses(poses["cam_a"], poses["cam_c"])
    result = estimate_camera_rig_poses(
        ["cam_a", "cam_b", "cam_c"],
        [("cam_a", "cam_b", *ab), ("cam_b", "cam_c", *bc), ("cam_a", "cam_c", *ac)],
        "cam_a",
    )
    assert np.allclose(result.camera_poses["cam_b"][1], poses["cam_b"][1])
    assert np.allclose(result.camera_poses["cam_c"][1], poses["cam_c"][1])
    assert result.mean_translation_cycle_error == pytest.approx(0.0, abs=1e-9)
    assert result.mean_rotation_cycle_error_degrees == pytest.approx(0.0, abs=1e-6)


def test_camera_rig_rejects_disconnected_graph_and_bad_edges():
    with pytest.raises(ValueError, match="disconnected"):
        estimate_camera_rig_poses(
            ["a", "b", "c"], [("a", "b", np.eye(3), [1, 0, 0])], "a"
        )
    with pytest.raises(ValueError, match="proper finite"):
        estimate_camera_rig_poses(
            ["a", "b"], [("a", "b", np.zeros((3, 3)), [1, 0, 0])], "a"
        )
