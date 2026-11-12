import numpy as np
import pytest

from blindspot.trajectory_evaluation import evaluate_trajectory


def _trajectory(xs):
    poses = []
    for x in xs:
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append(pose)
    return poses


def test_identical_trajectory_has_zero_error():
    trajectory = _trajectory([0.0, 1.0, 2.0, 3.0])
    result = evaluate_trajectory(trajectory, trajectory)
    assert result["ate_rmse"] == pytest.approx(0.0)
    assert result["rpe_translation_rmse"] == pytest.approx(0.0)
    assert result["rpe_rotation_rmse_degrees"] == pytest.approx(0.0)


def test_first_pose_alignment_removes_only_global_frame_offset():
    estimate = _trajectory([10.0, 11.1, 12.2])
    reference = _trajectory([0.0, 1.0, 2.0])
    result = evaluate_trajectory(estimate, reference)
    assert result["ate_rmse"] == pytest.approx(np.sqrt((0.0 + 0.01 + 0.04) / 3))
    assert result["rpe_translation_rmse"] == pytest.approx(0.1)


def test_trajectory_evaluation_rejects_length_and_rotation_errors():
    with pytest.raises(ValueError, match="equal lengths"):
        evaluate_trajectory(_trajectory([0, 1]), _trajectory([0, 1, 2]))
    bad = _trajectory([0, 1])
    bad[1][0, 0] = 2.0
    with pytest.raises(ValueError, match="proper orthonormal"):
        evaluate_trajectory(bad, _trajectory([0, 1]))
