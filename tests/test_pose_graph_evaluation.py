"""Tests for independent SE(3) loop-closure residual evaluation."""
import numpy as np
import pytest

from blindspot.pose_graph_evaluation import evaluate_pose_graph_constraints
from blindspot.pose_graph_se3 import optimize_pose_graph_se3


def _poses(last_x=3.4):
    return [
        (np.eye(3), np.array([0.0, 0.0, 0.0])),
        (np.eye(3), np.array([1.0, 0.0, 0.0])),
        (np.eye(3), np.array([2.0, 0.0, 0.0])),
        (np.eye(3), np.array([last_x, 0.0, 0.0])),
    ]


def test_loop_closure_residual_is_reported_and_reduced_by_optimisation():
    constraint = (0, 3, np.eye(3), np.array([-3.0, 0.0, 0.0]), 8.0)
    before = evaluate_pose_graph_constraints(_poses(), [constraint])
    optimized = optimize_pose_graph_se3(_poses(), [constraint])
    after = evaluate_pose_graph_constraints(optimized.poses, [constraint])

    assert before["weighted_translation_rmse"] == pytest.approx(0.4)
    assert after["weighted_translation_rmse"] < before["weighted_translation_rmse"]
    assert after["weighted_rotation_rmse_degrees"] < 1e-5
    assert after["constraints_evaluated"] == 1


def test_constraint_evaluator_accepts_exact_pose_measurements():
    poses = _poses(3.0)
    result = evaluate_pose_graph_constraints(
        poses, [(0, 3, np.eye(3), np.array([-3.0, 0.0, 0.0]))]
    )
    assert result["weighted_translation_rmse"] == pytest.approx(0.0)
    assert result["weighted_rotation_rmse_degrees"] == pytest.approx(0.0)


@pytest.mark.parametrize("constraints", [[], [(0, 4, np.eye(3), [0, 0, 0])],
                                         [(1, 0, np.eye(3), [0, 0, 0])],
                                         [(0, 3, np.eye(3), [0, 0, 0], 0)]])
def test_constraint_evaluator_rejects_invalid_edges(constraints):
    with pytest.raises(ValueError):
        evaluate_pose_graph_constraints(_poses(), constraints)
