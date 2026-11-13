import cv2
import numpy as np
import pytest

from blindspot.loop_closure import detect_loop_closure_candidates


def textured_scene(seed=8):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (240, 320), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    for _ in range(70):
        x, y = rng.integers(8, 312), rng.integers(8, 232)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 7)), 255, -1)
    return image


def test_loop_closure_detects_revisited_view_with_geometric_verification():
    first = textured_scene()
    transform = np.float32([[1, 0, 4], [0, 1, 3]])
    revisited = cv2.warpAffine(first, transform, (first.shape[1], first.shape[0]))
    unrelated = textured_scene(seed=91)

    candidates = detect_loop_closure_candidates(
        [first, unrelated, revisited],
        min_frame_gap=2,
        min_matches=10,
        min_inliers=8,
        min_inlier_ratio=0.25,
    )

    assert candidates
    best = candidates[0]
    assert (best.earlier_index, best.later_index) == (0, 2)
    assert best.match_count >= best.inlier_count >= 8
    assert 0.25 <= best.inlier_ratio <= 1.0
    assert best.verification_model in {"fundamental", "homography"}


def test_loop_closure_rejects_adjacent_frames_by_temporal_gap():
    image = textured_scene()
    assert detect_loop_closure_candidates(
        [image, image.copy()], min_frame_gap=2
    ) == []


@pytest.mark.parametrize("frames", [[], [np.zeros((20, 20), dtype=np.uint8)]])
def test_loop_closure_requires_multiple_frames(frames):
    with pytest.raises(ValueError, match="at least two"):
        detect_loop_closure_candidates(frames)


def test_loop_closure_rejects_invalid_frame_and_parameters():
    with pytest.raises(ValueError, match="uint8"):
        detect_loop_closure_candidates(
            [np.zeros((20, 20), dtype=np.float32), np.zeros((20, 20), dtype=np.uint8)]
        )
    with pytest.raises(ValueError, match="min_frame_gap"):
        detect_loop_closure_candidates(
            [np.zeros((20, 20), dtype=np.uint8)] * 2, min_frame_gap=0
        )
    with pytest.raises(ValueError, match="same image dimensions"):
        detect_loop_closure_candidates(
            [np.zeros((20, 20), dtype=np.uint8), np.zeros((21, 20), dtype=np.uint8)]
        )


from blindspot.loop_closure import verify_metric_loop_closure


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_metric_loop_closure_recovers_transform_with_outlier_matches():
    rng = np.random.default_rng(42)
    source = rng.uniform(-2.0, 2.0, size=(80, 3))
    expected_rotation = _rotation_z(0.12)
    expected_translation = np.array([0.3, -0.2, 0.1])
    target = (expected_rotation @ source.T).T + expected_translation
    target += rng.normal(scale=0.002, size=target.shape)
    outliers = rng.choice(len(source), size=20, replace=False)
    target[outliers] = rng.uniform(-3.0, 3.0, size=(20, 3))

    result = verify_metric_loop_closure(
        source, target, threshold=0.015, min_inliers=45,
        min_inlier_ratio=0.5, max_iterations=500, random_seed=3,
    )
    assert result.verified
    assert result.inlier_count >= 58
    assert result.inlier_ratio >= 0.7
    assert result.rmse < 0.01
    assert np.allclose(result.rotation, expected_rotation, atol=0.003)
    assert np.allclose(result.translation, expected_translation, atol=0.005)


def test_metric_loop_closure_rejects_inconsistent_candidate():
    rng = np.random.default_rng(7)
    source = rng.normal(size=(40, 3))
    target = rng.normal(size=(40, 3))
    result = verify_metric_loop_closure(
        source, target, threshold=0.001, min_inliers=20,
        min_inlier_ratio=0.5, max_iterations=100,
    )
    assert not result.verified


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0}, {"min_inliers": 2},
    {"min_inlier_ratio": 0}, {"max_iterations": 0},
])
def test_metric_loop_closure_rejects_invalid_configuration(kwargs):
    cloud = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    with pytest.raises(ValueError):
        verify_metric_loop_closure(cloud, cloud, **kwargs)


def test_metric_loop_closure_rejects_bad_correspondences():
    with pytest.raises(ValueError, match="matching"):
        verify_metric_loop_closure(np.ones((5, 3)), np.ones((6, 3)))
    line = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    with pytest.raises(ValueError, match="collinear"):
        verify_metric_loop_closure(line, line, min_inliers=3)


def test_metric_loop_closure_is_reproducible_with_fixed_seed():
    rng = np.random.default_rng(22)
    source = rng.normal(size=(50, 3))
    target = source + np.array([0.1, -0.2, 0.05])
    target[:10] = rng.normal(size=(10, 3))
    first = verify_metric_loop_closure(
        source, target, threshold=0.01, min_inliers=20,
        min_inlier_ratio=0.5, random_seed=11,
    )
    second = verify_metric_loop_closure(
        source, target, threshold=0.01, min_inliers=20,
        min_inlier_ratio=0.5, random_seed=11,
    )
    assert first.inlier_count == second.inlier_count
    assert np.array_equal(first.inlier_mask, second.inlier_mask)
    assert np.allclose(first.rotation, second.rotation)
    assert np.allclose(first.translation, second.translation)
