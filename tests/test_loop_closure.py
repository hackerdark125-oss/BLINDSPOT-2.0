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
