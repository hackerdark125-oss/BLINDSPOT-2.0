"""Paired chessboard stereo calibration for two fixed cameras."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Sequence

import cv2
import numpy as np

from blindspot.calibration import CameraCalibration


@dataclass(frozen=True)
class StereoCalibration:
    rotation_matrix: list[list[float]]
    translation_vector: list[float]
    essential_matrix: list[list[float]]
    fundamental_matrix: list[list[float]]
    rms_reprojection_error: float
    image_width: int
    image_height: int
    accepted_pairs: int
    rejected_pairs: int
    translation_unit: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def calibrate_stereo_cameras(
    images_a: Sequence[np.ndarray],
    images_b: Sequence[np.ndarray],
    intrinsics_a: CameraCalibration,
    intrinsics_b: CameraCalibration,
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    min_pairs: int = 3,
    translation_unit: str = "same as square_size",
) -> StereoCalibration:
    """Estimate relative pose from synchronized paired chessboard images.

    Pair i in images_a must depict the same rigid chessboard pose/time as
    pair i in images_b. Intrinsic calibrations must match each image size.
    """
    if len(images_a) != len(images_b):
        raise ValueError("camera A and camera B must have the same number of paired images")
    if isinstance(board_size, bool) or not isinstance(board_size, tuple) or len(board_size) != 2:
        raise ValueError("board_size must be a tuple of two integers >= 2")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in board_size):
        raise ValueError("board_size must be a tuple of two integers >= 2")
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be positive and finite")
    if isinstance(min_pairs, bool) or not isinstance(min_pairs, int) or min_pairs < 3:
        raise ValueError("min_pairs must be an integer of at least 3")
    if not translation_unit.strip():
        raise ValueError("translation_unit must not be empty")
    if not images_a:
        raise ValueError("at least three synchronized image pairs are required")

    size_a = (intrinsics_a.image_width, intrinsics_a.image_height)
    size_b = (intrinsics_b.image_width, intrinsics_b.image_height)
    if size_a != size_b:
        raise ValueError("both cameras must use the same image resolution for this calibration")
    if intrinsics_a.board_columns != board_size[0] or intrinsics_a.board_rows != board_size[1]:
        raise ValueError("camera A intrinsics were calibrated with a different chessboard size")
    if intrinsics_b.board_columns != board_size[0] or intrinsics_b.board_rows != board_size[1]:
        raise ValueError("camera B intrinsics were calibrated with a different chessboard size")
    if not np.isclose(intrinsics_a.square_size, square_size) or not np.isclose(intrinsics_b.square_size, square_size):
        raise ValueError("intrinsic calibrations and stereo calibration must use the same square size")

    columns, rows = board_size
    object_template = np.zeros((rows * columns, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_template *= float(square_size)
    object_points, points_a, points_b = [], [], []
    rejected = 0
    pattern = (columns, rows)

    for image_a, image_b in zip(images_a, images_b):
        if not isinstance(image_a, np.ndarray) or not isinstance(image_b, np.ndarray):
            rejected += 1
            continue
        if image_a.dtype != np.uint8 or image_b.dtype != np.uint8:
            rejected += 1
            continue
        if image_a.ndim == 2:
            gray_a = image_a
        elif image_a.ndim == 3 and image_a.shape[2] in (3, 4):
            gray_a = cv2.cvtColor(image_a[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            rejected += 1
            continue
        if image_b.ndim == 2:
            gray_b = image_b
        elif image_b.ndim == 3 and image_b.shape[2] in (3, 4):
            gray_b = cv2.cvtColor(image_b[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            rejected += 1
            continue
        if (gray_a.shape[1], gray_a.shape[0]) != size_a or (gray_b.shape[1], gray_b.shape[0]) != size_b:
            rejected += 1
            continue

        found_a, corners_a = cv2.findChessboardCorners(
            gray_a, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        found_b, corners_b = cv2.findChessboardCorners(
            gray_b, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not found_a or not found_b:
            rejected += 1
            continue
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_a = cv2.cornerSubPix(gray_a, corners_a, (11, 11), (-1, -1), criteria)
        corners_b = cv2.cornerSubPix(gray_b, corners_b, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        points_a.append(corners_a)
        points_b.append(corners_b)

    if len(object_points) < min_pairs:
        raise ValueError(
            f"Need at least {min_pairs} usable synchronized pairs; "
            f"found {len(object_points)} and rejected {rejected}."
        )

    matrix_a = np.asarray(intrinsics_a.camera_matrix, dtype=np.float64)
    dist_a = np.asarray(intrinsics_a.distortion_coefficients, dtype=np.float64)
    matrix_b = np.asarray(intrinsics_b.camera_matrix, dtype=np.float64)
    dist_b = np.asarray(intrinsics_b.distortion_coefficients, dtype=np.float64)
    if matrix_a.shape != (3, 3) or matrix_b.shape != (3, 3) or dist_a.size < 4 or dist_b.size < 4:
        raise ValueError("camera intrinsic parameters are invalid")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    rms, _, _, _, _, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points, points_a, points_b,
        matrix_a, dist_a, matrix_b, dist_b,
        size_a, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC,
    )
    arrays = (rotation, translation, essential, fundamental)
    if not np.isfinite(rms) or any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError("stereo calibration returned non-finite parameters")

    return StereoCalibration(
        rotation_matrix=rotation.tolist(),
        translation_vector=translation.reshape(-1).tolist(),
        essential_matrix=essential.tolist(),
        fundamental_matrix=fundamental.tolist(),
        rms_reprojection_error=float(rms),
        image_width=size_a[0],
        image_height=size_a[1],
        accepted_pairs=len(object_points),
        rejected_pairs=rejected,
        translation_unit=translation_unit,
    )
