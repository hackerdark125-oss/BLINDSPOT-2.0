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
    if float(np.linalg.norm(translation)) <= 1e-9:
        raise ValueError("stereo calibration produced a near-zero camera baseline")

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


@dataclass(frozen=True)
class StereoDepthResult:
    rectified_a: np.ndarray
    rectified_b: np.ndarray
    disparity_pixels: np.ndarray
    depth_map: np.ndarray
    valid_mask: np.ndarray
    translation_unit: str
    rectified_camera_matrix: np.ndarray | None = None
    rectification_rotation: np.ndarray | None = None


def load_stereo_calibration_json(content: str) -> StereoCalibration:
    """Load and validate relative camera pose from JSON text."""
    try:
        data = json.loads(content)
        calibration = StereoCalibration(**data)
        rotation = np.asarray(calibration.rotation_matrix, dtype=float)
        translation = np.asarray(calibration.translation_vector, dtype=float)
        essential = np.asarray(calibration.essential_matrix, dtype=float)
        fundamental = np.asarray(calibration.fundamental_matrix, dtype=float)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid stereo calibration JSON") from exc
    if (
        rotation.shape != (3, 3) or translation.size != 3
        or essential.shape != (3, 3) or fundamental.shape != (3, 3)
        or not all(np.isfinite(x).all() for x in (rotation, translation, essential, fundamental))
        or calibration.image_width < 1 or calibration.image_height < 1
        or not calibration.translation_unit.strip()
        or float(np.linalg.norm(translation)) <= 1e-9
    ):
        raise ValueError("stereo calibration JSON contains invalid parameters")
    return calibration


def rectify_and_compute_depth(
    image_a: np.ndarray,
    image_b: np.ndarray,
    intrinsics_a: CameraCalibration,
    intrinsics_b: CameraCalibration,
    stereo: StereoCalibration,
    num_disparities: int = 128,
    block_size: int = 5,
    left_right_consistency_threshold: float = 1.5,
) -> StereoDepthResult:
    """Rectify a calibrated stereo pair and estimate disparity/depth with SGBM.

    Depth values use the translation unit stored in the stereo calibration
    (which should match the chessboard square-size unit).
    """
    for label, image in (("image_a", image_a), ("image_b", image_b)):
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(f"{label} must be a non-empty NumPy image")
        if image.dtype != np.uint8:
            raise ValueError(f"{label} must use uint8 pixel values")
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (3, 4)):
            raise ValueError(f"{label} must be grayscale, BGR, or BGRA")

    expected_size = (stereo.image_width, stereo.image_height)
    if (image_a.shape[1], image_a.shape[0]) != expected_size or (image_b.shape[1], image_b.shape[0]) != expected_size:
        raise ValueError("both images must match the stereo calibration resolution")
    if (intrinsics_a.image_width, intrinsics_a.image_height) != expected_size:
        raise ValueError("camera A intrinsics do not match stereo calibration resolution")
    if (intrinsics_b.image_width, intrinsics_b.image_height) != expected_size:
        raise ValueError("camera B intrinsics do not match stereo calibration resolution")
    if isinstance(num_disparities, bool) or not isinstance(num_disparities, int) or num_disparities < 16 or num_disparities % 16:
        raise ValueError("num_disparities must be a positive multiple of 16")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3")
    if not np.isfinite(left_right_consistency_threshold) or left_right_consistency_threshold <= 0:
        raise ValueError("left_right_consistency_threshold must be positive and finite")

    matrix_a = np.asarray(intrinsics_a.camera_matrix, dtype=np.float64)
    dist_a = np.asarray(intrinsics_a.distortion_coefficients, dtype=np.float64)
    matrix_b = np.asarray(intrinsics_b.camera_matrix, dtype=np.float64)
    dist_b = np.asarray(intrinsics_b.distortion_coefficients, dtype=np.float64)
    rotation = np.asarray(stereo.rotation_matrix, dtype=np.float64)
    translation = np.asarray(stereo.translation_vector, dtype=np.float64).reshape(3, 1)
    if matrix_a.shape != (3, 3) or matrix_b.shape != (3, 3) or rotation.shape != (3, 3):
        raise ValueError("camera or stereo matrices have invalid dimensions")
    if float(np.linalg.norm(translation)) <= 1e-9:
        raise ValueError("stereo camera baseline must be non-zero")

    size = expected_size
    rect_a, rect_b, projection_a, projection_b, q_matrix, _, _ = cv2.stereoRectify(
        matrix_a, dist_a, matrix_b, dist_b, size, rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
    )
    map_ax, map_ay = cv2.initUndistortRectifyMap(
        matrix_a, dist_a, rect_a, projection_a, size, cv2.CV_32FC1
    )
    map_bx, map_by = cv2.initUndistortRectifyMap(
        matrix_b, dist_b, rect_b, projection_b, size, cv2.CV_32FC1
    )
    rectified_a = cv2.remap(image_a, map_ax, map_ay, cv2.INTER_LINEAR)
    rectified_b = cv2.remap(image_b, map_bx, map_by, cv2.INTER_LINEAR)
    gray_a = cv2.cvtColor(rectified_a, cv2.COLOR_BGR2GRAY) if rectified_a.ndim == 3 else rectified_a
    gray_b = cv2.cvtColor(rectified_b, cv2.COLOR_BGR2GRAY) if rectified_b.ndim == 3 else rectified_b

    matcher_options = dict(
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    matcher_left = cv2.StereoSGBM_create(minDisparity=0, **matcher_options)
    matcher_right = cv2.StereoSGBM_create(
        minDisparity=-num_disparities, **matcher_options
    )
    disparity = matcher_left.compute(gray_a, gray_b).astype(np.float32) / 16.0
    disparity_right = matcher_right.compute(gray_b, gray_a).astype(np.float32) / 16.0

    # Explicit left-right consistency rejects occlusions and many mismatches.
    # For left pixel x with disparity d, its corresponding right pixel is x-d.
    height, width = disparity.shape
    x_coordinates = np.broadcast_to(np.arange(width, dtype=np.int32), (height, width))
    right_x = np.rint(x_coordinates - disparity).astype(np.int32)
    in_bounds = (right_x >= 0) & (right_x < width)
    sampled_right = np.full(disparity.shape, np.nan, dtype=np.float32)
    rows = np.broadcast_to(np.arange(height, dtype=np.int32)[:, None], disparity.shape)
    sampled_right[in_bounds] = disparity_right[rows[in_bounds], right_x[in_bounds]]
    valid = (
        np.isfinite(disparity)
        & (disparity > 0)
        & np.isfinite(sampled_right)
        & (sampled_right < 0)
        & (np.abs(disparity + sampled_right) <= left_right_consistency_threshold)
    )
    points_3d = cv2.reprojectImageTo3D(disparity, q_matrix)
    depth = points_3d[:, :, 2].astype(np.float32)
    valid &= np.isfinite(depth) & (depth > 0)
    depth[~valid] = np.nan

    return StereoDepthResult(
        rectified_a, rectified_b, disparity, depth, valid, stereo.translation_unit,
        projection_a[:3, :3], rect_a,
    )
