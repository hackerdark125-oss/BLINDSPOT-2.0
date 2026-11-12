"""Chessboard-based single-camera intrinsic calibration utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    rms_reprojection_error: float
    image_width: int
    image_height: int
    board_columns: int
    board_rows: int
    square_size: float
    accepted_views: int
    rejected_views: int
    mean_view_reprojection_error_px: float = 0.0
    max_view_reprojection_error_px: float = 0.0

    def to_json(self) -> str:
        """Serialize calibration parameters in a portable JSON format."""
        return json.dumps(asdict(self), indent=2)


def calibrate_camera(
    images: Sequence[np.ndarray],
    board_size: tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    min_views: int = 3,
) -> CameraCalibration:
    """Estimate camera intrinsics from multiple chessboard images.

    board_size is (inner_corner_columns, inner_corner_rows). square_size is
    the real-world distance between adjacent corners, in any consistent unit.
    """
    if (not isinstance(board_size, tuple) or len(board_size) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 2 for v in board_size)):
        raise ValueError("board_size must be a tuple of two integers >= 2")
    if not np.isfinite(square_size) or square_size <= 0:
        raise ValueError("square_size must be a positive finite number")
    if isinstance(min_views, bool) or not isinstance(min_views, int) or min_views < 3:
        raise ValueError("min_views must be an integer of at least 3")
    if not images:
        raise ValueError("at least three chessboard images are required")

    columns, rows = board_size
    object_template = np.zeros((rows * columns, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_template *= float(square_size)

    object_points, image_points = [], []
    image_size = None
    rejected = 0
    pattern = (columns, rows)

    for image in images:
        if not isinstance(image, np.ndarray) or image.size == 0:
            rejected += 1
            continue
        if image.dtype != np.uint8:
            rejected += 1
            continue
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            rejected += 1
            continue

        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            rejected += 1
            continue

        found, corners = cv2.findChessboardCorners(
            gray, pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            rejected += 1
            continue
        refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        object_points.append(object_template.copy())
        image_points.append(refined)

    if len(object_points) < min_views:
        raise ValueError(
            f"Need at least {min_views} usable chessboard views; "
            f"found {len(object_points)} and rejected {rejected}."
        )

    def fit(points_3d, points_2d):
        return cv2.calibrateCamera(
            points_3d, points_2d, image_size, None, None
        )

    rms, camera_matrix, distortion, rotations, translations = fit(
        object_points, image_points
    )
    if not np.isfinite(rms) or not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise RuntimeError("Camera calibration returned non-finite parameters")

    def per_view_errors(points_3d, points_2d, matrix, coefficients, rvecs, tvecs):
        errors = []
        for object_view, image_view, rotation_vector, translation_vector in zip(
            points_3d, points_2d, rvecs, tvecs
        ):
            projected, _ = cv2.projectPoints(
                object_view, rotation_vector, translation_vector,
                matrix, coefficients,
            )
            residual = projected.reshape(-1, 2) - image_view.reshape(-1, 2)
            errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
        return np.asarray(errors, dtype=np.float64)

    initial_errors = per_view_errors(
        object_points, image_points, camera_matrix, distortion, rotations, translations
    )
    median_error = float(np.median(initial_errors))
    median_absolute_deviation = float(np.median(np.abs(initial_errors - median_error)))
    outlier_limit = median_error + max(3.5 * 1.4826 * median_absolute_deviation, 0.5)
    keep = initial_errors <= outlier_limit
    outlier_count = int(np.count_nonzero(~keep))

    # Refit once after removing grossly inconsistent chessboard views. Never
    # discard so many views that the caller's requested minimum is violated.
    if outlier_count and int(np.count_nonzero(keep)) >= min_views:
        object_points = [points for points, accepted in zip(object_points, keep) if accepted]
        image_points = [points for points, accepted in zip(image_points, keep) if accepted]
        rms, camera_matrix, distortion, rotations, translations = fit(
            object_points, image_points
        )
        if not np.isfinite(rms) or not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
            raise RuntimeError("Camera calibration returned non-finite parameters")

    final_errors = per_view_errors(
        object_points, image_points, camera_matrix, distortion, rotations, translations
    )
    if not np.isfinite(final_errors).all():
        raise RuntimeError("Camera calibration produced non-finite per-view reprojection errors")

    return CameraCalibration(
        camera_matrix=camera_matrix.tolist(),
        distortion_coefficients=distortion.reshape(-1).tolist(),
        rms_reprojection_error=float(rms),
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        board_columns=columns,
        board_rows=rows,
        square_size=float(square_size),
        accepted_views=len(object_points),
        rejected_views=rejected + (outlier_count if len(object_points) < len(initial_errors) else 0),
        mean_view_reprojection_error_px=float(np.mean(final_errors)),
        max_view_reprojection_error_px=float(np.max(final_errors)),
    )


def undistort_frame(frame: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    """Correct lens distortion using previously estimated camera parameters."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty NumPy image")
    if (frame.shape[1], frame.shape[0]) != (
        calibration.image_width, calibration.image_height
    ):
        raise ValueError("frame dimensions must match calibration image dimensions")
    matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    distortion = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    if matrix.shape != (3, 3) or distortion.size < 4:
        raise ValueError("calibration parameters are invalid")
    return cv2.undistort(frame, matrix, distortion)


def load_calibration_json(content: str) -> CameraCalibration:
    """Load and validate a calibration object from JSON text."""
    try:
        data = json.loads(content)
        calibration = CameraCalibration(**data)
        matrix = np.asarray(calibration.camera_matrix, dtype=float)
        distortion = np.asarray(calibration.distortion_coefficients, dtype=float)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid calibration JSON") from exc
    if (matrix.shape != (3, 3) or distortion.size < 4
            or not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
            or not np.isfinite(calibration.rms_reprojection_error)
            or not np.isfinite(calibration.mean_view_reprojection_error_px)
            or not np.isfinite(calibration.max_view_reprojection_error_px)
            or calibration.mean_view_reprojection_error_px < 0
            or calibration.max_view_reprojection_error_px < 0
            or calibration.image_width < 1 or calibration.image_height < 1):
        raise ValueError("calibration JSON contains invalid camera parameters")
    return calibration
