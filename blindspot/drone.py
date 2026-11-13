"""Drone telemetry and georeferencing utilities for BLINDSPOT AERIAL.

Coordinates:
- GPS origin and telemetry altitude must use the same vertical datum.
- Point-cloud XYZ values and altitude are metres.
- camera_to_enu_rotation maps camera-local XYZ into local East/North/Up.
- Camera-to-drone lever-arm offsets are expressed in local ENU metres.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class DroneTelemetry:
    timestamp_seconds: float
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


_REQUIRED_COLUMNS = {
    "timestamp_seconds", "latitude_deg", "longitude_deg", "altitude_m"
}


def parse_telemetry_csv(content: str) -> list[DroneTelemetry]:
    """Parse timestamped GPS/attitude telemetry CSV with strict validation."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("telemetry CSV content must be non-empty text")
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise ValueError("telemetry CSV must include a header row")
    missing = _REQUIRED_COLUMNS.difference(name.strip() for name in reader.fieldnames)
    if missing:
        raise ValueError(f"telemetry CSV is missing columns: {', '.join(sorted(missing))}")

    records: list[DroneTelemetry] = []
    for line_number, row in enumerate(reader, start=2):
        try:
            values = {
                key: float(row.get(key, "") or 0.0)
                for key in (
                    "timestamp_seconds", "latitude_deg", "longitude_deg",
                    "altitude_m", "roll_deg", "pitch_deg", "yaw_deg",
                )
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric telemetry value on CSV line {line_number}") from exc
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite telemetry value on CSV line {line_number}")
        if not -90.0 <= values["latitude_deg"] <= 90.0:
            raise ValueError(f"latitude out of range on CSV line {line_number}")
        if not -180.0 <= values["longitude_deg"] <= 180.0:
            raise ValueError(f"longitude out of range on CSV line {line_number}")
        records.append(DroneTelemetry(**values))

    if not records:
        raise ValueError("telemetry CSV contains no data rows")
    timestamps = [item.timestamp_seconds for item in records]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("telemetry timestamps must be strictly increasing")
    return records


def interpolate_telemetry(
    records: Iterable[DroneTelemetry], timestamp_seconds: float
) -> DroneTelemetry:
    """Interpolate GPS/attitude at a video-relative timestamp; reject extrapolation."""
    items = list(records)
    if not items:
        raise ValueError("at least one telemetry record is required")
    if not np.isfinite(timestamp_seconds):
        raise ValueError("timestamp_seconds must be finite")
    times = np.asarray([item.timestamp_seconds for item in items], dtype=float)
    if timestamp_seconds < times[0] or timestamp_seconds > times[-1]:
        raise ValueError("requested timestamp is outside the telemetry time range")
    if len(items) == 1:
        return items[0]

    def interp(attribute: str) -> float:
        values = np.asarray([getattr(item, attribute) for item in items], dtype=float)
        if attribute == "yaw_deg":
            values = np.unwrap(np.deg2rad(values))
            return float(np.rad2deg(np.interp(timestamp_seconds, times, values)))
        return float(np.interp(timestamp_seconds, times, values))

    return DroneTelemetry(
        timestamp_seconds=float(timestamp_seconds),
        latitude_deg=interp("latitude_deg"),
        longitude_deg=interp("longitude_deg"),
        altitude_m=interp("altitude_m"),
        roll_deg=interp("roll_deg"),
        pitch_deg=interp("pitch_deg"),
        yaw_deg=interp("yaw_deg"),
    )


def gps_to_local_enu(
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
    origin_latitude_deg: float,
    origin_longitude_deg: float,
    origin_altitude_m: float,
) -> np.ndarray:
    """Convert nearby GPS coordinates to a local East/North/Up metre frame."""
    values = (
        latitude_deg, longitude_deg, altitude_m,
        origin_latitude_deg, origin_longitude_deg, origin_altitude_m,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("GPS coordinates and altitudes must be finite")
    if not (-90 <= latitude_deg <= 90 and -90 <= origin_latitude_deg <= 90):
        raise ValueError("latitude must be between -90 and 90 degrees")
    if not (-180 <= longitude_deg <= 180 and -180 <= origin_longitude_deg <= 180):
        raise ValueError("longitude must be between -180 and 180 degrees")
    radius_m = 6_378_137.0
    lat = np.deg2rad(latitude_deg)
    origin_lat = np.deg2rad(origin_latitude_deg)
    delta_lat = lat - origin_lat
    delta_lon = np.deg2rad(longitude_deg - origin_longitude_deg)
    east = radius_m * delta_lon * np.cos((lat + origin_lat) / 2.0)
    north = radius_m * delta_lat
    up = altitude_m - origin_altitude_m
    return np.asarray([east, north, up], dtype=np.float64)


def georeference_point_cloud(
    camera_point_cloud: np.ndarray,
    telemetry: DroneTelemetry,
    origin_gps: tuple[float, float, float],
    camera_to_enu_rotation: np.ndarray | list[list[float]],
    camera_offset_enu_m: np.ndarray | list[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Transform a camera-local metric XYZRGB cloud into local ENU coordinates.

    The supplied rotation must be calibrated/derived for the actual camera and
    gimbal orientation. This function deliberately does not guess camera pose
    from aircraft attitude because camera mounting and gimbal angles vary.
    """
    cloud = np.asarray(camera_point_cloud, dtype=np.float64)
    rotation = np.asarray(camera_to_enu_rotation, dtype=np.float64)
    offset = np.asarray(camera_offset_enu_m, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 6 or not np.isfinite(cloud).all():
        raise ValueError("camera_point_cloud must be a finite Nx6 XYZRGB array")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("camera_to_enu_rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("camera_to_enu_rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("camera_to_enu_rotation must be a proper rotation")
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("camera_offset_enu_m must be a finite 3-vector")
    if len(origin_gps) != 3:
        raise ValueError("origin_gps must be (latitude_deg, longitude_deg, altitude_m)")

    drone_position = gps_to_local_enu(
        telemetry.latitude_deg, telemetry.longitude_deg, telemetry.altitude_m,
        float(origin_gps[0]), float(origin_gps[1]), float(origin_gps[2]),
    )
    camera_position = drone_position + offset
    world_xyz = (rotation @ cloud[:, :3].T).T + camera_position
    return np.column_stack((world_xyz, cloud[:, 3:6])).astype(np.float32)
