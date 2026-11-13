import numpy as np
import pytest

from blindspot.drone import (
    DroneTelemetry,
    georeference_point_cloud,
    gps_to_local_enu,
    interpolate_telemetry,
    parse_telemetry_csv,
)


def test_parse_telemetry_csv_and_optional_attitude():
    rows = parse_telemetry_csv(
        "timestamp_seconds,latitude_deg,longitude_deg,altitude_m,yaw_deg\n"
        "0,30,78,100,359\n1,30.0001,78.0001,101,1\n"
    )
    assert len(rows) == 2
    assert rows[0].roll_deg == 0.0
    assert rows[1].yaw_deg == 1.0


@pytest.mark.parametrize(
    "content",
    [
        "",
        "timestamp_seconds,latitude_deg,longitude_deg\n0,30,78\n",
        "timestamp_seconds,latitude_deg,longitude_deg,altitude_m\n0,91,78,100\n",
        "timestamp_seconds,latitude_deg,longitude_deg,altitude_m\n0,30,78,100\n0,30,78,100\n",
    ],
)
def test_parse_telemetry_csv_rejects_invalid_data(content):
    with pytest.raises(ValueError):
        parse_telemetry_csv(content)


def test_interpolate_telemetry_uses_short_yaw_path():
    records = [
        DroneTelemetry(0, 30, 78, 100, yaw_deg=359),
        DroneTelemetry(10, 30.001, 78.001, 110, yaw_deg=1),
    ]
    middle = interpolate_telemetry(records, 5)
    assert middle.altitude_m == pytest.approx(105)
    assert abs(middle.yaw_deg) < 1e-6 or abs(middle.yaw_deg - 360) < 1e-6


def test_interpolate_telemetry_rejects_extrapolation():
    records = [DroneTelemetry(0, 30, 78, 100)]
    with pytest.raises(ValueError):
        interpolate_telemetry(records, 1)


def test_gps_to_local_enu_origin_is_zero():
    result = gps_to_local_enu(30, 78, 100, 30, 78, 100)
    assert np.allclose(result, np.zeros(3))


def test_georeference_point_cloud_applies_rotation_and_gps_anchor():
    cloud = np.asarray([[1, 0, 0, 255, 0, 0]], dtype=float)
    telemetry = DroneTelemetry(0, 30, 78, 100)
    rotation = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    result = georeference_point_cloud(
        cloud, telemetry, (30, 78, 100), rotation, (2, 3, 4)
    )
    assert np.allclose(result[0, :3], [2, 4, 4])
    assert np.allclose(result[0, 3:], [255, 0, 0])


def test_georeference_rejects_non_rotation():
    cloud = np.zeros((1, 6), dtype=float)
    with pytest.raises(ValueError):
        georeference_point_cloud(
            cloud, DroneTelemetry(0, 30, 78, 100), (30, 78, 100), np.eye(3) * 2
        )
