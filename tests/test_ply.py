import numpy as np
import pytest

from blindspot.ply import read_ascii_ply


def test_read_ascii_ply_round_trips_expected_xyzrgb_rows(tmp_path):
    content = (
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n1 2 3 10 20 30\n4 5 6 40 50 60\n"
    )
    path = tmp_path / "cloud.ply"
    path.write_text(content, encoding="ascii")
    points = read_ascii_ply(path)
    assert points.shape == (2, 6)
    assert np.allclose(points, [[1, 2, 3, 10, 20, 30], [4, 5, 6, 40, 50, 60]])


def test_read_ascii_ply_supports_bytes_and_empty_cloud():
    content = (
        b"ply\nformat ascii 1.0\nelement vertex 0\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    assert read_ascii_ply(content).shape == (0, 6)


@pytest.mark.parametrize("content, message", [
    (b"not ply", "not a PLY"),
    (b"ply\nformat binary_little_endian 1.0\n", "ASCII PLY"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n", "properties"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 0 0\n", "six values"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 nan 0 0 0\n", "finite"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 0 0 300\n", "RGB channels"),
])
def test_read_ascii_ply_rejects_invalid_inputs(content, message):
    with pytest.raises(ValueError, match=message):
        read_ascii_ply(content)
