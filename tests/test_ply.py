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
    (b"ply\nformat binary_little_endian 1.1\nend_header\n", "unsupported PLY format"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n0\n", "properties"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 0 0\n", "incorrect field count"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 nan 0 0 0\n", "finite"),
    (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 0 0 300\n", "RGB channels"),
])
def test_read_ascii_ply_rejects_invalid_inputs(content, message):
    with pytest.raises(ValueError, match=message):
        read_ascii_ply(content)



@pytest.mark.parametrize("encoding, byte_order", [
    ("binary_little_endian", "<"),
    ("binary_big_endian", ">"),
])
def test_read_ply_supports_binary_and_reordered_extra_properties(encoding, byte_order):
    import struct

    header = (
        f"ply\nformat {encoding} 1.0\nelement vertex 2\n"
        "property uchar blue\nproperty float z\nproperty uchar red\n"
        "property float x\nproperty uchar green\nproperty float y\n"
        "property float intensity\nend_header\n"
    ).encode("ascii")
    row = byte_order + "BfBfBff"
    payload = struct.pack(row, 30, 3.0, 10, 1.0, 20, 2.0, 0.5)
    payload += struct.pack(row, 60, 6.0, 40, 4.0, 50, 5.0, 0.7)
    points = read_ascii_ply(header + payload)
    assert np.allclose(points, [[1, 2, 3, 10, 20, 30], [4, 5, 6, 40, 50, 60]])


def test_read_ply_rejects_truncated_binary_vertex_data():
    content = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n\x00\x00"
    )
    with pytest.raises(ValueError, match="fewer vertex bytes"):
        read_ascii_ply(content)


def test_read_ascii_ply_ignores_standard_trailing_face_element():
    content = (
        b"ply\nformat ascii 1.0\nelement vertex 3\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"element face 1\nproperty list uchar int vertex_indices\n"
        b"end_header\n0 0 0 1 2 3\n1 0 0 4 5 6\n0 1 0 7 8 9\n"
        b"3 0 1 2\n"
    )
    points = read_ascii_ply(content)
    assert points.shape == (3, 6)
    assert np.allclose(points[2], [0, 1, 0, 7, 8, 9])
