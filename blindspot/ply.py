"""Strict XYZRGB PLY reader supporting ASCII and binary scalar vertex data."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_bytes(source: str | Path | bytes | bytearray | BinaryIO) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        raw = source.read()
        return raw.encode("ascii") if isinstance(raw, str) else bytes(raw)
    return Path(source).read_bytes()


def read_ascii_ply(source: str | Path | bytes | bytearray | BinaryIO) -> np.ndarray:
    """Read a PLY vertex cloud and return float32 Nx6 points in XYZRGB order.

    Despite the historical function name, ASCII 1.0, binary-little-endian 1.0,
    and binary-big-endian 1.0 are supported. Vertex scalar properties may be
    reordered or include additional fields; x/y/z/red/green/blue are required.
    List-valued vertex properties and preceding non-vertex elements are
    rejected explicitly. Colour channels must be integral values in [0, 255].
    """
    raw = _read_bytes(source)
    if not raw.startswith(b"ply\n") and not raw.startswith(b"ply\r\n"):
        raise ValueError("file is not a PLY file")
    marker = raw.find(b"end_header")
    if marker < 0:
        raise ValueError("PLY header is missing end_header")
    newline = raw.find(b"\n", marker)
    if newline < 0:
        raise ValueError("PLY header must end with a newline")
    header_bytes = raw[:newline + 1]
    try:
        header = header_bytes.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("PLY header must be ASCII") from exc

    if len(header) < 3 or header[0].strip() != "ply":
        raise ValueError("file is not a PLY file")
    format_parts = header[1].split()
    if len(format_parts) != 3 or format_parts[0] != "format" or format_parts[2] != "1.0":
        raise ValueError("unsupported PLY format or version")
    formats = {
        "ascii": None,
        "binary_little_endian": "<",
        "binary_big_endian": ">",
    }
    if format_parts[1] not in formats:
        raise ValueError("unsupported PLY encoding")
    byte_order = formats[format_parts[1]]

    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    current_name: str | None = None
    current_count = 0
    current_properties: list[tuple[str, str]] = []
    found_end = False
    for raw_line in header[2:]:
        parts = raw_line.strip().split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "element":
            if len(parts) != 3:
                raise ValueError("invalid PLY element declaration")
            if current_name is not None:
                elements.append((current_name, current_count, current_properties))
            current_name = parts[1]
            try:
                current_count = int(parts[2])
            except ValueError as exc:
                raise ValueError("invalid PLY element count") from exc
            if current_count < 0:
                raise ValueError("PLY element count must be non-negative")
            current_properties = []
        elif parts[0] == "property":
            if current_name is None or len(parts) != 3 or parts[1] == "list":
                raise ValueError("list-valued or malformed PLY properties are unsupported")
            if parts[1] not in _PLY_TYPES:
                raise ValueError(f"unsupported PLY scalar type: {parts[1]}")
            current_properties.append((parts[2], _PLY_TYPES[parts[1]]))
        elif parts[0] == "end_header":
            found_end = True
            break
        else:
            raise ValueError(f"unsupported PLY header directive: {parts[0]}")
    if current_name is not None:
        elements.append((current_name, current_count, current_properties))
    if not found_end:
        raise ValueError("PLY header is missing end_header")

    vertex_indices = [i for i, item in enumerate(elements) if item[0] == "vertex"]
    if not vertex_indices:
        raise ValueError("PLY header must define a vertex element")
    vertex_index = vertex_indices[0]
    if vertex_index != 0:
        raise ValueError("vertex element must precede other PLY elements")
    _, vertex_count, properties = elements[vertex_index]
    property_names = [name for name, _ in properties]
    required = ("x", "y", "z", "red", "green", "blue")
    if len(set(property_names)) != len(property_names) or any(name not in property_names for name in required):
        raise ValueError("PLY vertex properties must include unique x y z red green blue fields")

    payload = raw[newline + 1:]
    if byte_order is None:
        try:
            rows = payload.decode("ascii").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError("ASCII PLY vertex data must be ASCII") from exc
        if len(rows) < vertex_count:
            raise ValueError("PLY file contains fewer vertex rows than declared")
        values = np.empty((vertex_count, len(properties)), dtype=np.float64)
        for index, line in enumerate(rows[:vertex_count]):
            fields = line.split()
            if len(fields) != len(properties):
                raise ValueError(f"PLY vertex row {index + 1} has an incorrect field count")
            try:
                values[index] = [float(value) for value in fields]
            except ValueError as exc:
                raise ValueError(f"PLY vertex row {index + 1} contains a non-numeric value") from exc
        columns = {name: values[:, i] for i, (name, _) in enumerate(properties)}
    else:
        dtype = np.dtype([(name, byte_order + kind) for name, kind in properties], align=False)
        expected_bytes = vertex_count * dtype.itemsize
        if len(payload) < expected_bytes:
            raise ValueError("binary PLY file contains fewer vertex bytes than declared")
        records = np.frombuffer(payload[:expected_bytes], dtype=dtype, count=vertex_count)
        columns = {name: records[name].astype(np.float64) for name in property_names}

    points = np.column_stack([columns[name] for name in required]).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("PLY vertices must contain only finite values")
    colours = points[:, 3:]
    if np.any(colours < 0) or np.any(colours > 255) or np.any(colours != np.floor(colours)):
        raise ValueError("PLY RGB channels must be integers between 0 and 255")
    return points
