"""Strict reader for the ASCII XYZRGB PLY files exported by BLINDSPOT."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np


def read_ascii_ply(source: str | Path | bytes | bytearray | BinaryIO) -> np.ndarray:
    """Read an ASCII PLY vertex cloud with x/y/z/red/green/blue properties.

    Returns a float32 Nx6 array in XYZRGB order. Binary PLY, extra vertex
    properties, and files without the expected colour channels are rejected
    rather than silently misinterpreted.
    """
    if isinstance(source, (bytes, bytearray)):
        lines = bytes(source).decode("ascii").splitlines()
    elif hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            lines = raw.decode("ascii").splitlines()
        else:
            lines = str(raw).splitlines()
    else:
        lines = Path(source).read_text(encoding="ascii").splitlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError("file is not a PLY file")
    if len(lines) < 3 or lines[1].strip() != "format ascii 1.0":
        raise ValueError("only ASCII PLY format 1.0 is supported")

    vertex_count = None
    properties: list[str] = []
    in_vertex = False
    end_header = None
    for index, raw_line in enumerate(lines[2:], start=2):
        parts = raw_line.strip().split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "element":
            in_vertex = len(parts) == 3 and parts[1] == "vertex"
            if in_vertex:
                try:
                    vertex_count = int(parts[2])
                except ValueError as exc:
                    raise ValueError("invalid PLY vertex count") from exc
                if vertex_count < 0:
                    raise ValueError("PLY vertex count must be non-negative")
            continue
        if parts[0] == "property" and in_vertex:
            if len(parts) != 3:
                raise ValueError("list-valued PLY vertex properties are not supported")
            properties.append(parts[2])
            continue
        if parts[0] == "end_header":
            end_header = index
            break

    expected = ["x", "y", "z", "red", "green", "blue"]
    if vertex_count is None or end_header is None:
        raise ValueError("PLY header must define a vertex element and end_header")
    if properties != expected:
        raise ValueError("PLY vertex properties must be x y z red green blue in that order")
    data_lines = lines[end_header + 1:end_header + 1 + vertex_count]
    if len(data_lines) != vertex_count:
        raise ValueError("PLY file contains fewer vertex rows than declared")

    points = np.empty((vertex_count, 6), dtype=np.float32)
    for index, line in enumerate(data_lines):
        values = line.split()
        if len(values) != 6:
            raise ValueError(f"PLY vertex row {index + 1} must contain six values")
        try:
            points[index] = [float(value) for value in values]
        except ValueError as exc:
            raise ValueError(f"PLY vertex row {index + 1} contains a non-numeric value") from exc
    if not np.isfinite(points).all():
        raise ValueError("PLY vertices must contain only finite values")
    colours = points[:, 3:]
    if np.any(colours < 0) or np.any(colours > 255) or np.any(colours != np.floor(colours)):
        raise ValueError("PLY RGB channels must be integers between 0 and 255")
    return points
