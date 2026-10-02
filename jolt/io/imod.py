"""Read and write IMOD files: positional ``.xf`` alignments, ``.tlt`` tilt angles and ``.mod`` models.

An ``.xf`` row applies to the stack image at the same position; rows are never sorted by tilt angle.
A ``.tlt`` whose length disagrees with its ``.xf`` is an error, never padded or truncated.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np


def _content_lines(path: Path) -> list[str]:
    """Return the lines of a text file without trailing blank lines; a blank line inside the file is an error."""
    lines = path.read_text().splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"{path}: blank line {number} inside the file; rows are positional, so this is ambiguous")
    return lines


def read_tlt(path: str | PathLike) -> list[float]:
    """Read the tilt angles of a ``.tlt`` or ``.rawtlt`` file, in file order.

    Parameters
    ----------
    path : str or path-like
        The ``.tlt`` file: one angle (degrees) per line. Trailing blank lines are ignored.

    Returns
    -------
    list of float
        Tilt angles in the order of the stack images. Never sorted.
    """
    path = Path(path)
    angles = []
    for number, line in enumerate(_content_lines(path), start=1):
        try:
            angles.append(float(line.strip()))
        except ValueError as exc:
            raise ValueError(f"{path}: line {number} is not a tilt angle: {line!r}") from exc
    return angles


def read_xf(path: str | PathLike, tilt_angles: list[float]) -> list[dict]:
    """Read an IMOD ``.xf`` file and attach the tilt angle of each row.

    Parameters
    ----------
    path : str or path-like
        The ``.xf`` file: six numbers per line, ``A11 A12 A21 A22 DX DY`` (translations in pixels).
    tilt_angles : list of float
        Tilt angles (degrees) from the ``.tlt`` that belongs to the same stack (see :func:`read_tlt`).
        Row ``i`` of the ``.xf`` gets ``tilt_angles[i]``.

    Returns
    -------
    list of dict
        One dictionary per stack image, in file order, with keys ``A11 A12 A21 A22 DX DY DETERMINANT TILT``.

    Raises
    ------
    ValueError
        If a line does not have six numbers, or if the number of rows differs from the number of angles.
    """
    path = Path(path)
    rows = []
    for number, line in enumerate(_content_lines(path), start=1):
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{path}: line {number} has {len(parts)} values, expected 6: {line.strip()!r}")
        a11, a12, a21, a22, dx, dy = (float(p) for p in parts)
        rows.append(
            {"A11": a11, "A12": a12, "A21": a21, "A22": a22, "DX": dx, "DY": dy, "DETERMINANT": a11 * a22 - a12 * a21}
        )
    if len(rows) != len(tilt_angles):
        raise ValueError(
            f"{path}: {len(rows)} .xf rows but {len(tilt_angles)} tilt angles. An IMOD .xf is positional "
            "(row i applies to image i), so the .xf and the .tlt must have the same number of rows."
        )
    for row, tilt in zip(rows, tilt_angles, strict=True):
        row["TILT"] = tilt
    return rows


def write_xf(path: str | PathLike, rows: list[dict]) -> None:
    """Write an IMOD ``.xf`` file, one row per stack image, in the order given.

    The translations are rounded to 32-bit floats before formatting (as the original JOLT writer did), and
    every value is written with seven decimals.

    Parameters
    ----------
    path : str or path-like
        Output file.
    rows : list of dict
        Rows in stack order, with keys ``A11 A12 A21 A22 DX DY`` (a missing matrix entry defaults to identity).
    """
    shifts = np.array([[row["DX"], row["DY"]] for row in rows], dtype=np.float32).reshape(-1, 2)
    with open(path, "w") as handle:
        for row, (dx, dy) in zip(rows, shifts, strict=True):
            a11, a12 = row.get("A11", 1.0), row.get("A12", 0.0)
            a21, a22 = row.get("A21", 0.0), row.get("A22", 1.0)
            handle.write(f"{a11:.7f}   {a12:.7f}   {a21:.7f}   {a22:.7f}      {dx:.7f}      {dy:.7f}\n")


def write_points_mod(path: str | PathLike, points_xyz: np.ndarray) -> None:
    """Write 3D points as one scattered-point object of an IMOD model (``.mod``) for viewing in ``3dmod``.

    Parameters
    ----------
    path : str or path-like
        Output ``.mod`` file.
    points_xyz : numpy.ndarray
        ``(N, 3)`` coordinates in tomogram pixels (the same values as the particle STAR file).
    """
    import imodmodel
    import pandas as pd

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    frame = pd.DataFrame(
        {"object_id": 0, "contour_id": 0, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]},
        index=range(len(pts)),
    )
    imodmodel.write(frame, path)
