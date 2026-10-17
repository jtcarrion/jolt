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


def write_points_mod(
    path: str | PathLike, points_xyz: np.ndarray, volume_dims_xyz: tuple[int, int, int] | None = None
) -> None:
    """Write 3D points as one scattered-point object of an IMOD model (``.mod``) for viewing in ``3dmod``.

    IMOD model coordinates place the center of voxel ``i`` at ``i + 0.5`` (a point on a 0-based voxel index ``i``
    sits on the voxel center after adding 0.5), so 0.5 is added on every axis. The volume size is stamped in the
    model header; otherwise 3dmod sizes the model from the bounding box of its points and can misplace it.

    Parameters
    ----------
    path : str or path-like
        Output ``.mod`` file.
    points_xyz : numpy.ndarray
        ``(N, 3)`` coordinates in tomogram voxels, 0-based (the same values as the particle STAR file).
    volume_dims_xyz : tuple of int, optional
        Tomogram size in voxels, written to the model header.
    """
    import pandas as pd
    from imodmodel import ImodModel

    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3) + 0.5
    frame = pd.DataFrame(
        {"object_id": 0, "contour_id": 0, "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]},
        index=range(len(pts)),
    )
    model = ImodModel.from_dataframe(frame)
    if volume_dims_xyz is not None:
        model.header.xmax, model.header.ymax, model.header.zmax = (int(v) for v in volume_dims_xyz)
    model.to_file(path)


def write_picks_mod(
    path: str | PathLike,
    picks_by_section: dict[int, np.ndarray],
    image_dims_xy: tuple[int, int],
    n_sections: int,
    sphere_radius: float = 2.0,
) -> None:
    """Write 2D picks on a tilt stack as an IMOD model (``.mod``), for ``3dmod <stack> <model>``.

    The convention is etomo's for models on a stack (e.g. its ``.fid`` bead model): ``x, y`` at pixel index + 0.5
    (IMOD places pixel centers there) and ``z`` the 0-based section (tilt image) index, an integer, so every pick
    shows on its own tilt image. One scattered-point object with one contour per tilt image, so 3dmod lists the
    picks per tilt. Points are drawn as circles of ``sphere_radius`` pixels on their own section only (3dmod would
    otherwise draw each sphere on the neighboring sections too, i.e. on other tilt images). The image size and
    number of sections are stamped in the header.

    Parameters
    ----------
    path : str or path-like
        Output ``.mod`` file.
    picks_by_section : dict
        ``{0-based section index: (n, 2) array of x, y}`` in the image's 0-based pixel coordinates (as the picks STAR).
    image_dims_xy : tuple of int
        Tilt-image width and height in pixels.
    n_sections : int
        Number of images in the stack.
    sphere_radius : float
        Radius of the circle drawn at each pick, in pixels (``detect_2d`` uses its extraction radius).
    """
    import pandas as pd
    from imodmodel import ImodModel

    parts = []
    for section in sorted(picks_by_section):
        xy = np.asarray(picks_by_section[section], dtype=float).reshape(-1, 2)
        if not 0 <= int(section) < int(n_sections):
            raise ValueError(f"section {section} outside 0..{int(n_sections) - 1}")
        if len(xy):
            parts.append(
                pd.DataFrame(
                    {
                        "object_id": 0,
                        "contour_id": int(section),
                        "x": xy[:, 0] + 0.5,
                        "y": xy[:, 1] + 0.5,
                        "z": float(section),
                    }
                )
            )
    if not parts:
        raise ValueError("no picks to write")
    model = ImodModel.from_dataframe(pd.concat(parts, ignore_index=True))
    obj = model.objects[0].header
    obj.flags.draw_spheres_central_section_only = True  # a pick belongs to one tilt image, not its neighbors
    obj.pdrawsize = max(1, int(round(float(sphere_radius))))
    model.header.xmax, model.header.ymax = (int(v) for v in image_dims_xy)
    model.header.zmax = int(n_sections)
    model.to_file(path)


def read_fiducial_beads(
    fid_path: str | PathLike, prexg_path: str | PathLike, n_tilts: int, raw_dims_xy: tuple[int, int]
):
    """Gold-bead observations from an IMOD fiducial model (``.fid``), in the raw tilt-image frame.

    etomo tracks beads on the coarse-prealigned stack, so the ``.fid`` is in the prealigned frame; each observation is
    mapped back to the raw ``.st`` frame with the inverse of that view's ``.prexg`` transform (about the image center
    ``(N - 1)/2``), after removing IMOD's model convention (voxel centers at index + 0.5). One contour is one bead, so
    bead identity across tilts is known. Validated on EMPIAR-10499: beads mapped this way and run through JOLT land on
    the gold in the tomograms to < 0.1 voxel.

    Parameters
    ----------
    fid_path, prexg_path : str or path-like
        The ``<ts>.fid`` model and the ``<ts>.prexg`` coarse prealignment of the same stack.
    n_tilts : int
        Number of images in the stack (rows of the ``.prexg``).
    raw_dims_xy : tuple of int
        Raw image width and height.

    Returns
    -------
    pandas.DataFrame
        Columns ``bead_id`` (0-based, one per contour), ``tilt_index`` (0-based stack index), ``x``, ``y`` (raw frame,
        pixels, y down as in STAR files).
    """
    import imodmodel
    import pandas as pd

    model = imodmodel.read(fid_path)
    pre = np.loadtxt(prexg_path, ndmin=2)
    if pre.shape[0] != int(n_tilts) or pre.shape[1] < 6:
        raise ValueError(f"{prexg_path}: {pre.shape[0]} rows for a stack of {n_tilts} images")
    a_pre = pre[:, :4].reshape(-1, 2, 2)
    d_pre = pre[:, 4:6]
    center = np.array([(raw_dims_xy[0] - 1) / 2.0, (raw_dims_xy[1] - 1) / 2.0])
    view = np.rint(model["z"].to_numpy(float)).astype(int)
    if view.min() < 0 or view.max() >= int(n_tilts):
        raise ValueError(f"{fid_path}: bead observations on views outside 0..{int(n_tilts) - 1}")
    xy = model[["x", "y"]].to_numpy(float) - 0.5
    raw = np.empty_like(xy)
    for i, (v, p) in enumerate(zip(view, xy, strict=True)):
        raw[i] = np.linalg.solve(a_pre[v], p - center - d_pre[v]) + center
    ids = model["object_id"].astype(str) + ":" + model["contour_id"].astype(str)
    bead_id = pd.factorize(ids)[0]
    return pd.DataFrame({"bead_id": bead_id, "tilt_index": view, "x": raw[:, 0], "y": raw[:, 1]})
