"""Read MRC headers and images (wrapper around ``mrcfile``)."""

from __future__ import annotations

from os import PathLike

import mrcfile
import numpy as np


def read_header_dims(path: str | PathLike) -> tuple[int, int, int]:
    """Return the dimensions ``(nx, ny, nz)`` of an MRC file or IMOD stack, read from its header only.

    For a tilt stack, ``nx`` and ``ny`` are the image width and height in pixels and ``nz`` the number of
    tilt images.
    """
    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        return int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz)


def read_pixel_size(path: str | PathLike) -> float:
    """Return the pixel size along x in Angstrom, read from the MRC header."""
    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        return float(mrc.voxel_size.x)


def read_stack(path: str | PathLike) -> np.ndarray:
    """Read a whole stack into memory as an array of shape ``(nz, ny, nx)``."""
    with mrcfile.open(path, permissive=True) as mrc:
        return np.asarray(mrc.data).copy()
