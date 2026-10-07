"""Tilt-series geometry: coordinate frames, the raw-to-aligned pick transform, and projection.

Frames used inside JOLT
-----------------------
2D picks (STAR files)
    Corner frame of a tilt image, in pixels: ``x`` to the right and ``y`` down from the top-left corner.
Internal 2D frame
    Centered, ``y`` up: ``u = x - cx``, ``v = (H - 1 - y) - cy`` with ``(cx, cy) = ((W - 1)/2, (H - 1)/2)``
    (see :func:`corner_to_center_2d_bottom_left`).
3D points
    Tomogram pixels at the tilt-stack pixel size, corner frame ``(x, y, z)`` with dimensions ``(Z, Y, X)``,
    or the centered frame obtained by subtracting :func:`tomogram_center`.
Projection
    A centered 3D point ``(X, Y, Z)`` projects to tilt ``theta`` (single tilt axis along ``y``) as
    ``u = cos(theta) X + sin(theta) Z + dx`` and ``v = Y + dy``, where ``(dx, dy)`` are the per-tilt shifts
    refined by bundle adjustment.

Image center
------------
Every center is ``((N - 1)/2)`` in 0-based pixel coordinates (an integer coordinate is the center of a pixel,
as in Topaz and STAR files). This is the center IMOD's ``newstack -xf`` rotates about: measured with Gaussian
blobs at known positions, its translations match ``(N - 1)/2`` to 0.003 px, while ``N/2`` is off by up to about
0.45 px on rotated or reshaped canvases.
"""

from __future__ import annotations

import numpy as np


def tomogram_center(tomogram_dims_zyx: tuple[int, int, int]) -> tuple[float, float, float]:
    """Return the tomogram center ``(cx, cy, cz)`` in pixels for dimensions ``(Z, Y, X)``."""
    return (
        (tomogram_dims_zyx[2] - 1) / 2.0,
        (tomogram_dims_zyx[1] - 1) / 2.0,
        (tomogram_dims_zyx[0] - 1) / 2.0,
    )


def image_center(image_dims_hw: tuple[int, int]) -> tuple[float, float]:
    """Return the image center ``(cx, cy)`` in pixels for dimensions ``(height, width)``."""
    h, w = image_dims_hw
    return (w - 1) / 2.0, (h - 1) / 2.0


def corner_to_center_3d(points: np.ndarray, tomogram_dims_zyx: tuple[int, int, int]) -> np.ndarray:
    """Convert ``(N, 3)`` corner-frame 3D points to the centered frame."""
    pts = np.asarray(points, dtype=float).copy()
    cx, cy, cz = tomogram_center(tomogram_dims_zyx)
    pts[:, 0] -= cx
    pts[:, 1] -= cy
    pts[:, 2] -= cz
    return pts


def center_to_corner_3d(points: np.ndarray, tomogram_dims_zyx: tuple[int, int, int]) -> np.ndarray:
    """Convert ``(N, 3)`` centered 3D points to the corner frame."""
    pts = np.asarray(points, dtype=float).copy()
    cx, cy, cz = tomogram_center(tomogram_dims_zyx)
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts[:, 2] += cz
    return pts


def flip_y_corner_3d(points: np.ndarray, tomogram_dims_zyx: tuple[int, int, int]) -> np.ndarray:
    """Mirror ``(N, 3)`` corner-frame 3D points in y: ``y' = (Y - 1) - y``."""
    pts = np.asarray(points, dtype=float).copy()
    pts[:, 1] = float(tomogram_dims_zyx[1] - 1) - pts[:, 1]
    return pts


def corner_to_center_2d_bottom_left(points: np.ndarray, image_dims_hw: tuple[int, int]) -> np.ndarray:
    """Convert ``(N, 2)`` STAR picks (corner frame, ``y`` down) to the internal centered frame (``y`` up)."""
    pts = np.asarray(points, dtype=float).copy()
    cx, cy = image_center(image_dims_hw)
    h, _ = image_dims_hw
    pts[:, 0] -= cx
    pts[:, 1] = (float(h) - 1.0 - pts[:, 1]) - cy
    return pts


def center_to_corner_2d_bottom_left(points: np.ndarray, image_dims_hw: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`corner_to_center_2d_bottom_left`."""
    pts = np.asarray(points, dtype=float).copy()
    cx, cy = image_center(image_dims_hw)
    h, _ = image_dims_hw
    pts[:, 0] += cx
    pts[:, 1] = float(h) - 1.0 - (pts[:, 1] + cy)
    return pts


def raw_picks_to_aligned(
    picks: np.ndarray,
    xf_row: dict,
    raw_dims_xy: tuple[int, int],
    aligned_dims_xy: tuple[int, int],
) -> np.ndarray:
    """Map picks made on a raw tilt image into the aligned frame with that image's IMOD ``.xf`` row.

    ``x' = A (x - c_raw) + (DX, DY) + c_aligned``, with the centers ``c = ((W - 1)/2, (H - 1)/2)`` of each
    canvas, the convention IMOD's ``newstack -xf`` uses.

    Parameters
    ----------
    picks : numpy.ndarray
        ``(N, 3)`` array of ``x, y, score`` in raw-image pixels (corner frame). The score column is kept.
    xf_row : dict
        The ``.xf`` row of this tilt image (keys ``A11 A12 A21 A22 DX DY``).
    raw_dims_xy, aligned_dims_xy : tuple of int
        Width and height of the raw and the aligned tilt images in pixels.

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` array in aligned-image pixels.
    """
    arr = np.asarray(picks, dtype=np.float64)
    cx_raw, cy_raw = (raw_dims_xy[0] - 1) / 2.0, (raw_dims_xy[1] - 1) / 2.0
    cx_out, cy_out = (aligned_dims_xy[0] - 1) / 2.0, (aligned_dims_xy[1] - 1) / 2.0
    a11, a12 = float(xf_row["A11"]), float(xf_row["A12"])
    a21, a22 = float(xf_row["A21"]), float(xf_row["A22"])
    dx, dy = float(xf_row["DX"]), float(xf_row["DY"])
    xr = arr[:, 0] - cx_raw
    yr = arr[:, 1] - cy_raw
    xp = a11 * xr + a12 * yr + dx
    yp = a21 * xr + a22 * yr + dy
    return np.column_stack([xp + cx_out, yp + cy_out, arr[:, 2]])


def project_y_axis(xyz: np.ndarray, theta_deg: float, translation: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Project centered 3D points to one tilt in the internal 2D frame.

    Parameters
    ----------
    xyz : numpy.ndarray
        ``(N, 3)`` or ``(3,)`` centered 3D points.
    theta_deg : float
        Tilt angle in degrees.
    translation : tuple of float
        Per-tilt shift ``(dx, dy)`` in pixels.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` (or ``(2,)``) array of ``u, v``.
    """
    theta = np.deg2rad(float(theta_deg))
    c, s = np.cos(theta), np.sin(theta)
    dx, dy = translation
    pts = np.asarray(xyz, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    out = np.stack([c * pts[:, 0] + s * pts[:, 2] + dx, pts[:, 1] + dy], axis=1)
    return out[0] if single else out


def tilt_projection_matrices(tilt_angles_deg, device=None, dtype=None):
    """Build one 4x4 projection matrix per tilt for the centered frame (single tilt axis along y).

    Row 0 maps a homogeneous centered 3D point ``(X, Y, Z, 1)`` to ``u = cos(theta) X + sin(theta) Z`` and row 1
    to ``v = Y``. Shifts are not included; they are added separately (they are what bundle adjustment refines).
    The matrices are built in 32-bit floats exactly as the original JOLT code built them (a product of x, y and
    z rotations with zero x and z angles), so projections are bit-for-bit the same.

    Parameters
    ----------
    tilt_angles_deg : sequence of float
        Tilt angles in degrees, in the order the matrices should have.
    device : torch.device, optional
    dtype : torch.dtype, optional
        Defaults to ``torch.float32``.

    Returns
    -------
    torch.Tensor
        ``(M, 4, 4)``.
    """
    import torch

    device = device or torch.device("cpu")
    dtype = dtype or torch.float32
    zero = np.deg2rad(0.0)
    cx, sx = np.cos(zero), np.sin(zero)
    r_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    r_z = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]], dtype=np.float32)
    matrices = []
    for theta in tilt_angles_deg:
        ry = np.deg2rad(float(theta))
        cy, sy = np.cos(ry), np.sin(ry)
        r_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        r_forward = r_z @ r_y @ r_x
        m = torch.eye(4, dtype=dtype, device=device)
        m[:2, :3] = torch.from_numpy(r_forward[:2, :]).to(device)
        m[0, 3] = 0.0
        m[1, 3] = 0.0
        matrices.append(m)
    return torch.stack(matrices, dim=0)


def project_points(points_center, projection_matrices):
    """Project ``(N, 3)`` centered 3D points with ``(M, 4, 4)`` matrices; returns ``(N, M, 2)`` (torch)."""
    import torch

    ones = torch.ones(points_center.shape[0], 1, device=points_center.device, dtype=points_center.dtype)
    homogeneous = torch.cat([points_center, ones], dim=1)
    return torch.einsum("mij,nj->nmi", projection_matrices, homogeneous)[:, :, :2]


def picks_to_center_tensor(picks: np.ndarray, image_dims_hw: tuple[int, int], device):
    """Convert ``(K, >=2)`` STAR picks to a float32 torch tensor in the internal centered frame.

    The coordinates are rounded to 32-bit floats before the conversion, as the original code did.
    """
    import torch

    corner = np.asarray([[float(p[0]), float(p[1])] for p in picks], dtype=np.float32)
    center = corner_to_center_2d_bottom_left(corner, image_dims_hw).astype(np.float32)
    return torch.as_tensor(center, dtype=torch.float32, device=device)


def aligned_to_raw(points_xy: np.ndarray, xf_row: dict, raw_dims_xy, aligned_dims_xy) -> np.ndarray:
    """Inverse of :func:`raw_picks_to_aligned` for ``(N, 2)`` aligned-image pixel positions."""
    pts = np.asarray(points_xy, dtype=np.float64)
    a = np.array([[float(xf_row["A11"]), float(xf_row["A12"])], [float(xf_row["A21"]), float(xf_row["A22"])]])
    shifted = np.vstack(
        [
            pts[:, 0] - (aligned_dims_xy[0] - 1) / 2.0 - float(xf_row["DX"]),
            pts[:, 1] - (aligned_dims_xy[1] - 1) / 2.0 - float(xf_row["DY"]),
        ]
    )
    raw = np.linalg.solve(a, shifted)
    return np.column_stack([raw[0] + (raw_dims_xy[0] - 1) / 2.0, raw[1] + (raw_dims_xy[1] - 1) / 2.0])


def project_star_points_to_raw(
    points_xyz: np.ndarray,
    tilt_deg: float,
    xf_row: dict,
    tomo_dims_xyz: tuple[int, int, int],
    raw_dims_xy: tuple[int, int],
) -> np.ndarray:
    """Project 3D particles given in the particle-STAR frame into one raw tilt image.

    Parameters
    ----------
    points_xyz : numpy.ndarray
        ``(N, 3)`` corner-frame tomogram pixels at the tilt-stack pixel size, y down (the frame of the
        ``joint_optimize`` particle STAR files).
    tilt_deg : float
        Tilt angle of the image.
    xf_row : dict
        The image's ``.xf`` row (raw to aligned).
    tomo_dims_xyz : tuple of int
        Tomogram dimensions; X, Y are also the aligned image size.
    raw_dims_xy : tuple of int
        Raw image width and height.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` raw-image pixel positions ``x, y`` (corner frame, y down).
    """
    x_dim, y_dim, z_dim = tomo_dims_xyz
    pts = np.asarray(points_xyz, dtype=float).copy()
    pts[:, 1] = (y_dim - 1) - pts[:, 1]  # STAR frame -> internal frame (y up)
    center = corner_to_center_3d(pts, (z_dim, y_dim, x_dim))
    aligned = center_to_corner_2d_bottom_left(project_y_axis(center, tilt_deg), (y_dim, x_dim))
    return aligned_to_raw(aligned, xf_row, raw_dims_xy, (x_dim, y_dim))
