"""Select 3D particles: residual, border, near-duplicate and figure-of-merit gates.

Used by ``jolt joint_optimize`` (its final selection) and by ``jolt filter``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def inside_volume_mask(
    points_corner: np.ndarray, tomogram_dims_zyx: tuple[int, int, int], margin_xyz=(10.0, 10.0, 5.0)
):
    """Boolean mask of points at least ``margin_xyz`` pixels inside the tomogram (corner frame)."""
    z_dim, y_dim, x_dim = tomogram_dims_zyx
    lower = np.array([0.0, 0.0, 0.0], dtype=float) + np.asarray(margin_xyz, dtype=float)
    upper = np.array([x_dim - 1, y_dim - 1, z_dim - 1], dtype=float) - np.asarray(margin_xyz, dtype=float)
    pts = np.asarray(points_corner, dtype=float)
    return np.all((pts >= lower) & (pts <= upper), axis=1)


def border_mask(points_corner: np.ndarray, tomogram_dims_zyx: tuple[int, int, int], padding: float) -> np.ndarray:
    """Boolean mask of points at least ``padding`` pixels from every tomogram face (corner frame)."""
    z_dim, y_dim, x_dim = (float(v) for v in tomogram_dims_zyx)
    p = np.asarray(points_corner, dtype=float)
    return (
        (p[:, 0] >= padding)
        & (p[:, 0] < x_dim - padding)
        & (p[:, 1] >= padding)
        & (p[:, 1] < y_dim - padding)
        & (p[:, 2] >= padding)
        & (p[:, 2] < z_dim - padding)
    )


def dedup_indices(points: np.ndarray, priority: np.ndarray, radius: float, candidates: np.ndarray) -> np.ndarray:
    """Among ``candidates`` (row indices), keep one point per ``radius`` neighborhood.

    Points are visited in ascending ``priority`` (stable sort); each kept point removes its neighbors.

    Returns
    -------
    numpy.ndarray
        Kept row indices in ascending order.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size <= 1 or float(radius) <= 0:
        return candidates
    order = np.argsort(np.asarray(priority, dtype=float)[candidates], kind="stable")
    sorted_rows = candidates[order]
    sorted_points = np.asarray(points, dtype=float)[sorted_rows]
    tree = cKDTree(sorted_points)
    valid = np.ones(sorted_rows.shape[0], dtype=bool)
    for i in range(sorted_rows.shape[0]):
        if not valid[i]:
            continue
        for neighbor in tree.query_ball_point(sorted_points[i], r=float(radius)):
            if neighbor != i:
                valid[int(neighbor)] = False
    return np.sort(sorted_rows[valid]).astype(np.int64, copy=False)


def residual_selection(
    points_corner: np.ndarray,
    residuals: np.ndarray,
    residual_threshold: float,
    dedup_radius: float = 5.0,
    border_padding: float = 0.0,
    tomogram_dims_zyx: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Rows that pass the residual threshold, the border padding, and near-duplicate removal.

    Duplicates are resolved in favor of the lower residual. Returns row indices in ascending order.
    """
    points = np.asarray(points_corner, dtype=float)
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    if residuals.shape[0] != points.shape[0]:
        raise ValueError(f"{residuals.shape[0]} residuals for {points.shape[0]} points")
    rows = np.flatnonzero(np.isfinite(residuals) & (residuals <= float(residual_threshold))).astype(np.int64)
    if rows.size == 0:
        return rows
    if border_padding > 0 and tomogram_dims_zyx is not None:
        rows = rows[border_mask(points[rows], tomogram_dims_zyx, border_padding)]
    return dedup_indices(points, residuals, dedup_radius, rows)
