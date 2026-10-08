"""Per-particle reprojection residual, figure of merit (FOM) and mean 2D detector score.

The FOM is the original JOLT "top-5" score, computed without ground truth::

    FOM = 0.96 geom + 0.01 (match_fraction + global_coverage + avg_dist_quality + rmse_quality)

``geom = exp(-residual / rmse_scale)`` carries almost all of the discrimination. The residual is the RMS
distance to the nearest pick over the tilts where a pick is within the match distance (12 px, the tilts bundle
adjustment fits; :func:`matched_residuals`), not over all tilts: the all-tilt RMS penalizes tilts where the 2D
picker found nothing and ranks true particles worse on every dataset measured. The four benchmark features
come from projecting every particle into every tilt and matching the projections one-to-one with the picks.

Every projection includes the per-tilt shifts refined by bundle adjustment (``shifts_by_tilt``), so a particle
is scored against the alignment that was actually written to the refined ``.xf``. Before this, the residual
used the input alignment, so on a misaligned tilt series good particles kept large residuals even after
bundle adjustment had corrected the misalignment; the benchmark features also compared y-mirrored
projections with the picks. On synthetic data both fixes raised the AUROC of the FOM (true particles against
the pipeline's false positives) from 0.75 to 0.88, and from 0.86 to 0.98 with 8 px of misalignment.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from jolt.geometry import (
    center_to_corner_2d_bottom_left,
    corner_to_center_2d_bottom_left,
    corner_to_center_3d,
    project_points,
    tilt_projection_matrices,
)

#: Weights of (match_fraction, global_coverage, avg_dist_quality, rmse_quality, geom).
TOP5_FOM_WEIGHTS = (0.01, 0.01, 0.01, 0.01, 0.96)


def _device(device=None) -> torch.device:
    return device or torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _shift_for_tilt(shifts_by_tilt: dict[float, tuple[float, float]] | None, tilt: float) -> tuple[float, float]:
    """Internal-frame shift ``(dx, dy)`` of the tilt closest to ``tilt`` (within 0.5 degrees), else zero."""
    if not shifts_by_tilt:
        return (0.0, 0.0)
    keys = np.asarray([float(k) for k in shifts_by_tilt], dtype=float)
    i = int(np.argmin(np.abs(keys - float(tilt))))
    if abs(keys[i] - float(tilt)) > 0.5:
        return (0.0, 0.0)
    dx, dy = list(shifts_by_tilt.values())[i]
    return (float(dx), float(dy))


def _nearest_pick_distances(
    points_corner: np.ndarray,
    coords_by_tilt: dict[float, np.ndarray],
    tilts_file_order: list[float],
    image_dims_hw: tuple[int, int],
    tomogram_dims_zyx: tuple[int, int, int],
    device=None,
    shifts_by_tilt: dict[float, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Distance from each point's projection (plus the tilt's shift) to the nearest pick, per tilt.

    Returns an ``(N, n_tilts)`` array, NaN on tilts without picks.
    """
    x = np.asarray(points_corner, dtype=float)
    n = x.shape[0]
    dist = np.full((n, len(tilts_file_order)), np.nan, dtype=float)
    if n == 0:
        return dist
    device = _device(device)
    tomo_cx = (tomogram_dims_zyx[2] - 1) / 2.0
    tomo_cy = (tomogram_dims_zyx[1] - 1) / 2.0
    tomo_cz = (tomogram_dims_zyx[0] - 1) / 2.0
    xc = x.copy()
    xc[:, 0] -= tomo_cx
    xc[:, 1] -= tomo_cy
    xc[:, 2] -= tomo_cz
    projected = project_points(
        torch.as_tensor(xc, dtype=torch.float32, device=device),
        tilt_projection_matrices(tilts_file_order, device=device),
    )
    keys = sorted(float(t) for t in coords_by_tilt.keys())
    keys_arr = np.asarray(keys, dtype=float)
    for v, tilt in enumerate(tilts_file_order):
        if keys_arr.size == 0:
            continue
        i = int(np.argmin(np.abs(keys_arr - float(tilt))))
        if abs(keys_arr[i] - float(tilt)) > 0.5:
            continue
        picks = coords_by_tilt[keys[i]]
        if len(picks) == 0:
            continue
        raw = np.asarray([[p[0], p[1]] for p in picks], dtype=np.float32)
        raw_t = torch.as_tensor(corner_to_center_2d_bottom_left(raw, image_dims_hw), dtype=torch.float32, device=device)
        shift = torch.as_tensor(_shift_for_tilt(shifts_by_tilt, tilt), dtype=torch.float32, device=device)
        dist[:, v] = torch.min(torch.cdist(projected[:, v, :] + shift, raw_t, p=2), dim=1)[0].detach().cpu().numpy()
    return dist


def reprojection_residuals(
    points_corner: np.ndarray,
    coords_by_tilt: dict[float, np.ndarray],
    tilts_file_order: list[float],
    image_dims_hw: tuple[int, int],
    tomogram_dims_zyx: tuple[int, int, int],
    device=None,
    shifts_by_tilt: dict[float, tuple[float, float]] | None = None,
) -> np.ndarray:
    """RMS distance from each point's projections to the nearest pick, over all tilts with picks.

    Parameters
    ----------
    points_corner : numpy.ndarray
        ``(N, 3)`` points in the corner 3D frame.
    coords_by_tilt : dict of float to numpy.ndarray
        Aligned-frame picks per tilt.
    tilts_file_order : list of float
        Tilt angles in stack (``.tlt``) order; the sum over tilts runs in this order.
    image_dims_hw, tomogram_dims_zyx : tuple of int
        Aligned image and tomogram dimensions.
    device : torch.device, optional
    shifts_by_tilt : dict of float to (float, float), optional
        Per-tilt shifts ``(dx, dy)`` in the internal frame (y up) added to the projections, e.g. the change
        made by bundle adjustment. Tilts not listed get no shift.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` residuals in pixels (NaN for a point with no tilt to compare against).
    """
    dist = _nearest_pick_distances(
        points_corner, coords_by_tilt, tilts_file_order, image_dims_hw, tomogram_dims_zyx, device, shifts_by_tilt
    )
    count = np.isfinite(dist).sum(axis=1)
    residuals = np.full(dist.shape[0], np.nan, dtype=float)
    valid = count > 0
    residuals[valid] = np.sqrt(np.nansum(dist[valid] ** 2, axis=1) / count[valid])
    return residuals


def matched_residuals(
    points_corner: np.ndarray,
    coords_by_tilt: dict[float, np.ndarray],
    tilts_file_order: list[float],
    image_dims_hw: tuple[int, int],
    tomogram_dims_zyx: tuple[int, int, int],
    match_distance: float,
    device=None,
    shifts_by_tilt: dict[float, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RMS distance to the nearest pick over the tilts where a pick is within ``match_distance``, and their count.

    These are the tilts bundle adjustment fits (it matches each projection to its nearest pick within the same
    distance), so this residual measures geometric fit, not pick coverage: a tilt where the 2D picker found nothing
    near the particle does not count against it. On EMPIAR-10499 and CZI it ranks true particles above false ones
    better than the all-tilt RMS (:func:`reprojection_residuals`) on every pick set measured (AUROC +0.02 to +0.22).

    Returns
    -------
    tuple of numpy.ndarray
        ``(N,)`` residuals in pixels (``inf`` for a point with no matched tilt) and ``(N,)`` matched-tilt counts.
    """
    dist = _nearest_pick_distances(
        points_corner, coords_by_tilt, tilts_file_order, image_dims_hw, tomogram_dims_zyx, device, shifts_by_tilt
    )
    matched = np.isfinite(dist) & (dist < float(match_distance))
    count = matched.sum(axis=1)
    residuals = np.full(dist.shape[0], np.inf, dtype=float)
    valid = count > 0
    squared = np.where(matched, dist, 0.0) ** 2
    residuals[valid] = np.sqrt(squared[valid].sum(axis=1) / count[valid])
    return residuals, count


def _row_normalized_assignments(
    points_center: torch.Tensor,
    picks: np.ndarray,
    projection_row: int,
    projection: torch.Tensor,
    image_dims_hw: tuple[int, int],
    search_radius: float,
    affinity_temp: float,
    shift: tuple[float, float] = (0.0, 0.0),
) -> torch.Tensor:
    """Soft assignment of every point to the picks of one tilt (softmax per point, with a no-match entry).

    Returns an ``(N, K)`` float32 CPU tensor; each row sums to at most one.
    """
    n = points_center.shape[0]
    k = len(picks)
    if k == 0:
        return torch.zeros(n, 0, dtype=torch.float32)
    raw = np.asarray([[float(p[0]), float(p[1])] for p in picks], dtype=np.float32)
    raw_t = torch.as_tensor(
        corner_to_center_2d_bottom_left(raw, image_dims_hw), dtype=torch.float32, device=points_center.device
    )
    with torch.no_grad():
        proj_2d = project_points(points_center, projection)[:, projection_row, :]
        proj_2d = proj_2d + torch.as_tensor(shift, dtype=torch.float32, device=points_center.device)
    sr = float(search_radius)
    dists = torch.cdist(proj_2d, raw_t, p=2)
    costs = (dists / sr) ** 2
    large_cost = float((dists.max().item() / sr) ** 2) + 10.0
    costs = torch.where(dists <= sr, costs, torch.full_like(costs, large_cost))
    dustbin = torch.full((n, 1), 2.25, device=points_center.device)
    affinities = torch.exp(-torch.cat([costs, dustbin], dim=1) / float(affinity_temp))
    p_full = affinities / affinities.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return p_full[:, :k].detach().cpu()


def assignment_components(
    points_corner: np.ndarray,
    residuals: np.ndarray,
    coords_by_tilt: dict[float, np.ndarray],
    tilts_file_order: list[float],
    image_dims_hw: tuple[int, int],
    tomogram_dims_zyx: tuple[int, int, int],
    *,
    search_radius: float,
    affinity_temp: float,
    tau: float,
    rmse_scale: float,
    device=None,
    shifts_by_tilt: dict[float, tuple[float, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Per-point ``conf``, ``geom``, ``breadth``, ``cnn`` and ``cnn_raw`` (the mean 2D detector score).

    ``shifts_by_tilt`` (internal frame) is added to the projections, as in :func:`reprojection_residuals`.
    """
    device = _device(device)
    x_center_t = torch.as_tensor(
        corner_to_center_3d(np.asarray(points_corner, dtype=float), tomogram_dims_zyx),
        dtype=torch.float32,
        device=device,
    )
    projection = tilt_projection_matrices(tilts_file_order, device=device)
    file_tilts = np.asarray([float(t) for t in tilts_file_order], dtype=float)
    keys = list(coords_by_tilt.keys())
    key_vals = np.asarray([float(k) for k in keys], dtype=float)
    p_by_tilt = {}
    for tilt in sorted(float(t) for t in coords_by_tilt.keys()):
        i = int(np.argmin(np.abs(key_vals - tilt)))
        j = int(np.argmin(np.abs(file_tilts - tilt)))
        if abs(key_vals[i] - tilt) > 0.5 or abs(file_tilts[j] - tilt) > 0.5:
            continue
        p_by_tilt[tilt] = _row_normalized_assignments(
            x_center_t,
            coords_by_tilt[keys[i]],
            j,
            projection,
            image_dims_hw,
            search_radius,
            affinity_temp,
            _shift_for_tilt(shifts_by_tilt, tilt),
        )
    return _components_from_assignments(p_by_tilt, residuals, coords_by_tilt, len(coords_by_tilt), tau, rmse_scale)


def _components_from_assignments(p_by_tilt, residuals, coords_by_tilt, n_tilts_total, tau, rmse_scale):
    res = np.asarray(residuals, dtype=float)
    if not p_by_tilt:
        geom = np.clip(np.exp(-res / max(rmse_scale, 1e-6)), 0.0, 1.0).astype(np.float64)
        zeros = np.zeros_like(geom, dtype=np.float64)
        return {
            "conf": zeros.copy(),
            "geom": geom,
            "breadth": zeros.copy(),
            "cnn": zeros.copy(),
            "cnn_raw": np.full_like(geom, np.nan, dtype=np.float64),
        }
    n_points = int(list(p_by_tilt.values())[0].shape[0])
    usable = []
    for tilt, p_t in p_by_tilt.items():
        if p_t.shape[1] == 0:
            continue
        p_np = p_t.numpy()
        if np.all(np.isnan(p_np)):
            continue
        usable.append((tilt, np.nan_to_num(p_np, nan=0.0)))

    sum_conf = np.zeros(n_points, dtype=np.float64)
    n_confident = np.zeros(n_points, dtype=np.float64)
    for _, p_np in usable:
        m_t, n_t = p_np.shape
        budget = min(m_t, n_t) / max(max(m_t, n_t), 1)
        max_p = p_np.max(axis=1)
        sum_conf += np.clip(max_p / (budget + 1e-8), 0.0, 1.0)
        n_confident += (max_p / (budget + 1e-8) > tau).astype(float)
    conf = np.clip(sum_conf / max(len(usable), 1), 0.0, 1.0)

    res = np.where(np.isfinite(res), res, float(np.nanmax(res[np.isfinite(res)])) if np.any(np.isfinite(res)) else 1e3)
    geom = np.clip(np.exp(-res / max(rmse_scale, 1e-6)), 0.0, 1.0)
    breadth = np.clip(n_confident / max(int(n_tilts_total), 1), 0.0, 1.0)

    weighted_sum = np.zeros(n_points, dtype=np.float64)
    weight_total = np.zeros(n_points, dtype=np.float64)
    for tilt, p_np in usable:
        picks = coords_by_tilt.get(float(tilt), [])
        if len(picks) == 0:
            continue
        picks_arr = np.asarray(picks, dtype=float)
        if picks_arr.ndim != 2 or picks_arr.shape[1] < 3:
            continue
        scores = picks_arr[:, 2][: p_np.shape[1]]
        weighted_sum += p_np[:, : len(scores)] @ scores
        weight_total += p_np[:, : len(scores)].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_cnn = np.where(weight_total > 1e-12, weighted_sum / weight_total, 0.0)
        cnn_raw = np.where(weight_total > 1e-12, weighted_sum / weight_total, np.nan)
    lo, hi = float(raw_cnn.min()), float(raw_cnn.max())
    cnn = np.clip((raw_cnn - lo) / (hi - lo), 0.0, 1.0) if hi > lo + 1e-12 else np.zeros_like(raw_cnn)
    return {
        "conf": np.asarray(conf, dtype=np.float64),
        "geom": np.asarray(geom, dtype=np.float64),
        "breadth": np.asarray(breadth, dtype=np.float64),
        "cnn": np.asarray(cnn, dtype=np.float64),
        "cnn_raw": np.asarray(cnn_raw, dtype=np.float64),
    }


def _benchmark_features(
    points_corner, coords_by_tilt, tilts_file_order, image_dims_hw, tomogram_dims_zyx, shifts_by_tilt=None
):
    """match_fraction, global_coverage, avg_dist_quality and rmse_quality of every point.

    Each point is projected into every tilt (plus that tilt's shift) and the projections are matched one-to-one
    with the picks, in the picks' own frame (STAR corner frame, y down).
    """
    n = len(points_corner)
    points_center = torch.as_tensor(
        corner_to_center_3d(np.asarray(points_corner, dtype=float), tomogram_dims_zyx), dtype=torch.float32
    )
    projected = project_points(points_center, tilt_projection_matrices(tilts_file_order)).numpy()
    projected_by_tilt = {}
    for tilt_idx, tilt in enumerate(tilts_file_order):
        uv = projected[:, tilt_idx, :].astype(float) + np.asarray(_shift_for_tilt(shifts_by_tilt, tilt))
        projected_by_tilt[float(tilt)] = center_to_corner_2d_bottom_left(uv, image_dims_hw)

    per_point = {i: [] for i in range(n)}
    for tilt in sorted(coords_by_tilt.keys()):
        matched_tilt, min_diff = None, float("inf")
        for proj_tilt in projected_by_tilt:
            diff = abs(proj_tilt - tilt)
            if diff < 0.5 and diff < min_diff:
                matched_tilt, min_diff = proj_tilt, diff
        if matched_tilt is None:
            continue
        inputs = np.array([[x, y] for x, y, _ in coords_by_tilt[tilt]], dtype=np.float32)
        proj = projected_by_tilt[matched_tilt]
        if len(inputs) == 0 or len(proj) == 0:
            continue
        distances, closest = cKDTree(inputs).query(proj, k=1)
        proj_idx = np.arange(len(proj))
        keep = distances < 12.0
        distances, closest, proj_idx = distances[keep], closest[keep], proj_idx[keep]
        if len(distances) == 0:
            continue
        order = np.argsort(distances)
        distances, closest, proj_idx = distances[order], closest[order], proj_idx[order]
        assigned, chosen = set(), []
        for i in range(len(distances)):
            if closest[i] not in assigned:
                assigned.add(closest[i])
                chosen.append(i)
        if chosen:
            distances, proj_idx = distances[chosen], proj_idx[chosen]
        for i in range(len(distances)):
            per_point[int(proj_idx[i])].append(float(distances[i]))

    avg_dist = np.full(n, np.nan, dtype=float)
    avg_rmse = np.full(n, np.nan, dtype=float)
    n_matched = np.zeros(n, dtype=float)
    n_good = np.zeros(n, dtype=float)
    for i, d in per_point.items():
        if d:
            avg_dist[i] = float(np.mean(d))
            avg_rmse[i] = float(np.sqrt(np.mean(np.array(d) ** 2)))
            n_matched[i] = float(len(d))
            n_good[i] = float(sum(1 for v in d if v < 5.0))
    n_tilts_total = float(len(coords_by_tilt))
    if not np.isfinite(n_tilts_total) or n_tilts_total <= 0:
        n_tilts_total = float(max(1.0, np.nanmax(n_matched)))
    match_fraction = np.divide(
        n_good, np.maximum(n_matched, 1.0), out=np.zeros_like(n_good, dtype=float), where=np.isfinite(n_matched)
    )
    global_coverage = np.divide(
        n_good, np.maximum(n_tilts_total, 1.0), out=np.zeros_like(n_good, dtype=float), where=np.isfinite(n_good)
    )
    clean = {"nan": 0.0, "posinf": 0.0, "neginf": 0.0}
    return {
        "match_fraction": np.nan_to_num(match_fraction, **clean),
        "global_coverage": np.nan_to_num(global_coverage, **clean),
        "avg_dist_quality": np.nan_to_num(1.0 - _robust_unit_interval(avg_dist), **clean),
        "rmse_quality": np.nan_to_num(1.0 - _robust_unit_interval(avg_rmse), **clean),
    }


def _robust_unit_interval(x: np.ndarray) -> np.ndarray:
    """Map finite values to [0, 1] between their 5th and 95th percentiles."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=float)
    vals = x[finite]
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    out[finite] = np.clip((vals - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    return out


def figure_of_merit(
    points_corner: np.ndarray,
    coords_by_tilt: dict[float, np.ndarray],
    tilts_file_order: list[float],
    image_dims_hw: tuple[int, int],
    tomogram_dims_zyx: tuple[int, int, int],
    *,
    search_radius: float,
    affinity_temp: float = 1.0,
    tau: float = 0.2,
    rmse_scale: float = 5.9999,
    device=None,
    shifts_by_tilt: dict[float, tuple[float, float]] | None = None,
    match_distance: float | None = 12.0,
) -> dict:
    """Top-5 figure of merit and its components for a final set of points.

    Parameters
    ----------
    points_corner : numpy.ndarray
        ``(N, 3)`` points in the internal corner 3D frame (y up).
    coords_by_tilt : dict of float to numpy.ndarray
        Aligned-frame picks per tilt.
    tilts_file_order : list of float
        Tilt angles in stack order.
    image_dims_hw, tomogram_dims_zyx : tuple of int
    search_radius : float
        Radius (pixels) of the soft pick assignment used for ``conf``, ``breadth`` and the detector score.
    affinity_temp, tau, rmse_scale : float
        Assignment temperature, confidence threshold and residual scale of ``geom``.
    device : torch.device, optional
    shifts_by_tilt : dict of float to (float, float), optional
        Per-tilt shifts ``(dx, dy)`` (internal frame, y up) refined by bundle adjustment; every projection
        includes them. ``None`` scores against the input alignment.
    match_distance : float or None
        ``geom`` uses the residual over the tilts with a pick within this distance (:func:`matched_residuals`, the
        tilts bundle adjustment fits); ``None`` uses the RMS over all tilts (versions before 0.0.1).

    Returns
    -------
    dict
        ``fom`` (``(N,)`` in [0, 1]) and ``components`` (including ``cnn_raw``, the detector score).
    """
    x = np.asarray(points_corner, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return {"fom": np.zeros(0, dtype=float), "components": {}}
    features = _benchmark_features(
        x, coords_by_tilt, tilts_file_order, image_dims_hw, tomogram_dims_zyx, shifts_by_tilt
    )
    if match_distance is None:
        residuals = reprojection_residuals(
            x, coords_by_tilt, tilts_file_order, image_dims_hw, tomogram_dims_zyx, device, shifts_by_tilt
        )
    else:
        residuals, _ = matched_residuals(
            x,
            coords_by_tilt,
            tilts_file_order,
            image_dims_hw,
            tomogram_dims_zyx,
            match_distance,
            device,
            shifts_by_tilt,
        )
    family = assignment_components(
        x,
        residuals,
        coords_by_tilt,
        tilts_file_order,
        image_dims_hw,
        tomogram_dims_zyx,
        search_radius=search_radius,
        affinity_temp=affinity_temp,
        tau=tau,
        rmse_scale=rmse_scale,
        device=device,
        shifts_by_tilt=shifts_by_tilt,
    )
    stacked = np.column_stack(
        [
            features["match_fraction"],
            features["global_coverage"],
            features["avg_dist_quality"],
            features["rmse_quality"],
            np.asarray(family["geom"], dtype=float),
        ]
    )
    stacked = np.clip(np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    w = np.asarray(TOP5_FOM_WEIGHTS, dtype=float)
    w = w / float(np.sum(np.abs(w)))
    fom = np.clip(stacked @ w, 0.0, 1.0)
    components = dict(features)
    components.update(family)
    return {"fom": np.asarray(fom, dtype=float), "components": components}
