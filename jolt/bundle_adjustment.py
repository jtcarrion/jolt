"""Bundle adjustment: jointly refine 3D particle positions and per-tilt shifts (torch, LBFGS).

Two losses are available (``BundleAdjustmentSettings.loss``); both keep the quadratic penalties that hold the
shifts and the points near their starting values.

``evidence`` (default)
    MLESAC's per-view evidence (:func:`jolt.mlesac.mlesac_view_logratio`) evaluated on the moving parameters,
    with the nearest pick replaced by the sum over every pick within ``max_match_distance``::

        e[t, n] = (1 - gamma) + gamma * pi R^2 / (2 pi sigma^2) * sum_k exp(-|z[t, k] - h_t(X_n)|^2 / (2 sigma^2))
        loss    = -sum_{t, n} I_t(X_n) log e[t, n]  +  M (shift_reg mean|d - d0|^2 + point_reg mean|X - X0|^2)

    ``I_t`` is a soft in-image indicator (a product of sigmoids on the four image borders), so a particle inside
    the image with no pick nearby pays ``-log(1 - gamma)`` per tilt instead of leaving the loss, and ``M`` is the
    number of in-image particle-tilt pairs at the start (a constant that keeps the per-point gradients of order
    one for LBFGS's tolerances). The gradient is the responsibility-weighted residual: identical to the
    nearest-match gradient when one pick dominates, an average over ambiguous picks otherwise. The loss is
    smooth, so no match switching: with the same starting points perturbed by 0.001 px the refined shifts
    change by less than 0.04 px and the selected particles are identical (CZI). The neighborhoods (picks
    within the radius of each projection) are refreshed at every outer LBFGS step and are a fixed sparsity
    pattern inside a step.

``nearest`` (the previous behavior)
    At every evaluation each projected point is matched to its nearest 2D pick in that tilt, within
    ``max_match_distance``, with a Huber (smooth L1) loss on the residual length. The loss changes in steps
    whenever a match switches, so LBFGS uses a strong-Wolfe line search (every accepted step must decrease the
    loss) and stops when the loss no longer changes. Without the line search, nudging the starting points by
    0.001 px changed the solution of about 40% of the CZI tilt series, with occasional divergence (particles
    moved 6-14 px) and single-tilt shifts running away by 20-35 px; with it, no tilt series diverged and refined
    shifts changed by at most 0.34 px, but individual particles still moved by up to 16 px.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from jolt.geometry import (
    center_to_corner_3d,
    corner_to_center_3d,
    picks_to_center_tensor,
    project_points,
    tilt_projection_matrices,
)

#: Losses accepted by :class:`BundleAdjustmentSettings`.
LOSSES = ("evidence", "nearest")


@dataclass
class BundleAdjustmentSettings:
    """Parameters of the bundle adjustment (see the module docstring for the two losses)."""

    loss: str = "evidence"
    gamma: float = 0.7  # evidence: detection probability of a particle per tilt (MLESAC's gamma)
    sigma_px: float = 2.5  # evidence: pick localization scale; 2.0-2.9 px measured on CZI DS-10441
    border_width_px: float = 3.0  # evidence: width of the soft in-image indicator
    scale_by_m: bool = True  # evidence: multiply the loss by the number of in-image particle-tilt pairs
    shift_reg: float = 0.001
    point_reg: float = 0.0015
    max_steps: int = 20  # outer LBFGS steps; stops earlier once the loss no longer changes
    lbfgs_max_iter: int = 50  # iterations inside one LBFGS step
    lbfgs_history: int = 20
    lr: float = 1.0
    convergence_rtol: float = 1e-7
    tolerance_grad: float = 1e-6
    tolerance_change: float = 1e-9
    huber_delta: float = 1.0
    max_match_distance: float = 12.0
    freeze_shifts: bool = False
    fiducial_weight: float = 1.0  # weight of the gold-bead term (mean Huber loss per bead observation) vs the picks


@dataclass
class FiducialTracks:
    """Gold-bead observations with known identity across tilts (e.g. from an IMOD ``.fid``).

    Attributes
    ----------
    tilt_index : numpy.ndarray
        ``(K,)`` index of each observation's tilt in the bundle adjustment's ``tilt_angles``.
    bead_index : numpy.ndarray
        ``(K,)`` bead of each observation, ``0..B-1``.
    uv : numpy.ndarray
        ``(K, 2)`` observations in the internal centered 2D frame (aligned frame, y up).
    """

    tilt_index: np.ndarray
    bead_index: np.ndarray
    uv: np.ndarray

    @property
    def n_beads(self) -> int:
        """Number of beads."""
        return int(self.bead_index.max()) + 1 if len(self.bead_index) else 0


def triangulate_tracks(tracks: FiducialTracks, tilt_angles: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares 3D position of every bead from its observations (``u = cos(t) X + sin(t) Z``, ``v = Y``).

    Returns ``(B, 3)`` centered positions and the ``(K,)`` reprojection distance of every observation.
    """
    th = np.deg2rad(np.asarray(tilt_angles, dtype=float))[tracks.tilt_index]
    pts = np.zeros((tracks.n_beads, 3))
    dist = np.zeros(len(tracks.uv))
    for b in range(tracks.n_beads):
        sel = tracks.bead_index == b
        m = np.column_stack([np.cos(th[sel]), np.sin(th[sel])])
        xz, *_ = np.linalg.lstsq(m, tracks.uv[sel, 0], rcond=None)
        y = tracks.uv[sel, 1].mean()
        pts[b] = (xz[0], y, xz[1])
        dist[sel] = np.hypot(m @ xz - tracks.uv[sel, 0], y - tracks.uv[sel, 1])
    return pts, dist


def _nearest_pick_targets(proj_valid: torch.Tensor, picks: torch.Tensor, max_match_distance: float):
    """Nearest pick of every projected point, written as a one-neighbor softmax (as in the original code).

    Returns the matched pick positions (the projection itself where no pick is within ``max_match_distance``),
    the distance to the nearest pick, and a mask of points that have a pick within range.
    """
    distances = torch.cdist(proj_valid, picks, p=2)
    min_dists = torch.min(distances, dim=1).values
    k_eff = min(1, int(distances.shape[1]))
    if k_eff == int(distances.shape[1]):
        # Only one pick in the tilt: full softmax over that one column.
        logits = -distances / 1.0
        within = distances <= float(max_match_distance)
        no_match = ~within.any(dim=1)
        logits = logits.masked_fill(~within, -1e9)
        logits = torch.where(no_match.unsqueeze(1), torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1)
        if no_match.any():
            weights = torch.where(no_match.unsqueeze(1), torch.zeros_like(weights), weights)
        matched = torch.matmul(weights, picks)
        if no_match.any():
            matched = torch.where(no_match.unsqueeze(1), proj_valid, matched)
        return matched, min_dists, ~no_match
    d_top, idx_top = torch.topk(distances, k=k_eff, largest=False, dim=1)
    logits = -d_top / 1.0
    within = d_top <= float(max_match_distance)
    no_match = ~within.any(dim=1)
    logits = logits.masked_fill(~within, -1e9)
    logits = torch.where(no_match.unsqueeze(1), torch.zeros_like(logits), logits)
    weights = torch.softmax(logits, dim=1)
    if no_match.any():
        weights = torch.where(no_match.unsqueeze(1), torch.zeros_like(weights), weights)
    matched = torch.sum(weights.unsqueeze(-1) * picks[idx_top], dim=1)
    if no_match.any():
        matched = torch.where(no_match.unsqueeze(1), proj_valid, matched)
    return matched, min_dists, ~no_match


def _matched_residuals(projected_with_shifts, tilt_angles, picks_by_tilt, max_match_distance):
    """Residuals (projection minus matched pick) of every point that has a pick in range, over all tilts."""
    residuals, match_distances = [], []
    n_matches = 0
    n_no_match = 0
    for tilt_idx, tilt in enumerate(tilt_angles):
        picks = picks_by_tilt.get(tilt)
        if picks is None or len(picks) == 0:
            continue
        proj = projected_with_shifts[:, tilt_idx]
        valid = torch.isfinite(proj).all(dim=1)
        if not valid.any():
            continue
        proj_valid = proj[valid]
        matched, min_dists, has_match = _nearest_pick_targets(proj_valid, picks, max_match_distance)
        res = proj_valid - matched
        n_no_match += int((~has_match).sum().item())
        mask = (min_dists < max_match_distance) & has_match
        if not mask.any():
            continue
        residuals.append(res[mask])
        match_distances.append(min_dists[mask])
        n_matches += mask.sum().item()
    return residuals, match_distances, n_matches, n_no_match


def greedy_bundle_adjustment(
    points_corner: np.ndarray,
    coords_by_tilt: dict[float, np.ndarray],
    tilt_angles: list[float],
    shifts_init: np.ndarray,
    image_dims_hw: tuple[int, int],
    tomogram_dims_zyx: tuple[int, int, int],
    settings: BundleAdjustmentSettings | None = None,
    device: torch.device | None = None,
    log=print,
    fiducials: FiducialTracks | None = None,
) -> dict:
    """Jointly refine 3D points and per-tilt shifts against the 2D picks (and gold beads, if given).

    Parameters
    ----------
    points_corner : numpy.ndarray
        ``(N, 3)`` starting points in the corner 3D frame (pixels).
    coords_by_tilt : dict of float to numpy.ndarray
        Tilt angle to ``(K, 3)`` aligned-frame picks ``x, y, score`` (corner frame).
    tilt_angles : list of float
        Tilt angles (degrees) to use, in ascending order; ``shifts_init`` rows follow this order.
    shifts_init : numpy.ndarray
        ``(M, 2)`` starting shifts ``(dx, dy)`` in the internal 2D frame (pixels).
    image_dims_hw : tuple of int
        Aligned tilt-image height and width.
    tomogram_dims_zyx : tuple of int
        Tomogram dimensions.
    settings : BundleAdjustmentSettings, optional
    device : torch.device, optional
        Defaults to CUDA when available.
    log : callable
        Receives progress messages.
    fiducials : FiducialTracks, optional
        Gold beads with known tracks. They enter with fixed correspondences (no matching): their 3D positions are
        refined together with the shifts, and ``settings.fiducial_weight`` times their mean Huber loss is added.

    Returns
    -------
    dict
        ``X`` (refined corner-frame points), ``translations`` (refined shifts, ``(M, 2)``), ``rmse``,
        ``data_loss``, ``shift_reg_loss``, ``point_reg_loss``, ``loss_history``, ``steps``, ``converged``
        (the loss stopped changing before ``max_steps``) and ``match_statistics``; with ``fiducials`` also
        ``fiducials_X`` (refined bead positions, corner frame) and ``fiducial_distances`` (per observation, px).
    """
    s = settings or BundleAdjustmentSettings()
    if s.loss not in LOSSES:
        raise ValueError(f"unknown bundle-adjustment loss {s.loss!r}; expected one of {LOSSES}")
    if s.loss == "evidence":
        return evidence_bundle_adjustment(
            points_corner,
            coords_by_tilt,
            tilt_angles,
            shifts_init,
            image_dims_hw,
            tomogram_dims_zyx,
            s,
            device=device,
            log=log,
            fiducials=fiducials,
        )
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tilts = [float(t) for t in tilt_angles]
    x_center = corner_to_center_3d(np.asarray(points_corner, dtype=float).copy(), tomogram_dims_zyx)
    shifts0 = np.asarray(shifts_init, dtype=np.float32)
    projection = tilt_projection_matrices(tilts, device=device)
    if projection.shape[0] != len(tilts) or shifts0.shape[0] != len(tilts):
        raise ValueError(
            f"tilt count mismatch: {len(tilts)} tilts, {projection.shape[0]} matrices, {shifts0.shape[0]} shifts"
        )
    picks_by_tilt = {
        float(t): picks_to_center_tensor(c, image_dims_hw, device)
        for t, c in coords_by_tilt.items()
        if float(t) in tilts
    }

    points = torch.nn.Parameter(torch.as_tensor(x_center, dtype=torch.float32, device=device))
    points_init = torch.as_tensor(x_center, dtype=torch.float32, device=device)
    shifts_init_t = torch.as_tensor(shifts0, dtype=torch.float32, device=device)
    if s.freeze_shifts:
        shifts = shifts_init_t.clone()
        params = [points]
    else:
        shifts = torch.nn.Parameter(shifts_init_t.clone())
        params = [points, shifts]
    use_beads = fiducials is not None and len(fiducials.uv) > 0 and s.fiducial_weight > 0
    if use_beads:
        bead_init, _ = triangulate_tracks(fiducials, tilts)
        beads = torch.nn.Parameter(torch.as_tensor(bead_init, dtype=torch.float32, device=device))
        params.append(beads)
        bead_t = torch.as_tensor(fiducials.tilt_index, dtype=torch.long, device=device)
        bead_b = torch.as_tensor(fiducials.bead_index, dtype=torch.long, device=device)
        bead_uv = torch.as_tensor(fiducials.uv, dtype=torch.float32, device=device)
    optimizer = torch.optim.LBFGS(
        params,
        lr=s.lr,
        max_iter=s.lbfgs_max_iter,
        history_size=s.lbfgs_history,
        line_search_fn="strong_wolfe",
        tolerance_grad=s.tolerance_grad,
        tolerance_change=s.tolerance_change,
    )
    stats_history = []

    def closure():
        optimizer.zero_grad()
        projected = project_points(points, projection) + shifts.unsqueeze(0)
        residuals, match_distances, n_matches, _ = _matched_residuals(
            projected, tilts, picks_by_tilt, s.max_match_distance
        )
        if not residuals:
            return torch.tensor(1e6, dtype=torch.float32, device=device, requires_grad=True)
        res = torch.cat(residuals, dim=0)
        norms = torch.norm(res, dim=-1)
        data_loss = F.smooth_l1_loss(norms, torch.zeros_like(norms), beta=s.huber_delta, reduction="mean")
        shift_delta = shifts - shifts_init_t
        shift_loss = s.shift_reg * torch.mean(shift_delta * shift_delta)
        point_delta = points - points_init
        point_loss = s.point_reg * torch.mean(torch.norm(point_delta, dim=-1) ** 2)
        total = data_loss + (shift_loss + point_loss)
        if use_beads:
            bead_proj = project_points(beads, projection)[bead_b, bead_t] + shifts[bead_t]
            bead_norms = torch.norm(bead_proj - bead_uv, dim=-1)
            total = total + s.fiducial_weight * F.smooth_l1_loss(
                bead_norms, torch.zeros_like(bead_norms), beta=s.huber_delta, reduction="mean"
            )
        with torch.no_grad():
            stats_history.append({"n_matches": n_matches, "rmse": torch.sqrt(torch.mean(norms**2)).item()})
        total.backward()
        return total

    log(
        f"[bundle adjustment] {len(x_center)} points, {len(tilts)} tilts, max match distance "
        f"{s.max_match_distance} px, shift_reg {s.shift_reg}, point_reg {s.point_reg}"
        + (f", {fiducials.n_beads} gold beads (weight {s.fiducial_weight})" if use_beads else "")
    )
    loss_history = []
    previous = None
    converged = False
    for step in range(s.max_steps):
        loss = optimizer.step(closure)
        loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        loss_history.append(loss_val)
        last = stats_history[-1] if stats_history else {}
        log(
            f"  step {step:3d}: loss={loss_val:.6f} matches={last.get('n_matches', 0)} "
            f"RMSE={last.get('rmse', float('inf')):.3f} px"
        )
        if previous is not None and abs(previous - loss_val) <= s.convergence_rtol * max(1.0, abs(previous)):
            converged = True
            break
        previous = loss_val

    with torch.no_grad():
        projected = project_points(points, projection) + shifts.unsqueeze(0)
        residuals, match_distances, n_matches, n_no_match = _matched_residuals(
            projected, tilts, picks_by_tilt, s.max_match_distance
        )
        if residuals:
            norms = torch.norm(torch.cat(residuals, dim=0), dim=-1)
            rmse = torch.sqrt(torch.mean(norms**2)).item()
            data_loss = F.smooth_l1_loss(norms, torch.zeros_like(norms), beta=s.huber_delta, reduction="mean").item()
        else:
            rmse = float("inf")
            data_loss = float("inf")
        shift_loss = s.shift_reg * torch.mean((shifts - shifts_init_t) ** 2).item()
        point_loss = s.point_reg * torch.mean(torch.norm(points - points_init, dim=-1) ** 2).item()
        x_refined = center_to_corner_3d(points.detach().cpu().numpy(), tomogram_dims_zyx)
        translations = shifts.detach().cpu().numpy()
        bead_out = {}
        if use_beads:
            bead_proj = project_points(beads, projection)[bead_b, bead_t] + shifts[bead_t]
            bead_out = {
                "fiducials_X": center_to_corner_3d(beads.detach().cpu().numpy(), tomogram_dims_zyx),
                "fiducial_distances": torch.norm(bead_proj - bead_uv, dim=-1).cpu().numpy(),
            }
    all_d = torch.cat(match_distances, dim=0) if match_distances else None
    return {
        "loss": "nearest",
        "X": x_refined,
        "translations": translations,
        "loss_history": loss_history,
        "steps": len(loss_history),
        "converged": converged,
        "rmse": rmse,
        "data_loss": data_loss,
        "shift_reg_loss": shift_loss,
        "point_reg_loss": point_loss,
        "match_statistics": {
            "n_matches": n_matches,
            "n_no_match_state": n_no_match,
            "mean_match_distance": all_d.mean().item() if all_d is not None else float("inf"),
            "max_match_distance": all_d.max().item() if all_d is not None else float("inf"),
        },
        **bead_out,
    }


def soft_inside(projected: torch.Tensor, image_dims_hw: tuple[int, int], width_px: float) -> torch.Tensor:
    """Soft in-image indicator in ``[0, 1]`` of projections ``(..., 2)`` in the internal centered frame.

    A product of four sigmoids, one per image border, each of width ``width_px``: 1 well inside the image,
    0.5 on a border, 0 well outside.
    """
    h, w = image_dims_hw
    u_half, v_half = (w - 1) / 2.0, (h - 1) / 2.0
    u, v = projected[..., 0], projected[..., 1]
    return (
        torch.sigmoid((u_half - u) / width_px)
        * torch.sigmoid((u + u_half) / width_px)
        * torch.sigmoid((v_half - v) / width_px)
        * torch.sigmoid((v + v_half) / width_px)
    )


def pick_neighborhoods(
    projected: np.ndarray, picks_by_tilt: list[np.ndarray | None], radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index every pick within ``radius`` of every projection.

    Parameters
    ----------
    projected : numpy.ndarray
        ``(N, M, 2)`` projections (internal centered frame) of ``N`` points in ``M`` tilts.
    picks_by_tilt : list
        One ``(K_t, 2)`` array of centered picks per tilt (``None`` or empty for a tilt without picks).
    radius : float

    Returns
    -------
    point_index, tilt_index, pick_index : numpy.ndarray
        One entry per (point, tilt, pick) triple, ``pick_index`` local to its tilt.
    """
    n_idx, t_idx, k_idx = [], [], []
    for t, picks in enumerate(picks_by_tilt):
        if picks is None or len(picks) == 0:
            continue
        for n, ks in enumerate(cKDTree(picks).query_ball_point(projected[:, t, :], r=float(radius))):
            if ks:
                n_idx.append(np.full(len(ks), n, dtype=np.int64))
                t_idx.append(np.full(len(ks), t, dtype=np.int64))
                k_idx.append(np.asarray(ks, dtype=np.int64))
    if not n_idx:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy()
    return np.concatenate(n_idx), np.concatenate(t_idx), np.concatenate(k_idx)


def evidence_terms(
    projected: torch.Tensor,
    picks_flat: torch.Tensor,
    point_index: torch.Tensor,
    tilt_index: torch.Tensor,
    pick_flat_index: torch.Tensor,
    *,
    sigma_px: float,
    gamma: float,
    radius_px: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(point, tilt) evidence ``e[t, n]`` and the inlier share of it (the module docstring's formula).

    Parameters
    ----------
    projected : torch.Tensor
        ``(N, M, 2)`` projections including the per-tilt shifts.
    picks_flat : torch.Tensor
        ``(sum_t K_t, 2)`` picks of all tilts concatenated in tilt order.
    point_index, tilt_index, pick_flat_index : torch.Tensor
        One (point, tilt, pick) triple per neighbor, the pick index into ``picks_flat``.

    Returns
    -------
    evidence : torch.Tensor
        ``(N, M)``; ``1 - gamma`` where a point has no pick within the radius.
    inlier_share : torch.Tensor
        ``(N, M)`` fraction of the evidence carried by the picks (the soft "matched" mass).
    """
    n_pts, m_tilts = projected.shape[0], projected.shape[1]
    g = float(np.clip(gamma, 1e-6, 1.0 - 1e-6))
    peak = g * np.pi * float(radius_px) ** 2 / (2.0 * np.pi * float(sigma_px) ** 2)
    ssum = torch.zeros(n_pts * m_tilts, dtype=projected.dtype, device=projected.device)
    if point_index.numel():
        diff = projected[point_index, tilt_index] - picks_flat[pick_flat_index]
        kernel = torch.exp(-(diff * diff).sum(dim=1) / (2.0 * float(sigma_px) ** 2))
        ssum.index_add_(0, point_index * m_tilts + tilt_index, kernel)
    ssum = ssum.view(n_pts, m_tilts)
    evidence = (1.0 - g) + peak * ssum
    return evidence, peak * ssum / evidence


def evidence_bundle_adjustment(
    points_corner: np.ndarray,
    coords_by_tilt: dict[float, np.ndarray],
    tilt_angles: list[float],
    shifts_init: np.ndarray,
    image_dims_hw: tuple[int, int],
    tomogram_dims_zyx: tuple[int, int, int],
    settings: BundleAdjustmentSettings | None = None,
    device: torch.device | None = None,
    log=print,
    fiducials: FiducialTracks | None = None,
) -> dict:
    """Refine 3D points and per-tilt shifts with the ``evidence`` loss (see the module docstring).

    Same parameters and result keys as :func:`greedy_bundle_adjustment`, plus ``soft_support`` (per point, the
    sum over tilts of the inlier share of its evidence: the expected number of tilts in which it is detected)
    and ``log_evidence`` (per point, the sum over tilts of ``I_t log e[t, n]``). ``rmse`` and
    ``match_statistics`` are computed from the nearest pick within the radius, as for the ``nearest`` loss, so
    the two are comparable. Each outer step refreshes the neighborhoods from the current projections and runs
    a fresh LBFGS (its curvature history would not survive the refresh).
    """
    s = settings or BundleAdjustmentSettings()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tilts = [float(t) for t in tilt_angles]
    n_pts, m_tilts = int(np.asarray(points_corner).shape[0]), len(tilts)
    x_center = corner_to_center_3d(np.asarray(points_corner, dtype=float).copy(), tomogram_dims_zyx)
    shifts0 = np.asarray(shifts_init, dtype=np.float32)
    projection = tilt_projection_matrices(tilts, device=device)
    if projection.shape[0] != m_tilts or shifts0.shape[0] != m_tilts:
        raise ValueError(
            f"tilt count mismatch: {m_tilts} tilts, {projection.shape[0]} matrices, {shifts0.shape[0]} shifts"
        )
    picks_t = [
        picks_to_center_tensor(coords_by_tilt[t], image_dims_hw, device)
        if t in coords_by_tilt and len(coords_by_tilt[t])
        else None
        for t in tilts
    ]
    picks_np = [p.cpu().numpy() if p is not None else None for p in picks_t]
    offsets = np.zeros(m_tilts + 1, dtype=np.int64)
    for t, p in enumerate(picks_np):
        offsets[t + 1] = offsets[t] + (0 if p is None else len(p))
    picks_flat = (
        torch.cat([p for p in picks_t if p is not None], dim=0)
        if offsets[-1] > 0
        else torch.zeros((0, 2), dtype=torch.float32, device=device)
    )

    points = torch.nn.Parameter(torch.as_tensor(x_center, dtype=torch.float32, device=device))
    points_init = points.detach().clone()
    shifts_init_t = torch.as_tensor(shifts0, dtype=torch.float32, device=device)
    if s.freeze_shifts:
        shifts = shifts_init_t.clone()
        params = [points]
    else:
        shifts = torch.nn.Parameter(shifts_init_t.clone())
        params = [points, shifts]
    use_beads = fiducials is not None and len(fiducials.uv) > 0 and s.fiducial_weight > 0
    if use_beads:
        bead_init, _ = triangulate_tracks(fiducials, tilts)
        beads = torch.nn.Parameter(torch.as_tensor(bead_init, dtype=torch.float32, device=device))
        params.append(beads)
        bead_t = torch.as_tensor(fiducials.tilt_index, dtype=torch.long, device=device)
        bead_b = torch.as_tensor(fiducials.bead_index, dtype=torch.long, device=device)
        bead_uv = torch.as_tensor(fiducials.uv, dtype=torch.float32, device=device)

    def project() -> torch.Tensor:
        return project_points(points, projection) + shifts.unsqueeze(0)

    with torch.no_grad():
        m_norm = float(soft_inside(project(), image_dims_hw, s.border_width_px).sum().clamp(min=1.0))
    scale = m_norm if s.scale_by_m else 1.0
    stats: dict = {}

    def make_closure(neighborhoods):
        point_index = torch.as_tensor(neighborhoods[0], device=device)
        tilt_index = torch.as_tensor(neighborhoods[1], device=device)
        pick_flat_index = torch.as_tensor(offsets[neighborhoods[1]] + neighborhoods[2], device=device)

        def closure():
            optimizer.zero_grad()
            projected = project()
            inside = soft_inside(projected, image_dims_hw, s.border_width_px)
            evidence, inlier_share = evidence_terms(
                projected,
                picks_flat,
                point_index,
                tilt_index,
                pick_flat_index,
                sigma_px=s.sigma_px,
                gamma=s.gamma,
                radius_px=s.max_match_distance,
            )
            log_evidence = inside * torch.log(evidence)
            data_loss = -log_evidence.sum() / m_norm
            shift_delta = shifts - shifts_init_t
            shift_loss = s.shift_reg * torch.mean(shift_delta * shift_delta)
            point_delta = points - points_init
            point_loss = s.point_reg * torch.mean(torch.norm(point_delta, dim=-1) ** 2)
            total = data_loss + (shift_loss + point_loss)
            if use_beads:
                bead_proj = project_points(beads, projection)[bead_b, bead_t] + shifts[bead_t]
                bead_norms = torch.norm(bead_proj - bead_uv, dim=-1)
                total = total + s.fiducial_weight * F.smooth_l1_loss(
                    bead_norms, torch.zeros_like(bead_norms), beta=s.huber_delta, reduction="mean"
                )
            total = total * scale
            with torch.no_grad():
                stats["data_loss"] = float(data_loss)
                stats["soft_support"] = (inside * inlier_share).sum(dim=1)
                stats["log_evidence"] = log_evidence.sum(dim=1)
                stats["n_neighbors"] = int(point_index.numel())
            total.backward()
            return total

        return closure

    log(
        f"[bundle adjustment] evidence loss: {n_pts} points, {m_tilts} tilts, radius {s.max_match_distance} px, "
        f"sigma {s.sigma_px} px, gamma {s.gamma}, shift_reg {s.shift_reg}, point_reg {s.point_reg}"
        + (f", {fiducials.n_beads} gold beads (weight {s.fiducial_weight})" if use_beads else "")
    )
    loss_history: list[float] = []
    previous = None
    converged = False
    for step in range(s.max_steps):
        with torch.no_grad():
            neighborhoods = pick_neighborhoods(project().cpu().numpy(), picks_np, s.max_match_distance)
        optimizer = torch.optim.LBFGS(
            params,
            lr=s.lr,
            max_iter=s.lbfgs_max_iter,
            history_size=s.lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=s.tolerance_grad,
            tolerance_change=s.tolerance_change,
        )
        loss = optimizer.step(make_closure(neighborhoods))
        loss_val = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        loss_history.append(loss_val)
        log(
            f"  step {step:3d}: loss={loss_val / scale:.6f} neighbors={stats.get('n_neighbors', 0)} "
            f"data={stats.get('data_loss', float('nan')):.6f}"
        )
        if previous is not None and abs(previous - loss_val) <= s.convergence_rtol * max(1.0, abs(previous)):
            converged = True
            break
        previous = loss_val

    with torch.no_grad():
        projected_np = project().cpu().numpy()
        # Nearest-pick distances within the radius, for the same rmse and match statistics as the nearest loss.
        d_all = []
        for t, p in enumerate(picks_np):
            if p is None:
                continue
            d, _ = cKDTree(p).query(projected_np[:, t, :], k=1, distance_upper_bound=float(s.max_match_distance))
            d_all.append(d[np.isfinite(d)])
        d_all = np.concatenate(d_all) if d_all else np.zeros(0)
        n_matches = int(d_all.size)
        rmse = float(np.sqrt(np.mean(d_all**2))) if n_matches else float("inf")
        shift_loss = s.shift_reg * torch.mean((shifts - shifts_init_t) ** 2).item()
        point_loss = s.point_reg * torch.mean(torch.norm(points - points_init, dim=-1) ** 2).item()
        x_refined = center_to_corner_3d(points.detach().cpu().numpy(), tomogram_dims_zyx)
        translations = shifts.detach().cpu().numpy()
        bead_out = {}
        if use_beads:
            bead_proj = project_points(beads, projection)[bead_b, bead_t] + shifts[bead_t]
            bead_out = {
                "fiducials_X": center_to_corner_3d(beads.detach().cpu().numpy(), tomogram_dims_zyx),
                "fiducial_distances": torch.norm(bead_proj - bead_uv, dim=-1).cpu().numpy(),
            }
    return {
        "loss": "evidence",
        "sigma_px": float(s.sigma_px),
        "gamma": float(s.gamma),
        "X": x_refined,
        "translations": translations,
        "loss_history": loss_history,
        "steps": len(loss_history),
        "converged": converged,
        "rmse": rmse,
        "data_loss": float(stats.get("data_loss", float("nan"))),
        "shift_reg_loss": shift_loss,
        "point_reg_loss": point_loss,
        "soft_support": stats["soft_support"].cpu().numpy() if "soft_support" in stats else np.zeros(n_pts),
        "log_evidence": stats["log_evidence"].cpu().numpy() if "log_evidence" in stats else np.zeros(n_pts),
        "match_statistics": {
            "n_matches": n_matches,
            "n_no_match_state": int(n_pts * m_tilts - n_matches),
            "mean_match_distance": float(d_all.mean()) if n_matches else float("inf"),
            "max_match_distance": float(d_all.max()) if n_matches else float("inf"),
        },
        **bead_out,
    }
