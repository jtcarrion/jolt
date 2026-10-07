"""Bundle adjustment: jointly refine 3D particle positions and per-tilt shifts (torch, LBFGS).

Correspondences are not fixed: at every evaluation each projected point is matched to its nearest 2D pick in
that tilt, within ``max_match_distance``. The loss is a Huber (smooth L1) loss on the residual length, plus
quadratic penalties that keep the shifts and the points near their starting values.

The loss changes in steps whenever a match switches, so LBFGS uses a strong-Wolfe line search (every accepted
step must decrease the loss) and stops when the loss no longer changes, instead of taking a fixed number of
steps. Without the line search, nudging the starting points by 0.001 px changed the solution of about 40% of
the CZI tilt series, with occasional divergence (particles moved 6-14 px) and single-tilt shifts running away by
20-35 px; with it, no tilt series diverged and refined shifts changed by at most 0.34 px.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from jolt.geometry import (
    center_to_corner_3d,
    corner_to_center_3d,
    picks_to_center_tensor,
    project_points,
    tilt_projection_matrices,
)


@dataclass
class BundleAdjustmentSettings:
    """Parameters of the greedy bundle adjustment."""

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
) -> dict:
    """Jointly refine 3D points and per-tilt shifts against the 2D picks.

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

    Returns
    -------
    dict
        ``X`` (refined corner-frame points), ``translations`` (refined shifts, ``(M, 2)``), ``rmse``,
        ``data_loss``, ``shift_reg_loss``, ``point_reg_loss``, ``loss_history``, ``steps``, ``converged``
        (the loss stopped changing before ``max_steps``) and ``match_statistics``.
    """
    s = settings or BundleAdjustmentSettings()
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
        with torch.no_grad():
            stats_history.append({"n_matches": n_matches, "rmse": torch.sqrt(torch.mean(norms**2)).item()})
        total.backward()
        return total

    log(
        f"[bundle adjustment] {len(x_center)} points, {len(tilts)} tilts, max match distance "
        f"{s.max_match_distance} px, shift_reg {s.shift_reg}, point_reg {s.point_reg}"
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
    all_d = torch.cat(match_distances, dim=0) if match_distances else None
    return {
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
    }
