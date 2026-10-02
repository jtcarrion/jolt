"""Build 3D particle hypotheses from 2D picks with MLESAC.

The initializer never matches picks between neighboring tilts. It builds explicit 3D hypotheses from
minimal multi-view samples and asks which 2D picks across the whole tilt series are likely under each one.

Geometry (internal centered frame, single tilt axis along y)
-----------------------------------------------------------
A centered 3D point ``P = (X, Y, Z)`` projects to tilt ``theta`` as ``u = cos(theta) X + sin(theta) Z + dx`` and
``v = Y + dy``. Two consequences are used throughout:

1. ``v`` does not depend on ``theta``, so a particle's height is the same in every tilt. This exact constraint
   is the main gate for pairing picks (:func:`_y_gate_join`).
2. Two views with ``sin(theta_1 - theta_0) != 0`` determine ``(X, Z)`` exactly (:func:`solve_two_view_xz`).
   The conditioning of that solve is ``|cot((theta_1 - theta_0)/2)|``, which ``min_pair_baseline_deg`` bounds.

Every tilt serves as a base view; there is no anchor tilt and no angular weighting.

Scoring
-------
MLESAC (Torr and Zisserman 2000) scores a hypothesis with a Gaussian-inlier plus uniform-outlier mixture. For a
matched view at distance ``d`` and outlier area ``A_out = pi R^2``::

    s_i = log(gamma N(d; sigma) A_out + (1 - gamma))      matched view
    s_i = log(1 - gamma)                                  view inside the image but no pick within R
    S   = sum_i s_i                                       (higher is better)

Charging in-image misses makes a coincidental hypothesis (matched in a few tilts, absent from the rest) score
below a real particle, which is present in every tilt where it projects inside the image.

The picks passed in must already be in the aligned frame; the per-tilt shifts are zero at this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.spatial import cKDTree

from jolt.geometry import corner_to_center_2d_bottom_left, project_y_axis

_EPS = 1e-9

#: Seed of the random generator used when the seed hypotheses must be subsampled.
DEFAULT_SEED = 39


@dataclass
class MlesacSettings:
    """Parameters of the MLESAC initializer (the values JOLT has been validated with)."""

    sigma_px: float = 4.0
    search_radius_px: float = 12.0
    gamma: float = 0.7
    y_gate_px: float = 8.0
    min_pair_baseline_deg: float = 20.0
    max_pair_baseline_deg: float = 180.0
    max_partner_tilts_per_base: int = 4
    max_partner_picks_per_base: int = 6
    n_validation_tilts: int = 2
    min_validation_hits: int = 1
    seed_reproj_gate_px: float = 10.0
    seed_merge_radius_3d: float = 4.0
    max_seeds: int = 300000
    min_support: int = 5
    min_angular_span_deg: float = 20.0
    min_branch_support: int = 0
    max_condition: float = 1000.0
    min_logratio: float = 0.0
    merge_radius_3d: float = 8.0
    max_obs_overlap: float = 0.5
    huber_delta_sigmas: float = 2.0
    n_irls: int = 3
    seed: int = DEFAULT_SEED


@dataclass
class ObservationTable:
    """2D picks of one tilt series in the internal centered frame, indexed by tilt."""

    tilts: list[float]  # ascending; only tilts with at least one pick
    uv_by_tilt: dict[float, np.ndarray]  # (n_t, 2) centered picks
    y_by_tilt: dict[float, np.ndarray]  # (n_t,) tilt-invariant height v - dy
    gid_by_tilt: dict[float, np.ndarray]  # (n_t,) global pick ids
    trans_by_tilt: dict[float, np.ndarray]  # (2,) per-tilt shift (dx, dy)
    trees: dict[float, cKDTree]  # KD-tree over uv_by_tilt
    node_mapping: list[tuple[float, int]]  # global id -> (tilt, local index)
    image_dims_hw: tuple[int, int]
    n_observations: int

    def half_extent(self) -> tuple[float, float]:
        """Half width and half height of the tilt image in the centered frame."""
        h, w = self.image_dims_hw
        return (w - 1) / 2.0, (h - 1) / 2.0


@dataclass
class MlesacResult:
    """Accepted 3D hypotheses and the picks each one claimed."""

    X_3d: np.ndarray  # (N, 3) centered 3D points
    trajectories: list[set[int]]  # global pick ids per accepted point
    node_mapping: list[tuple[float, int]]
    metrics: list[dict]
    stage_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ScoredHypotheses:
    """Per-hypothesis assignment and score arrays (``N`` hypotheses, ``T`` tilts)."""

    xyz: np.ndarray  # (N, 3)
    assign: np.ndarray  # (N, T) local pick index per tilt, -1 if none
    dist: np.ndarray  # (N, T) distance to the assigned pick, +inf if none
    in_bounds: np.ndarray  # (N, T) projection inside the image
    logratio: np.ndarray  # (N,) MLESAC log-likelihood ratio
    support: np.ndarray  # (N,) matched views
    n_in_bounds: np.ndarray  # (N,)
    angular_span: np.ndarray  # (N,) degrees between the extreme matched tilts
    branch_balance: np.ndarray  # (N,) min(matched positive tilts, matched negative tilts)
    condition: np.ndarray  # (N,) condition number of the (X, Z) least-squares solve
    rmse: np.ndarray  # (N,) RMS distance over matched views
    nll: np.ndarray  # (N,) matched-views-only mixture negative log-likelihood (diagnostic)
    min_margin: np.ndarray  # (N,) smallest inlier-versus-outlier log margin over matched views

    def subset(self, mask: np.ndarray) -> ScoredHypotheses:
        """Return the hypotheses selected by a boolean mask."""
        return ScoredHypotheses(**{name: getattr(self, name)[mask] for name in self.__dataclass_fields__})


def build_observation_table(
    coords_by_tilt: dict[float, np.ndarray],
    image_dims_hw: tuple[int, int],
    trans_by_tilt: dict[float, tuple[float, float]] | None = None,
) -> ObservationTable:
    """Convert aligned-frame STAR picks into the internal centered frame.

    Parameters
    ----------
    coords_by_tilt : dict of float to numpy.ndarray
        Tilt angle to ``(n, 3)`` picks ``x, y, score`` (corner frame, aligned image).
    image_dims_hw : tuple of int
        Aligned tilt-image height and width.
    trans_by_tilt : dict, optional
        Per-tilt shift ``(dx, dy)``; zero when omitted.

    Returns
    -------
    ObservationTable
        Global pick ids number every pick tilt by tilt in ascending tilt order.
    """
    tilts: list[float] = []
    uv_by_tilt, y_by_tilt, gid_by_tilt, trans, trees = {}, {}, {}, {}, {}
    node_mapping: list[tuple[float, int]] = []
    gid = 0
    for tilt in sorted(float(t) for t in coords_by_tilt.keys()):
        picks = coords_by_tilt[tilt]
        raw = []
        for local_idx, pick in enumerate(picks):
            node_mapping.append((tilt, local_idx))
            raw.append((float(pick[0]), float(pick[1])))
        gid_base = gid
        gid += len(picks)
        if not raw:
            continue
        uv = corner_to_center_2d_bottom_left(np.asarray(raw, dtype=float), image_dims_hw)
        dx, dy = (trans_by_tilt or {}).get(tilt, (0.0, 0.0))
        dx, dy = float(dx), float(dy)
        tilts.append(tilt)
        uv_by_tilt[tilt] = uv
        y_by_tilt[tilt] = uv[:, 1] - dy
        gid_by_tilt[tilt] = np.arange(gid_base, gid_base + len(raw), dtype=np.int64)
        trans[tilt] = np.asarray([dx, dy], dtype=float)
        trees[tilt] = cKDTree(uv)
    return ObservationTable(
        tilts=tilts,
        uv_by_tilt=uv_by_tilt,
        y_by_tilt=y_by_tilt,
        gid_by_tilt=gid_by_tilt,
        trans_by_tilt=trans,
        trees=trees,
        node_mapping=node_mapping,
        image_dims_hw=(int(image_dims_hw[0]), int(image_dims_hw[1])),
        n_observations=sum(len(v) for v in uv_by_tilt.values()),
    )


def solve_two_view_xz(
    u0: np.ndarray, theta0_deg: float, u1: np.ndarray, theta1_deg: float, dx0: float = 0.0, dx1: float = 0.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve ``(X, Z)`` exactly from two views of the same point.

    Returns ``X``, ``Z`` and the condition number ``|cot((theta_1 - theta_0)/2)|`` of the solve.

    Raises
    ------
    ValueError
        If the two tilts are (nearly) the same, which leaves ``Z`` unobservable.
    """
    t0, t1 = np.deg2rad(float(theta0_deg)), np.deg2rad(float(theta1_deg))
    c0, s0 = np.cos(t0), np.sin(t0)
    c1, s1 = np.cos(t1), np.sin(t1)
    det = c0 * s1 - s0 * c1  # sin(theta_1 - theta_0)
    if abs(det) < 1e-6:
        raise ValueError(f"degenerate two-view baseline: theta_0={theta0_deg}, theta_1={theta1_deg}")
    b0 = np.asarray(u0, dtype=float) - float(dx0)
    b1 = np.asarray(u1, dtype=float) - float(dx1)
    X = (b0 * s1 - b1 * s0) / det
    Z = (b1 * c0 - b0 * c1) / det
    half = 0.5 * (t1 - t0)
    condition = float(abs(np.cos(half) / max(abs(np.sin(half)), _EPS)))
    return X, Z, condition


def solve_y_axis_ls(
    thetas_deg: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted least-squares 3D point from many views, batched over hypotheses.

    All inputs are ``(N, T)``; a zero weight means the view is absent. ``(X, Z)`` solve the 2x2 normal
    equations of ``u - dx = cos(theta) X + sin(theta) Z``; ``Y`` is the weighted mean of ``v - dy``.

    Returns
    -------
    xyz : torch.Tensor
        ``(N, 3)``; NaN where the system is singular.
    condition : torch.Tensor
        ``(N,)`` condition number of the 2x2 system (inf if singular).
    """
    th = torch.deg2rad(thetas_deg)
    c, s = torch.cos(th), torch.sin(th)
    ut = u - dx
    vt = v - dy
    w = weights
    w_sum = w.sum(dim=1)
    a = (w * c * c).sum(dim=1)
    b = (w * c * s).sum(dim=1)
    d = (w * s * s).sum(dim=1)
    rhs_x = (w * c * ut).sum(dim=1)
    rhs_z = (w * s * ut).sum(dim=1)
    det = a * d - b * b
    safe = det.abs() > 1e-12
    det_safe = torch.where(safe, det, torch.ones_like(det))
    X = torch.where(safe, (d * rhs_x - b * rhs_z) / det_safe, torch.full_like(det, float("nan")))
    Z = torch.where(safe, (a * rhs_z - b * rhs_x) / det_safe, torch.full_like(det, float("nan")))
    Y = torch.where(
        w_sum > _EPS,
        (w * vt).sum(dim=1) / torch.clamp(w_sum, min=_EPS),
        torch.full_like(w_sum, float("nan")),
    )
    tr, diff = a + d, 0.5 * (a - d)
    disc = torch.sqrt(torch.clamp(diff * diff + b * b, min=0.0))
    lam_max = 0.5 * tr + disc
    lam_min = 0.5 * tr - disc
    condition = torch.where(
        lam_min > 1e-12,
        torch.sqrt(torch.clamp(lam_max / torch.clamp(lam_min, min=1e-12), min=1.0)),
        torch.full_like(lam_max, float("inf")),
    )
    return torch.stack([X, Y, Z], dim=1), condition


def mlesac_view_logratio(
    d: torch.Tensor, matched: torch.Tensor, in_bounds: torch.Tensor, *, sigma_px: float, gamma: float, outlier_area_px2
) -> torch.Tensor:
    """Per-view MLESAC log-likelihood ratio against the all-outlier null (see the module docstring).

    ``d``, ``matched`` and ``in_bounds`` are ``(N, T)``; views outside the image contribute zero.
    """
    sigma2 = float(sigma_px) ** 2
    g = float(np.clip(gamma, 1e-6, 1.0 - 1e-6))
    a_out = float(outlier_area_px2)
    peak = g * (a_out / (2.0 * np.pi * sigma2))
    d_safe = torch.where(matched, d, torch.zeros_like(d))
    inlier_term = peak * torch.exp(-(d_safe * d_safe) / (2.0 * sigma2))
    s_matched = torch.log(inlier_term + (1.0 - g))
    s_miss = torch.full_like(d, float(np.log(1.0 - g)))
    s = torch.where(matched, s_matched, s_miss)
    return torch.where(in_bounds, s, torch.zeros_like(s))


def _select_partner_tilts(
    base_tilt: float, tilts: list[float], *, min_baseline_deg: float, max_baseline_deg: float, max_partners: int
) -> list[float]:
    """Partner tilts with an adequate angular baseline, spread evenly across the eligible range."""
    eligible = [t for t in tilts if min_baseline_deg - 1e-6 <= abs(t - base_tilt) <= max_baseline_deg + 1e-6]
    if len(eligible) <= max_partners:
        return eligible
    eligible.sort(key=lambda t: abs(t - base_tilt))
    idx = np.linspace(0, len(eligible) - 1, max_partners)
    return [eligible[int(round(i))] for i in idx]


def _y_gate_join(
    y_base: np.ndarray, y_partner: np.ndarray, *, y_gate_px: float, max_partners_per_base: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pair base picks with partner picks of nearly the same height (at most ``max_partners_per_base`` each).

    Returns index arrays ``(i_base, j_partner)``.
    """
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
    if y_base.size == 0 or y_partner.size == 0:
        return empty
    order = np.argsort(y_partner, kind="stable")
    y_sorted = y_partner[order]
    lo = np.searchsorted(y_sorted, y_base - y_gate_px, side="left")
    hi = np.searchsorted(y_sorted, y_base + y_gate_px, side="right")
    keep = (hi - lo) > 0
    if not np.any(keep):
        return empty
    base_idx, partner_idx = [], []
    for i in np.nonzero(keep)[0]:
        cand = order[lo[i] : hi[i]]
        if cand.size > max_partners_per_base:
            dy = np.abs(y_partner[cand] - y_base[i])
            cand = cand[np.argsort(dy, kind="stable")[:max_partners_per_base]]
        base_idx.append(np.full(cand.size, i, dtype=np.int64))
        partner_idx.append(cand.astype(np.int64))
    return np.concatenate(base_idx), np.concatenate(partner_idx)


def _validation_tilts(base: float, partner: float, tilts: list[float], *, n_validation: int) -> list[float]:
    """Return the tilts farthest from the seed pair's mean angle, used to reject accidental pairs."""
    others = [t for t in tilts if abs(t - base) > 1e-6 and abs(t - partner) > 1e-6]
    if not others:
        return []
    mid = 0.5 * (base + partner)
    others.sort(key=lambda t: -abs(t - mid))
    return others[:n_validation]


def _grid_dedup(xyz: np.ndarray, radius: float) -> np.ndarray:
    """Return indices of one representative seed per voxel of size ``radius`` (in input order)."""
    if xyz.shape[0] == 0 or radius <= 0:
        return np.arange(xyz.shape[0], dtype=np.int64)
    cells = np.floor(xyz / float(radius)).astype(np.int64)
    _, keep = np.unique(cells, axis=0, return_index=True)
    return np.sort(keep)


def generate_seed_hypotheses(
    table: ObservationTable, settings: MlesacSettings, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, int]]:
    """Triangulate seed hypotheses from height-gated pick pairs and validate them in independent tilts."""
    s = settings
    counts = {
        "n_observations": int(table.n_observations),
        "n_seed_pairs_after_y_gate": 0,
        "n_seed_triplets_triangulated": 0,
        "n_seed_triplets_after_validation": 0,
        "n_seeds_after_dedup": 0,
    }
    tilts = table.tilts
    counts["n_base_tilts"] = len(tilts)
    seeds: list[np.ndarray] = []
    n_seeds = 0
    for base in tilts:
        if n_seeds >= s.max_seeds:
            break
        partners = _select_partner_tilts(
            base,
            tilts,
            min_baseline_deg=s.min_pair_baseline_deg,
            max_baseline_deg=s.max_pair_baseline_deg,
            max_partners=s.max_partner_tilts_per_base,
        )
        if not partners:
            continue
        uv_b = table.uv_by_tilt[base]
        y_b = table.y_by_tilt[base]
        dx_b = float(table.trans_by_tilt[base][0])
        for partner in partners:
            if n_seeds >= s.max_seeds:
                break
            uv_p = table.uv_by_tilt[partner]
            y_p = table.y_by_tilt[partner]
            dx_p = float(table.trans_by_tilt[partner][0])
            i_b, j_p = _y_gate_join(y_b, y_p, y_gate_px=s.y_gate_px, max_partners_per_base=s.max_partner_picks_per_base)
            if i_b.size == 0:
                continue
            counts["n_seed_pairs_after_y_gate"] += int(i_b.size)
            X, Z, _ = solve_two_view_xz(uv_b[i_b, 0], base, uv_p[j_p, 0], partner, dx0=dx_b, dx1=dx_p)
            Y = 0.5 * (y_b[i_b] + y_p[j_p])
            cand = np.stack([X, Y, Z], axis=1)
            cand = cand[np.isfinite(cand).all(axis=1)]
            if cand.shape[0] == 0:
                continue
            counts["n_seed_triplets_triangulated"] += int(cand.shape[0])
            val_tilts = _validation_tilts(base, partner, tilts, n_validation=s.n_validation_tilts)
            if val_tilts and s.min_validation_hits > 0:
                hits = np.zeros(cand.shape[0], dtype=np.int64)
                for vt in val_tilts:
                    proj = project_y_axis(cand, vt, tuple(table.trans_by_tilt[vt]))
                    dist, _ = table.trees[vt].query(proj, k=1, distance_upper_bound=float(s.seed_reproj_gate_px))
                    hits += np.isfinite(dist).astype(np.int64)
                cand = cand[hits >= int(s.min_validation_hits)]
                if cand.shape[0] == 0:
                    continue
            counts["n_seed_triplets_after_validation"] += int(cand.shape[0])
            seeds.append(cand)
            n_seeds += int(cand.shape[0])
    if not seeds:
        return np.empty((0, 3), dtype=float), counts
    all_seeds = np.concatenate(seeds, axis=0)
    if all_seeds.shape[0] > s.max_seeds:
        sel = rng.choice(all_seeds.shape[0], size=s.max_seeds, replace=False)
        all_seeds = all_seeds[np.sort(sel)]
    all_seeds = all_seeds[_grid_dedup(all_seeds, s.seed_merge_radius_3d)]
    counts["n_seeds_after_dedup"] = int(all_seeds.shape[0])
    return all_seeds, counts


def _assign_and_score(
    xyz: np.ndarray, table: ObservationTable, settings: MlesacSettings, device: torch.device
) -> ScoredHypotheses:
    """Assign each hypothesis its nearest pick (within the search radius) in every tilt and score it."""
    s = settings
    n = xyz.shape[0]
    tilts = table.tilts
    t = len(tilts)
    assign = np.full((n, t), -1, dtype=np.int64)
    dist = np.full((n, t), np.inf, dtype=float)
    in_bounds = np.zeros((n, t), dtype=bool)
    u_obs = np.zeros((n, t), dtype=float)
    v_obs = np.zeros((n, t), dtype=float)
    u_half, v_half = table.half_extent()
    for k, tilt in enumerate(tilts):
        proj = project_y_axis(xyz, tilt, tuple(table.trans_by_tilt[tilt]))
        in_bounds[:, k] = (np.abs(proj[:, 0]) <= u_half) & (np.abs(proj[:, 1]) <= v_half)
        d, j = table.trees[tilt].query(proj, k=1, distance_upper_bound=float(s.search_radius_px))
        hit = np.isfinite(d)
        if np.any(hit):
            jj = j[hit]
            assign[hit, k] = jj
            dist[hit, k] = d[hit]
            uv = table.uv_by_tilt[tilt][jj]
            u_obs[hit, k] = uv[:, 0]
            v_obs[hit, k] = uv[:, 1]
    matched = assign >= 0

    d_t = torch.as_tensor(dist, dtype=torch.float32, device=device)
    m_t = torch.as_tensor(matched, device=device)
    b_t = torch.as_tensor(in_bounds, device=device)
    a_out = float(np.pi * s.search_radius_px**2)
    view_scores = mlesac_view_logratio(d_t, m_t, b_t, sigma_px=s.sigma_px, gamma=s.gamma, outlier_area_px2=a_out)
    logratio = view_scores.sum(dim=1).cpu().numpy()

    theta = np.asarray(tilts, dtype=float)[None, :]
    support = matched.sum(axis=1).astype(np.int64)
    with np.errstate(invalid="ignore"):
        th_masked = np.where(matched, theta, np.nan)
        t_max = np.nanmax(th_masked, axis=1)
        t_min = np.nanmin(th_masked, axis=1)
    angular_span = np.nan_to_num(np.where(support > 0, t_max - t_min, 0.0), nan=0.0)
    n_pos = (matched & (theta > 1e-6)).sum(axis=1)
    n_neg = (matched & (theta < -1e-6)).sum(axis=1)
    branch_balance = np.minimum(n_pos, n_neg).astype(np.int64)
    d_matched = np.where(matched, dist, 0.0)
    rmse = np.sqrt((d_matched**2).sum(axis=1) / np.maximum(support, 1))
    rmse = np.where(support > 0, rmse, np.inf)

    sigma2 = float(s.sigma_px) ** 2
    g = float(np.clip(s.gamma, 1e-6, 1.0 - 1e-6))
    log_in = -np.log(2.0 * np.pi * sigma2) - np.where(matched, dist, 0.0) ** 2 / (2.0 * sigma2)
    log_out = -np.log(a_out)
    log_mix = np.logaddexp(np.log(g) + log_in, np.log(1.0 - g) + log_out)
    nll = -np.where(matched, log_mix, 0.0).sum(axis=1)
    margin_all = (np.log(g) + log_in) - (np.log(1.0 - g) + log_out)
    min_margin = np.where(support > 0, np.min(np.where(matched, margin_all, np.inf), axis=1), -np.inf)

    th_t = torch.as_tensor(np.repeat(theta, n, axis=0), dtype=torch.float32, device=device)
    w_t = torch.as_tensor(matched.astype(np.float32), device=device)
    dxs = np.asarray([table.trans_by_tilt[tt][0] for tt in tilts], dtype=float)[None, :]
    dys = np.asarray([table.trans_by_tilt[tt][1] for tt in tilts], dtype=float)[None, :]
    _, cond_t = solve_y_axis_ls(
        th_t,
        torch.as_tensor(u_obs, dtype=torch.float32, device=device),
        torch.as_tensor(v_obs, dtype=torch.float32, device=device),
        torch.as_tensor(np.repeat(dxs, n, axis=0), dtype=torch.float32, device=device),
        torch.as_tensor(np.repeat(dys, n, axis=0), dtype=torch.float32, device=device),
        w_t,
    )
    return ScoredHypotheses(
        xyz=xyz,
        assign=assign,
        dist=dist,
        in_bounds=in_bounds,
        logratio=logratio,
        support=support,
        n_in_bounds=in_bounds.sum(axis=1).astype(np.int64),
        angular_span=angular_span,
        branch_balance=branch_balance,
        condition=cond_t.cpu().numpy(),
        rmse=rmse,
        nll=nll,
        min_margin=min_margin,
    )


def _refit(scored: ScoredHypotheses, table: ObservationTable, settings: MlesacSettings, device: torch.device):
    """Refit each hypothesis to its assigned picks with a Huber-weighted (IRLS) least-squares solve."""
    n, t = scored.assign.shape
    if n == 0:
        return scored.xyz.copy()
    tilts = table.tilts
    u_obs = np.zeros((n, t), dtype=float)
    v_obs = np.zeros((n, t), dtype=float)
    for k, tilt in enumerate(tilts):
        hit = scored.assign[:, k] >= 0
        if np.any(hit):
            uv = table.uv_by_tilt[tilt][scored.assign[hit, k]]
            u_obs[hit, k] = uv[:, 0]
            v_obs[hit, k] = uv[:, 1]
    theta = np.asarray(tilts, dtype=float)[None, :]
    dxs = np.asarray([table.trans_by_tilt[tt][0] for tt in tilts], dtype=float)[None, :]
    dys = np.asarray([table.trans_by_tilt[tt][1] for tt in tilts], dtype=float)[None, :]
    th_t = torch.as_tensor(np.repeat(theta, n, axis=0), dtype=torch.float32, device=device)
    u_t = torch.as_tensor(u_obs, dtype=torch.float32, device=device)
    v_t = torch.as_tensor(v_obs, dtype=torch.float32, device=device)
    dx_t = torch.as_tensor(np.repeat(dxs, n, axis=0), dtype=torch.float32, device=device)
    dy_t = torch.as_tensor(np.repeat(dys, n, axis=0), dtype=torch.float32, device=device)
    base_w = torch.as_tensor((scored.assign >= 0).astype(np.float32), device=device)
    w = base_w.clone()
    delta = float(settings.huber_delta_sigmas) * float(settings.sigma_px)
    xyz_t = torch.as_tensor(scored.xyz, dtype=torch.float32, device=device)
    for _ in range(max(1, int(settings.n_irls))):
        xyz_t, _ = solve_y_axis_ls(th_t, u_t, v_t, dx_t, dy_t, w)
        bad = ~torch.isfinite(xyz_t).all(dim=1)
        if bad.any():
            xyz_t[bad] = torch.as_tensor(scored.xyz, dtype=torch.float32, device=device)[bad]
        th_r = torch.deg2rad(th_t)
        pu = torch.cos(th_r) * xyz_t[:, 0:1] + torch.sin(th_r) * xyz_t[:, 2:3] + dx_t
        pv = xyz_t[:, 1:2] + dy_t
        r = torch.sqrt((u_t - pu) ** 2 + (v_t - pv) ** 2 + 1e-12)
        huber = torch.where(r <= delta, torch.ones_like(r), delta / torch.clamp(r, min=_EPS))
        w = base_w * huber
    return xyz_t.cpu().numpy().astype(float)


def _select_candidates(
    scored: ScoredHypotheses, table: ObservationTable, settings: MlesacSettings
) -> tuple[list[int], list[set[int]], dict[str, int]]:
    """Geometry and likelihood gates, then a greedy pass in score order with observation-overlap and 3D vetoes."""
    s = settings
    counts = {"n_candidates_scored": int(scored.xyz.shape[0])}
    geom_ok = (
        (scored.support >= int(s.min_support))
        & (scored.angular_span >= float(s.min_angular_span_deg))
        & (scored.branch_balance >= int(s.min_branch_support))
        & (scored.condition <= float(s.max_condition))
        & np.isfinite(scored.xyz).all(axis=1)
    )
    counts["n_candidates_after_geometry_gate"] = int(geom_ok.sum())
    idx = np.nonzero(geom_ok & (scored.logratio > float(s.min_logratio)))[0]
    counts["n_candidates_after_likelihood_gate"] = int(idx.size)
    if idx.size == 0:
        counts["n_candidates_after_dedup"] = 0
        counts["n_warmstart_trajectories"] = 0
        return [], [], counts
    order = idx[np.lexsort((scored.rmse[idx], -scored.support[idx], -scored.logratio[idx]))]
    used: set[int] = set()
    accepted: list[int] = []
    accepted_xyz: list[np.ndarray] = []
    trajectories: list[set[int]] = []
    for i in order:
        gids: set[int] = set()
        for k, tilt in enumerate(table.tilts):
            j = scored.assign[i, k]
            if j >= 0:
                gids.add(int(table.gid_by_tilt[tilt][j]))
        if not gids:
            continue
        if len(gids & used) / float(len(gids)) > float(s.max_obs_overlap):
            continue
        if accepted_xyz and s.merge_radius_3d > 0:
            d = np.linalg.norm(np.asarray(accepted_xyz) - scored.xyz[i][None, :], axis=1)
            if np.min(d) < float(s.merge_radius_3d):
                continue
        accepted.append(int(i))
        accepted_xyz.append(scored.xyz[i])
        trajectories.append(gids)
        used |= gids
    counts["n_candidates_after_dedup"] = len(accepted)
    counts["n_warmstart_trajectories"] = len(trajectories)
    return accepted, trajectories, counts


def run_mlesac(
    coords_by_tilt: dict[float, np.ndarray],
    image_dims_hw: tuple[int, int],
    settings: MlesacSettings | None = None,
    device: torch.device | None = None,
    log=print,
) -> MlesacResult:
    """Build MLESAC-ranked 3D points and the picks each one claims.

    Parameters
    ----------
    coords_by_tilt : dict of float to numpy.ndarray
        Tilt angle to ``(n, 3)`` aligned-frame picks ``x, y, score``.
    image_dims_hw : tuple of int
        Aligned tilt-image height and width in pixels.
    settings : MlesacSettings, optional
        Parameters (defaults: :class:`MlesacSettings`).
    device : torch.device, optional
        Defaults to CUDA when available.
    log : callable
        Receives progress messages.

    Returns
    -------
    MlesacResult
        Accepted points in the centered 3D frame, in acceptance order.
    """
    s = settings or MlesacSettings()
    rng = np.random.default_rng(s.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = build_observation_table(coords_by_tilt, image_dims_hw)

    def empty(counts):
        return MlesacResult(np.empty((0, 3), dtype=float), [], table.node_mapping, [], counts)

    if table.n_observations == 0 or len(table.tilts) < 3:
        log(f"[mlesac] too few observations ({table.n_observations}) or tilts ({len(table.tilts)}); no points")
        return empty({"n_observations": int(table.n_observations)})
    log(
        f"[mlesac] {table.n_observations} picks over {len(table.tilts)} tilts "
        f"({min(table.tilts):+.1f} to {max(table.tilts):+.1f} deg); sigma={s.sigma_px:.1f}px "
        f"R={s.search_radius_px:.1f}px gamma={s.gamma:.2f}"
    )
    seeds, seed_counts = generate_seed_hypotheses(table, s, rng)
    log(
        f"[mlesac] seeds: height-gated pairs={seed_counts['n_seed_pairs_after_y_gate']} "
        f"-> triangulated={seed_counts['n_seed_triplets_triangulated']} "
        f"-> validated={seed_counts['n_seed_triplets_after_validation']} "
        f"-> deduplicated={seed_counts['n_seeds_after_dedup']}"
    )
    if seeds.shape[0] == 0:
        log("[mlesac] no seed hypothesis survived validation")
        return empty(seed_counts)
    scored = _assign_and_score(seeds, table, s, device)
    prefilter = scored.support >= max(2, int(s.min_support) - 2)
    seed_counts["n_candidates_before_refit"] = int(prefilter.sum())
    if not np.any(prefilter):
        log("[mlesac] no hypothesis reached the minimum support")
        return empty(seed_counts)
    refit_xyz = _refit(scored.subset(prefilter), table, s, device)
    scored = _assign_and_score(refit_xyz, table, s, device)
    accepted, trajectories, sel_counts = _select_candidates(scored, table, s)
    stage_counts = {**seed_counts, **sel_counts}
    log(
        f"[mlesac] scored {stage_counts['n_candidates_scored']} -> geometry gate "
        f"{stage_counts['n_candidates_after_geometry_gate']} -> likelihood gate "
        f"{stage_counts['n_candidates_after_likelihood_gate']} -> {len(accepted)} accepted"
    )
    if not accepted:
        return empty(stage_counts)
    metrics = [
        {
            "candidate_id": int(rank),
            "x_center": float(scored.xyz[i, 0]),
            "y_center": float(scored.xyz[i, 1]),
            "z_center": float(scored.xyz[i, 2]),
            "support": int(scored.support[i]),
            "n_in_bounds": int(scored.n_in_bounds[i]),
            "angular_span": float(scored.angular_span[i]),
            "branch_balance": int(scored.branch_balance[i]),
            "triangulation_condition": float(scored.condition[i]),
            "mlesac_logratio": float(scored.logratio[i]),
            "mlesac_nll": float(scored.nll[i]),
            "min_mlesac_margin": float(scored.min_margin[i]),
            "robust_reproj_rmse": float(scored.rmse[i]),
        }
        for rank, i in enumerate(accepted)
    ]
    return MlesacResult(scored.xyz[accepted], trajectories, table.node_mapping, metrics, stage_counts)
